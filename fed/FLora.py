# stack matrix
from utils.utils import cosine_learning_rate, get_model_and_tokenizer, draw_loss_curve, setup_logger, get_peft_config
from utils.data import get_dataset, split_dataset, SFTDataset, get_dataset_this_round,get_classification_dataset,ClassificationDataset
from peft import get_peft_model, get_peft_model_state_dict
import torch
from torch.utils.data import DataLoader
import numpy as np
import copy
from datetime import datetime
from .FedBase import FedBase
class FLora(FedBase):
    def __init__(self, args):
        super(FLora,self).__init__(args)
    
    def run(self):
        args = self.args
        logger = setup_logger(args.fed_alg, f"./logs/{args.fed_alg}/{args.dataset_name.replace('/','_')}.txt")
        # init model and tokenizer
        model, tokenizer = get_model_and_tokenizer(args)
        peft_config = get_peft_config(args)

        model.config.use_cache = False
        # set up scale for lora update
        self.lora_scale = args.peft_lora_alpha / args.peft_lora_r
        # set up datasets
        if args.task == "classification":
            train_dataset,test_dataset = get_classification_dataset(args.dataset_name, args.dataset_sample)
            # test_dataset = ClassificationDataset(
            #     args.dataset_name,
            #     test_dataset, 
            #     tokenizer, 
            # )
            # test_dataloder = DataLoader(test_dataset, batch_size=64)
            # accuracy = self.eval_model(model, test_dataloder)
            # print(f"Initial Test Accuracy: {accuracy}")
        elif args.task == "sft":
            train_dataset = get_dataset(args.dataset_name, args.dataset_sample)
        
        local_datasets = split_dataset(args, train_dataset)

        rounds_loss = []
        for r in range(args.num_rounds):
            print(f"Round {r + 1}/{args.num_rounds}")

            sample_num_list = []
            participants = np.random.choice(range(args.num_clients), args.num_clients_per_round, replace=False)
            new_lr = cosine_learning_rate(r, args.num_rounds, args.lr, 1e-6) 
            round_loss = []
            local_dict_list = []
            for client_id in participants:
                print(f">> ==================== Round {r+1} : {client_id} ====================")
                # send the global model to the client
                client_model = copy.deepcopy(model)
                client_model = get_peft_model(client_model, peft_config)
                client_model.cuda()
                # get dataloader this round
                if args.task == "sft":
                    dataset_this_round = get_dataset_this_round(local_datasets[client_id], r, args)
                    dataset_this_round = SFTDataset(dataset_this_round, 
                                                    tokenizer, 
                                                    template_name=args.template, 
                                                    max_len=args.seq_len, 
                                                    math_reason=True if "math" in args.dataset_name else False)
                elif args.task == "classification":
                    dataset_this_round = get_dataset_this_round(local_datasets[client_id], r, args)
                    dataset_this_round = ClassificationDataset(
                        args.dataset_name,
                        dataset_this_round, 
                        tokenizer, 
                    )
                model.print_trainable_parameters()
                sample_num_list.append(len(dataset_this_round))
                local_dataloader = DataLoader(dataset_this_round, batch_size=args.batch_size, shuffle=True)
                # recieve the local model
                client_model, loss = self.local_train(client_model, local_dataloader, new_lr, args)
                # Save the local model state
                local_dict_list.append(copy.deepcopy(get_peft_model_state_dict(client_model)))
                round_loss.append(loss)
                # del client_model
                torch.cuda.empty_cache()  # 清理GPU缓存
            
            # Aggregate the local models to update the global model
            model = self.aggerate_local_models(model, local_dict_list, sample_num_list)

            avg_loss = sum(round_loss)/len(round_loss)
            rounds_loss.append(avg_loss)
            # if args.task == "classification":
                # accuracy = self.eval_model(model, test_dataloder)
                # logger.info(f"Round {r+1} test Accuracy: {accuracy}")

        model.save_pretrained(f"./output/{args.fed_alg}/{args.dataset_name.replace('/','_')}")
        tokenizer.save_pretrained(f"./output/{args.fed_alg}/{args.dataset_name.replace('/','_')}")
        logger.info("loss data:")
        logger.info(rounds_loss)
        # draw_loss_curve(range(args.num_rounds), rounds_loss, args)
    
    
    def aggerate_local_models(self, model, local_dict_list, sample_num_list):
        """
            FLoRA stacking
        """
        model.cuda()
        total_samples = sum(sample_num_list)

        # deal with classifier
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "classifier" in name:
                    local_key = "base_model.model." + name
                    param.data = sum([local_dict[local_key] * sample_num_list[idx] / total_samples 
                                        for idx, local_dict in enumerate(local_dict_list)])
        # 初始化聚合后的 A、B
        global_A, global_B = {}, {}

        with torch.no_grad():
            for key in local_dict_list[0].keys():
                if "lora_A" in key:
                    stacked_A = []
                    for state, n in zip(local_dict_list, sample_num_list):
                        pk = n / total_samples
                        stacked_A.append(state[key] * pk)  # 对 A 加权
                    global_A[key] = torch.cat(stacked_A, dim=0)
                elif "lora_B" in key:
                    stacked_B = []
                    for state, n in zip(local_dict_list, sample_num_list):
                        stacked_B.append(state[key])  # 只能对一个加权
                    global_B[key] = torch.cat(stacked_B, dim=1)

        # 根据 A、B 计算 ΔW = B @ A，并更新 base model 参数 
        with torch.no_grad():
            for name, param in model.named_parameters():
                lora_a_key = "base_model.model." + name.replace(".weight", ".lora_A.weight")
                lora_b_key = "base_model.model." + name.replace(".weight", ".lora_B.weight")
                
                if lora_a_key in global_A and lora_b_key in global_B:
                    # print(param.data.dtype)
                    A = global_A[lora_a_key]
                    B = global_B[lora_b_key]
                    delta_w = self.lora_scale * torch.matmul(B,A)
                    param.data += delta_w
        return model