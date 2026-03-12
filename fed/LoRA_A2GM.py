# federated layerwise selection
from utils.utils import cosine_learning_rate,get_model_and_tokenizer,draw_loss_curve, setup_logger,get_peft_config
from utils.data import get_dataset, split_dataset, SFTDataset, get_dataset_this_round,get_classification_dataset,ClassificationDataset
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
import torch
from torch.utils.data import DataLoader
import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy.stats import entropy
import numpy as np
from .FedBase import FedBase

def compute_weight_stats(param_dict):
    stat_dict = {}
    for name, param in param_dict.items():
        norm = torch.norm(param,1).cpu().item()#/torch.norm(global_dict[name],1).cpu().item()
        stat_dict[name] = norm
    return stat_dict

def fixed_sample_layers(stat_dict, ritio = 0.3):
    norms = []
    lora_keys = []
    for name, value in stat_dict.items():
        if "lora" in name:
            norms.append(value)
            lora_keys.append(name)
    
    norms = np.array(norms)
    idx = (ritio) * len(norms)
    sort_norms = sorted(norms)
    min_score = sort_norms[int(idx)]
    mask_dict = {}
    for i, name in zip(range(len(norms)),lora_keys):
        if norms[i] < min_score:
            mask_dict[name] = 0
        else:
            mask_dict[name] = 1
    
    return mask_dict


class LoRA_A2GM(FedBase):
    def __init__(self, args):
        super(LoRA_A2GM,self).__init__(args)
        self.ex_mask_dict_A = {}
        self.ex_mask_dict_B = {}
    def run(self):
        args = self.args
        logger = setup_logger(args.fed_alg, f"./logs/{args.fed_alg}/{args.dataset_name.replace('/','_')}.txt")
        # init model and tokenizer
        model, tokenizer = get_model_and_tokenizer(args)
        peft_config = get_peft_config(args)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        model.config.use_cache = False
        model.cuda()
        # print(model.base_model_torch_dtype)
        # set up the global and local models
        global_dict = copy.deepcopy(get_peft_model_state_dict(model))
        # local_dict_list = [copy.deepcopy(global_dict) for i in range(args.num_clients)]
        # set up datasets
        if args.task == "classification":
            train_dataset, test_dataset = get_classification_dataset(args.dataset_name, args.dataset_sample)
            # test_dataset = ClassificationDataset(
            #     args.dataset_name,
            #     test_dataset, 
            #     tokenizer, 
            # )
            # test_dataloder = DataLoader(test_dataset, batch_size=128)
            # accuracy = self.eval_model(model, test_dataloder)
            # print(f"Initial Test Accuracy: {accuracy}")
        elif args.task == "sft":
            train_dataset = get_dataset(args.dataset_name, args.dataset_sample)
        
        local_datasets = split_dataset(args, train_dataset)

        rounds_loss = []
        mask_dict = {}
        for key in global_dict.keys():
            if "lora_B" in key:
                mask_dict[key] = 1
                self.ex_mask_dict_B[key] = 1
            elif "lora_A" in key:
                mask_dict[key] = 0
                self.ex_mask_dict_A[key] = 1
        for r in range(1,args.num_rounds+1):
            sample_num_list = []
            participants = np.random.choice(range(args.num_clients), args.num_clients_per_round, replace=False)
            new_lr = cosine_learning_rate(r, args.num_rounds, args.lr, 1e-6) 
            round_loss = []
            local_dict_list = []
            for client_id in participants:
                print(f">> ==================== Round {r} : {client_id} ====================")
                # send the global model to the client
                if args.task == "sft":
                    set_peft_model_state_dict(model, global_dict)
                else:
                    set_peft_model_state_dict(model, copy.deepcopy(global_dict))# fast fix bug here

                if mask_dict:
                    for name, param in model.named_parameters():
                        if "lora" in name:
                            if mask_dict[name.replace("default.weight","weight")] == 1:
                                param.requires_grad = True
                            else:
                                param.requires_grad = False

                model.print_trainable_parameters()
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

                sample_num_list.append(len(dataset_this_round))
                local_dataloader = DataLoader(dataset_this_round, batch_size=args.batch_size, shuffle=True)
                # recieve the local model
                model, loss = self.local_train(model, local_dataloader, new_lr, args)
                # Save the local model state
                local_dict_list.append(copy.deepcopy(get_peft_model_state_dict(model)))
                round_loss.append(loss)
                torch.cuda.empty_cache()
            # Aggregate the local models to update the global model
            mask_dict = self.aggerate_local_models(local_dict_list, global_dict, sample_num_list,r+1)
            rounds_loss.append(sum(round_loss)/len(round_loss))
            torch.cuda.empty_cache()
            # if args.task == "classification":
            #     set_peft_model_state_dict(model, copy.deepcopy(global_dict))
            #     accuracy = self.eval_model(model, test_dataloder)
            #     logger.info(f"Round {r+1} test Accuracy: {accuracy}")

        model.save_pretrained(f"./output/{args.fed_alg}/{args.dataset_name.replace('/','_')+str(args.seed)+'_'+str(args.beta)}")
        tokenizer.save_pretrained(f"./output/{args.fed_alg}/{args.dataset_name.replace('/','_')+str(args.seed)+'_'+str(args.beta)}")
        logger.info("loss data:")
        logger.info(rounds_loss)
    
    def aggerate_local_models(self, local_dict_list, global_dict, sample_num_list,num_round):
        total_samples = sum(sample_num_list)
        avg_dict = {}
        for key in global_dict.keys():
            avg_dict[key] = sum([local_dict_list[idx][key] * sample_num_list[idx] / total_samples for idx, d in enumerate(local_dict_list)])
        
        if (num_round-1) % 2 == 0 and num_round > 1:
            delta_dict = {}
            for key in global_dict.keys():
                if "lora_A" in key:
                    delta_dict[key] = global_dict[key] - avg_dict[key]
            stat_dict = compute_weight_stats(delta_dict)
            ex_mask_dict = fixed_sample_layers(stat_dict, ritio=self.args.c + (num_round//20)*self.args.beta) 
            self.ex_mask_dict_A = ex_mask_dict

        elif (num_round-1) % 2 == 1 and num_round > 2:
            delta_dict = {}
            for key in global_dict.keys():
                if "lora_B" in key:
                    delta_dict[key] = avg_dict[key] - global_dict[key]
            stat_dict = compute_weight_stats(delta_dict)    
            ex_mask_dict = fixed_sample_layers(stat_dict, ritio=0.8)
            self.ex_mask_dict_B = ex_mask_dict
        with torch.no_grad():
            for key in global_dict.keys():
                global_dict[key] = avg_dict[key]
        
        mask_dict = {}
        for key in global_dict.keys():
            if num_round % 2 == 1:
                if "lora_B" in key:
                    mask_dict[key] = 1
                    if self.ex_mask_dict_B[key] == 0:
                        mask_dict[key] = 0
                elif "lora_A" in key:
                    mask_dict[key] = 0
            else:
                if "lora_B" in key:
                    mask_dict[key] = 0
                elif "lora_A" in key:
                    mask_dict[key] = 1
                    if self.ex_mask_dict_A[key] == 0:
                        mask_dict[key] = 0
        return mask_dict
