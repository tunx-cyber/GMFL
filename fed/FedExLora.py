from utils.utils import cosine_learning_rate,get_model_and_tokenizer,draw_loss_curve, setup_logger,get_peft_config
from utils.data import get_dataset, split_dataset, SFTDataset, get_dataset_this_round, get_classification_dataset, ClassificationDataset
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
import torch
from torch.utils.data import DataLoader
import numpy as np
import copy
from datetime import datetime
from .FedBase import FedBase
class FedEx(FedBase):
    def __init__(self, args):
        super(FedEx,self).__init__(args)

    def run(self):
        args = self.args
        logger = setup_logger(args.fed_alg, f"./logs/{args.fed_alg}/{args.dataset_name.replace('/','_')}.txt")
        # init model and tokenizer
        model, tokenizer = get_model_and_tokenizer(args)
        peft_config = get_peft_config(args)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        model.config.use_cache = False
        self.lora_scale = args.peft_lora_alpha / args.peft_lora_r
        model.cuda()
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
            # test_dataloder = DataLoader(test_dataset, batch_size=64)
            # accuracy = self.eval_model(model, test_dataloder)
            # print(f"Initial Test Accuracy: {accuracy}")
        elif args.task == "sft":
            train_dataset = get_dataset(args.dataset_name, args.dataset_sample)
        
        local_datasets = split_dataset(args, train_dataset)

        rounds_loss = []
        for r in range(args.num_rounds):
            sample_num_list = []
            participants = np.random.choice(range(args.num_clients), args.num_clients_per_round, replace=False)
            new_lr = cosine_learning_rate(r, args.num_rounds, args.lr, 1e-6)
            round_loss = []
            local_dict_list = []
            for client_id in participants:
                print(f">> ==================== Round {r+1} : {client_id} ====================")
                # send the global model to the client
                if args.task == "sft":
                    set_peft_model_state_dict(model, global_dict)
                else:
                    set_peft_model_state_dict(model, copy.deepcopy(global_dict))# fast fix bug here
                
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
                model, loss = self.local_train(model, local_dataloader, new_lr, args)
                # Save the local model state
                local_dict_list.append(copy.deepcopy(get_peft_model_state_dict(model)))
                round_loss.append(loss)
                torch.cuda.empty_cache()
            # Aggregate the local models to update the global model
            global_dict = self.aggerate_local_models(model, local_dict_list)
            rounds_loss.append(sum(round_loss)/len(round_loss))

            # if args.task == "classification":
            #     set_peft_model_state_dict(model, copy.deepcopy(global_dict))
            #     accuracy = self.eval_model(model, test_dataloder)
            #     logger.info(f"Round {r+1} test Accuracy: {accuracy}")
                
        model.save_pretrained(f"./output/{args.fed_alg}/{args.dataset_name.replace('/','_')+str(args.seed)}")
        tokenizer.save_pretrained(f"./output/{args.fed_alg}/{args.dataset_name.replace('/','_')+str(args.seed)}")
        logger.info("loss data:")
        logger.info(rounds_loss)
        # draw_loss_curve(range(args.num_rounds),rounds_loss,args)
    
    
    def aggerate_local_models(self, global_model, local_dict_list):
        # print(local_dict_list[0].keys())
        global_dict = global_model.state_dict()
        
        for k in global_dict.keys():
            if "classifier.modules_to_save" in k:
                local_key = k.replace(".modules_to_save.default","")
                global_dict[k] = torch.stack(
                    [local_dict_list[i][local_key].float() for i in range(len(local_dict_list))], 0
                ).mean(0)

        for name, module in global_model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                lora_A_keys = name + ".lora_A.weight"
                lora_B_keys = name + ".lora_B.weight"
                base_layer_keys = name + ".base_layer.weight"

                lora_A_weights = torch.stack(
                    [client_dict[lora_A_keys].detach() for client_dict in local_dict_list]
                )
                lora_B_weights = torch.stack(
                    [client_dict[lora_B_keys].detach() for client_dict in local_dict_list]
                )

                # M shape: (d, k)
                M = sum(
                    lora_B_weights[i] @ lora_A_weights[i] for i in range(len(local_dict_list))
                ) / len(local_dict_list)
                
                lora_A_avg = lora_A_weights.mean(0)
                lora_B_avg = lora_B_weights.mean(0)

                scaling_factor = self.lora_scale

                residue = M - lora_B_avg @ lora_A_avg
                # U, S, V = torch.svd_lowrank(residue, q=self.args.peft_lora_r, niter=2)
                # matrix1 = U
                # matrix2 = torch.diag(S) @ V.T
                # residue = matrix1 @ matrix2
                
                global_dict[name + ".lora_A.default.weight"] = lora_A_avg
                global_dict[name + ".lora_B.default.weight"] = lora_B_avg
                global_dict[name + ".base_layer.weight"] += residue * scaling_factor
                

        global_model.load_state_dict(global_dict)
        return get_peft_model_state_dict(global_model)
            
# reference implementation of FedEx
def aggregate_models_fedex(global_model, client_dicts, args):
    printer = 0 
    global_dict = global_model.state_dict()

    for k in global_dict.keys():
        if "classifier" in k:
            global_dict[k] = torch.stack(
                [client_dicts[i][k].float() for i in range(len(client_dicts))], 0
            ).mean(0)

    for client_dict in client_dicts:
        for k in global_dict.keys():
            if "classifier" in k:
                client_dict[k] = global_dict[k]

    for name, module in global_model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            lora_A_keys = name + ".lora_A.default.weight"
            lora_B_keys = name + ".lora_B.default.weight"
            base_layer_keys = name + ".base_layer.weight"

            lora_A_weights = torch.stack(
                [client_dict[lora_A_keys].detach() for client_dict in client_dicts]
            )
            lora_B_weights = torch.stack(
                [client_dict[lora_B_keys].detach() for client_dict in client_dicts]
            )

            # M shape: (d, k)
            M = sum(
                lora_B_weights[i] @ lora_A_weights[i] for i in range(len(client_dicts))
            ) / len(client_dicts)
            
            lora_A_avg = lora_A_weights.mean(0)
            lora_B_avg = lora_B_weights.mean(0)

            scaling_factor = args.lora_alpha / args.lora_r

            residue = M - lora_B_avg @ lora_A_avg
            
            global_dict[name + ".lora_A.default.weight"] = lora_A_avg
            global_dict[name + ".lora_B.default.weight"] = lora_B_avg
            global_dict[name + ".base_layer.weight"] += residue* scaling_factor
             

    global_model.load_state_dict(global_dict)
    return global_model