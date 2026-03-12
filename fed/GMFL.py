# federated layerwise selection
from utils.utils import cosine_learning_rate,get_model_and_tokenizer,draw_loss_curve, setup_logger,get_peft_config
from utils.data import get_dataset, split_dataset, SFTDataset, get_dataset_this_round,get_classification_dataset,ClassificationDataset
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
import torch
from torch.utils.data import DataLoader
import numpy as np
import copy
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import entropy
import numpy as np
from collections import OrderedDict
from .FedBase import FedBase
import math

def compute_importance(param_dict,global_dict):
    state_dict = {}
    for name, param in param_dict.items():
        state_dict[name] = torch.sum(torch.abs(param_dict[name] * global_dict[name]))
    return state_dict

def compute_weight_stats(param_dict,global_dict):
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

     
class GMFL(FedBase):
    def __init__(self, args):
        super(GMFL,self).__init__(args)
        self.history_g = {}
    
    def run(self):
        args = self.args
        logger = setup_logger(args.fed_alg, f"./logs/{args.fed_alg}/{args.dataset_name.replace('/','_')}.txt")
        logger.info(args)
        # init model and tokenizer
        model, tokenizer = get_model_and_tokenizer(args)
        peft_config = get_peft_config(args)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        model.config.use_cache = False
        model.cuda()
        
        # set up the global and local models
        global_dict = copy.deepcopy(get_peft_model_state_dict(model))
        # local_dict_list = [copy.deepcopy(global_dict) for i in range(args.num_clients)]
        # init history
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
        mask_dict = None
        buffer = {}
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
                if args.task == "sft":
                    set_peft_model_state_dict(model, global_dict)
                else:
                    set_peft_model_state_dict(model, copy.deepcopy(global_dict))# fast fix bug here
                if mask_dict:
                # 只处理lora部分，对前四轮进行被选中的训练，对第五轮进行没有被选中的训练。
                    for name, param in model.named_parameters():
                        if "lora" in name:
                            if mask_dict[name.replace("default.weight","weight")] == 0:
                                param.requires_grad = False
                            else:
                                param.requires_grad = True          
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
            if self.args.ex_style:
                if (r+1) % 20 == 0:
                    mask_dict, global_dict = self.aggerate_local_models_ex(local_dict_list, model, r, True)
                else:
                    _, global_dict = self.aggerate_local_models_ex(local_dict_list, model, r, False)
            else:
                if (r+1) % 20 == 0:
                    mask_dict = self.aggerate_local_models_avg(local_dict_list, global_dict, sample_num_list, buffer,r, mask_dict, True,model)
                else:
                    self.aggerate_local_models_avg(local_dict_list, global_dict, sample_num_list, buffer, r, mask_dict, False,model)
            
            rounds_loss.append(sum(round_loss)/len(round_loss))
            print(f"Round {r+1} loss: {rounds_loss[-1]}")
            # if args.task == "classification":
            #     set_peft_model_state_dict(model, copy.deepcopy(global_dict))
            #     accuracy = self.eval_model(model, test_dataloder)
            #     logger.info(f"Round {r+1} test Accuracy: {accuracy}")
        if self.args.ex_style:
            model.save_pretrained(f"./output/{args.fed_alg}/{args.dataset_name.replace('/','_')+str(args.seed)+'_ex_'+str(args.beta)}")
            tokenizer.save_pretrained(f"./output/{args.fed_alg}/{args.dataset_name.replace('/','_')+str(args.seed)+'_ex_'+str(args.beta)}")
        else:
            model.save_pretrained(f"./output/{args.fed_alg}/{args.dataset_name.replace('/','_')+str(args.seed)+'_'+str(args.beta)+'_c'+str(args.c)}")
            tokenizer.save_pretrained(f"./output/{args.fed_alg}/{args.dataset_name.replace('/','_')+str(args.seed)+'_'+str(args.beta)+'_c'+str(args.c)}")
        logger.info("loss data:")
        logger.info(rounds_loss)
        # draw_loss_curve(range(args.num_rounds),rounds_loss,args)
    
    # def aggerate_local_models_svd(self, local_dict_list, global_dict, sample_num_list):# for last agg
    #     total_samples = sum(sample_num_list)
    #     # print(g_state_dict.keys())
    #     with torch.no_grad():
    #         for key in global_dict.keys():
    #             if "lora_A" in key:
    #                 lora_B_key = key.replace("lora_A", "lora_B")
    #                 # base_layer_key = key.replace("lora_A", "base_layer")
    #                 base_weight = sum(local_dict[lora_B_key] @ local_dict[key] * sample_num_list[idx] / total_samples
    #                                   for idx, local_dict in enumerate(local_dict_list))
    #                 # print(base_layer_key)
    #                 r = self.args.peft_lora_r
    #                 U, S, V = torch.svd_lowrank(base_weight,q=r,niter=2)

    #                 # 构造两个矩阵
    #                 M1 =  U @ torch.diag(S)   # m x r
    #                 M2 = V.T  # r x n
    #                 # print(global_dict[key].shape, global_dict[lora_B_key].shape, M1.shape, M2.shape)
    #                 # A_approx = M1 @ M2
    #                 # print("误差:", torch.norm(base_weight - A_approx).item())
    #                 global_dict[key] = M2
    #                 global_dict[lora_B_key] = M1
                    
    #     return global_dict
    
    def aggerate_local_models_ex(self, local_dict_list, global_model,r, re_mask = True):
        global_dict = global_model.state_dict()
        if re_mask:
            old_global_dict = copy.deepcopy(get_peft_model_state_dict(global_model))
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

                scaling_factor = self.args.peft_lora_alpha / self.args.peft_lora_r

                residue = M - lora_B_avg @ lora_A_avg
                
                global_dict[name + ".lora_A.default.weight"] = lora_A_avg
                global_dict[name + ".lora_B.default.weight"] = lora_B_avg
                global_dict[name + ".base_layer.weight"] += residue * scaling_factor
                

        global_model.load_state_dict(global_dict)

        global_dict = get_peft_model_state_dict(global_model)
        if re_mask:
            delta_dict = {}
            for key in global_dict.keys():
                delta_dict[key] = global_dict[key] - old_global_dict[key]
            stat_dict = compute_weight_stats(delta_dict,old_global_dict)
            mask_dict = fixed_sample_layers(stat_dict, ritio=self.args.c + ((r+1)//20)*self.args.beta)
            return mask_dict,global_dict
        else:
            return None,global_dict

    def aggerate_local_models_avg(self, local_dict_list, global_dict, sample_num_list, buffer, r, mask_dict = None, re_mask = True,global_model=None):
        total_samples = sum(sample_num_list)
        avg_dict = {}
        for key in global_dict.keys():
            if mask_dict:
                if "classifier" in key or mask_dict[key] == 1:
                    avg_dict[key] = sum([local_dict_list[idx][key] * sample_num_list[idx] / total_samples for idx, d in enumerate(local_dict_list)])
                else:
                    avg_dict[key] = global_dict[key]
            else:
                avg_dict[key] = sum([local_dict_list[idx][key] * sample_num_list[idx] / total_samples for idx, d in enumerate(local_dict_list)])

        if re_mask:
            delta_dict = {}
            for key in global_dict.keys():
                delta_dict[key] = avg_dict[key] - global_dict[key]
            stat_dict = compute_weight_stats(delta_dict,global_dict)
            mask_dict = fixed_sample_layers(stat_dict, ritio=self.args.c + ((r+1)//20)*self.args.beta)
        with torch.no_grad():
            for key in global_dict.keys():
                global_dict[key] = avg_dict[key]
                # buffer[key] = delta_dict[key]
        if re_mask:
            return mask_dict
        else: 
            return None
