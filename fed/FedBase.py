import torch
from datetime import datetime

class FedBase:
    def __init__(self, args):
        self.args = args
    
    def run(self):
        pass

    def local_train(self, model, local_dataloader, lr, args):
        # torch.compile(model)
        model.cuda()
        model.train()
        steps = 0
        training_loss = 0
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay= args.weight_decay)
        for epoch in range(args.epochs):
            for idx, batch in enumerate(local_dataloader):
                batch = {k: v.to(model.device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # print(f"{datetime.now()} Loss: {loss.item()}")
                steps += 1
                training_loss += loss.item()
            print(f"Epoch: {epoch+1}/{args.epochs}: Loss: {loss.item()}")
        
        return model, training_loss/steps

    def eval_model(self, model, eval_dataloader):
        model.cuda()
        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for idx, batch in enumerate(eval_dataloader):
                batch = {k: v.to(model.device) for k, v in batch.items()}
                outputs = model(**batch)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                total += batch["labels"].size(0)
                correct += (predictions == batch["labels"]).sum().item()
        accuracy = correct / total
        model.cpu()
        return accuracy

    def aggerate_local_models(self, local_dict_list, global_dict, sample_num_list):
        total_samples = sum(sample_num_list)
        with torch.no_grad():
            for key in global_dict.keys():
                global_dict[key] = sum([local_dict[key] * sample_num_list[idx] / total_samples 
                                        for idx, local_dict in enumerate(local_dict_list)])
        return global_dict