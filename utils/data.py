from torch.utils.data import Dataset
from datasets import load_dataset
import datasets
import numpy as np
from collections import defaultdict
import torch
import random
# ==================================================sft==================================================
def preprocess_alpaca(example, tokenizer, max_length=512, math_reason = False):
    instruction = example["instruction"]
    response = example["response"]

    user_prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""
    if math_reason:
        user_prompt = user_prompt + "Let's think step by step.\n"
    # generate input_ids
    full_prompt = user_prompt + response + tokenizer.eos_token
    full_enc = tokenizer(full_prompt, truncation=True, max_length=max_length, padding="max_length")
    input_ids = full_enc["input_ids"]
    attention_mask = full_enc["attention_mask"]

    # only tokenize the prompt part to determine where the response starts
    user_prompt_enc = tokenizer(user_prompt, truncation=True, max_length=max_length)
    user_prompt_len = len(user_prompt_enc["input_ids"])

    # construct labels: -100 for prompt part, token ids for response part
    labels = [-100] * user_prompt_len + input_ids[user_prompt_len:]

    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels": torch.tensor(labels),
    }


def alpaca_format(example):
    if example['input'] == "":
        example["instruction"] = example["instruction"]
    else:
        example["instruction"] = example["instruction"] + " " + example['input']
    example["response"] = example['output']
    return example

def process_dolly(example):
    if example["context"] == "":
        example["instruction"] = example["instruction"]
    else:
        example["instruction"] = example["context"] + "\n" + example["instruction"]
    example["response"] = example["response"]
    return example

def get_dataset(dataset_name, dataset_sample):

    dataset = load_dataset(dataset_name, split="train")
    if dataset_name in ["vicgalle/alpaca-gpt4"]: # for chat and mmlu
        dataset = dataset.map(alpaca_format, remove_columns=['input', 'output', 'text'], desc=f"Preprocessing {dataset_name} for unified format.")
    elif dataset_name in ["zwhe99/commonsense_170k"]: # for QA
        dataset = dataset.map(alpaca_format, remove_columns=["answer"])
        # dataset = dataset.rename_column("output","response")
    elif dataset_name in ["meta-math/MetaMathQA"]: # for math
        dataset = dataset.rename_column("query", "instruction")
    elif dataset_name in ["databricks/databricks-dolly-15k"]:
        dataset = dataset.map(process_dolly, remove_columns=['category'], desc=f"Preprocessing {dataset_name} for unified format.")
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported.")
    dataset = dataset.shuffle(seed=2025)
    if dataset_sample:#for training, we sample a subset of the dataset
        num_sample = min(len(dataset), dataset_sample)
        dataset = dataset.select(range(num_sample))
    print(f">> ===== After processing, Dataset {dataset_name} has {len(dataset)} examples. =====")
    return dataset

class SFTDataset(Dataset):
    def __init__(self, data, tokenizer, template_name, max_len = 512, math_reason = False):
        self.data = data
        self.tokenizer = tokenizer
        self.template = template_name
        self.max_len = max_len
        self.math_reason = math_reason
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        example = self.data[index]
        if self.template == "alpaca":
            if self.math_reason == True:
                return preprocess_alpaca(example, self.tokenizer, self.max_len, True)
            else:
                return preprocess_alpaca(example, self.tokenizer, self.max_len)
            

#=================================================seq_cls==================================================
def get_classification_dataset(dataset_name, dataset_sample):
    tasks = ["sst2", "mnli", "qqp", "qnli", "rte", "cola", "stsb"]
    if "mnli" in dataset_name:
        dataset = load_dataset("glue", "mnli",download_mode="reuse_cache_if_exists")
    else:
        dataset = load_dataset("glue", dataset_name,download_mode="reuse_cache_if_exists")

    dataset = dataset.shuffle(seed=2025)

    if (
        dataset_name == "cola"
        or dataset_name == "sst2"
        or dataset_name == "mrpc"
        or dataset_name == "qqp"
        or dataset_name == "stsb"
        or dataset_name == "qnli"
        or dataset_name == "rte"
        or dataset_name == "wnli"
    ):
        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]
    elif dataset_name == "mnli_matched":
        train_dataset = dataset["train"]
        val_dataset = dataset["validation_matched"]
    elif dataset_name == "mnli_mismatched":
        train_dataset = dataset["train"]
        val_dataset = dataset["validation_mismatched"]

    if dataset_sample:
        num_sample = min(len(dataset["train"]), dataset_sample)
        train_dataset = train_dataset.select(range(num_sample))

    return train_dataset, val_dataset


def partition_data_dirichlet(dataset, num_clients, alpha):
    # random.seed(seed)
    # np.random.seed(seed)
    
    # 获取所有标签
    labels = np.array(dataset['label'])
    num_classes = len(set(labels))
    
    # 按类别分组数据索引
    class_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)
    
    # 为每个类别生成狄利克雷分布
    client_proportions = np.random.dirichlet([alpha] * num_clients, size=num_classes)
    
    # 分配数据到客户端
    client_indices = defaultdict(list)
    
    for class_id, indices in class_indices.items():
        # 打乱当前类别的数据
        random.shuffle(indices)
        
        # 计算每个客户端应该获得的当前类别的数据量
        proportions = client_proportions[class_id]
        proportions = proportions / proportions.sum()  # 确保和为1
        allocations = (proportions * len(indices)).astype(int)
        
        # 处理由于取整可能丢失的数据
        allocated_total = allocations.sum()
        if allocated_total < len(indices):
            # 将剩余的数据分配给比例最大的客户端
            remainder = len(indices) - allocated_total
            min_client = np.argmin(proportions)
            allocations[min_client] += remainder
        
        # 分配数据
        start = 0
        for client_id in range(num_clients):
            end = start + allocations[client_id]
            if end > len(indices):
                end = len(indices)
            client_indices[client_id].extend(indices[start:end])
            start = end
    
    return client_indices

# 获取特定客户端的数据
def get_client_data(dataset, client_indices, client_id):
    """获取指定客户端的数据"""
    indices = client_indices[client_id]
    return dataset.select(indices)


def split_dataset(args, dataset: datasets.Dataset):
    dataset = dataset.shuffle(seed=args.seed)        # Shuffle the dataset
    local_datasets = []

    if args.split_strategy == "iid":
        for i in range(args.num_clients):
            local_datasets.append(dataset.shard(args.num_clients, i))
    else:
        indices = partition_data_dirichlet(dataset, args.num_clients, args.alpha)
        for i in range(args.num_clients):
            local_datasets.append(get_client_data(dataset, indices, i))
    return local_datasets

            
class ClassificationDataset(Dataset):
    def __init__(self, task, data, tokenizer, max_len = 256):
        self.task = task
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        example = self.data[index]
        if "premise" in example and "hypothesis" in example:
            text = self.tokenizer(
                example["premise"],
                example["hypothesis"],
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
            )
        elif "question" in example and "sentence" in example:
            # QNLI and similar tasks
            text = self.tokenizer(
                example["question"],
                example["sentence"],
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
            )
        elif "sentence1" in example and "sentence2" in example:
            # MRPC, STS-B
            text = self.tokenizer(
                example["sentence1"],
                example["sentence2"],
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
                return_overflowing_tokens=False
            )
        elif "question1" in example and "question2" in example:
            # QQP
            text = self.tokenizer(
                example["question1"],
                example["question2"],
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
                return_overflowing_tokens=False
            )
        elif "sentence" in example:
            # CoLA, SST-2
            text = self.tokenizer(
                example["sentence"],
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
                
            )
        else:
            raise ValueError(f"Unexpected format for task {self.task}")

        if self.task in ["stsb"]:
            label = torch.tensor(self.data[index]["score"], dtype=torch.float)
        else:
            label = torch.tensor(self.data[index]["label"], dtype=torch.long)
            
        return {
            "input_ids": torch.tensor(text["input_ids"]),
            "attention_mask": torch.tensor(text["attention_mask"]),
            "labels": label,
        }
    
def get_dataset_this_round(dataset, round, args):
    num2sample = args.batch_size * args.gradient_accumulation_steps * args.max_steps
    num2sample = min(num2sample, len(dataset))
    random.seed(round)
    random_idx = random.sample(range(0, len(dataset)), num2sample)
    dataset_this_round = dataset.select(random_idx)

    return dataset_this_round