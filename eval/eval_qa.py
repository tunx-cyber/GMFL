import re
from vllm import LLM, SamplingParams
from datasets import load_dataset
import argparse
def process_dataset(example):
    if example["input"] == "":
        example["instruction"] = example["instruction"]
    else:
        example["instruction"] = f"{example['instruction']} {example['input']}"
    return example

def get_dataset_from_file(data_path, dataset_name):
    dataset = load_dataset('json', data_files=data_path)
    if dataset_name in ["boolq", "piqa", "social_i_qa", "ARC-Challenge", "ARC-Easy", "openbookqa", "hellaswag", "winogrande"]:
        dataset.map(process_dataset)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    # dataset = dataset.shuffle(seed=2025)
    return dataset["train"]#.select(range(1000))  # For testing, limit to 1000 samples
    
def generate_prompt(instruction, input=None):
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction} 

### Response:
"""


def extract_answer(dataset: str, sentence: str) -> str:
    """Extract the answer from model output based on dataset type."""
    sentence_ = sentence.strip().lower()
    
    if dataset == 'boolq':
        pred_answers = re.findall(r'true|false', sentence_)
    elif dataset == 'piqa':
        pred_answers = re.findall(r'solution1|solution2', sentence_)
    elif dataset in ['social_i_qa', 'ARC-Challenge', 'ARC-Easy', 'openbookqa']:
        pred_answers = re.findall(r'answer1|answer2|answer3|answer4|answer5', sentence_)
    elif dataset == 'hellaswag':
        pred_answers = re.findall(r'ending1|ending2|ending3|ending4', sentence_)
    elif dataset == 'winogrande':
        pred_answers = re.findall(r'option1|option2', sentence_)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
        
    return pred_answers[0] if pred_answers else ""

def test(model, dataset_names = ["ARC-Challenge", "ARC-Easy", "boolq", "piqa", "social_i_qa", "openbookqa", "hellaswag", "winogrande"]):
    # Setup VLLM
    stop_tokens = ["Instruction:", "Instruction", "Response:", "Response"]
    sampling_params = SamplingParams(temperature=0.1, top_p=0.75, top_k=40, max_tokens=32, stop=stop_tokens)
    llm = LLM(model=model, tensor_parallel_size=4,gpu_memory_utilization=0.7)#tensor_parallel_size means the number of GPUs to use
    test_dict = {}
    acc = []
    for dataset_name in dataset_names:
        data_path = f"dataset/{dataset_name}/test.json"
        dataset = get_dataset_from_file(data_path, dataset_name) # we can specify the detail
        batch_size = 128
        generate_res = []
        print("generating responses...")
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i: i+batch_size]
            prompts = [generate_prompt(ex) for ex in batch["instruction"]]
            outputs = llm.generate(prompts, sampling_params)
            for output in outputs:
                generated_text = output.outputs[0].text
                generate_res.append(generated_text)
        print("evaluating responses...")
        correct = 0
        for i, ex in enumerate(dataset):
            pred_answer = extract_answer(dataset_name, generate_res[i])
            if pred_answer == str(ex["answer"]).lower():
                correct += 1
        accuracy = correct / len(dataset)
        test_dict[dataset_name] = accuracy
        acc.append(accuracy)
    test_dict["avg"] = sum(acc)/len(acc)
    return test_dict
from vllm.lora.request import LoRARequest
def test_lora(model, lora_path,id, dataset_names = ["ARC-Challenge", "ARC-Easy", "boolq", "piqa", "social_i_qa", "openbookqa", "hellaswag", "winogrande"]):
    llm = model
    stop_tokens = ["Instruction:", "Instruction", "Response:", "Response"]
    sampling_params = SamplingParams(temperature=0.1, top_p=0.75, top_k=40, max_tokens=32, stop=stop_tokens)
    test_dict = {}
    acc = []
    for dataset_name in dataset_names:
        data_path = f"dataset/{dataset_name}/test.json"
        dataset = get_dataset_from_file(data_path, dataset_name) # we can specify the detail
        batch_size = 128
        generate_res = []
        print("generating responses...")
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i: i+batch_size]
            prompts = [generate_prompt(ex) for ex in batch["instruction"]]
            outputs = llm.generate(prompts, sampling_params,lora_request=LoRARequest(f"lora{id}", id+1, lora_path))
            for output in outputs:
                generated_text = output.outputs[0].text
                generate_res.append(generated_text)
        print("evaluating responses...")
        correct = 0
        for i, ex in enumerate(dataset):
            pred_answer = extract_answer(dataset_name, generate_res[i])
            if pred_answer == str(ex["answer"]).lower():
                correct += 1
        accuracy = correct / len(dataset)
        test_dict[dataset_name] = accuracy
        acc.append(accuracy)
    test_dict["avg"] = sum(acc)/len(acc)
    return test_dict

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model', default='qa_model')
#     try: args = parser.parse_args()
#     except IOError as msg: parser.error(str(msg))
#     print(test(args.model))
#     print("Testing completed.")

if __name__ == "__main__":
    seeds = [42,123,456]
    lora_path = "/root/llm/output/gmtest/zwhe99_commonsense_170k{seed}_0.05_c0.1"
    results = []
    llm = LLM(model="huggyllama/llama-7b", enable_lora=True, tensor_parallel_size=4,gpu_memory_utilization=0.7,max_lora_rank=32)
    for i, seed in enumerate(seeds):
        test_dict = test_lora(llm, lora_path.format(seed = seed),i)
        results.append((seed, test_dict))
    for seed, res in results:
        print(f"Seed: {seed}, Results: {res}")
    