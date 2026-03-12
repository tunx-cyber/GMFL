import re
from vllm import LLM, SamplingParams
from datasets import load_dataset
import argparse
from . import math_test

def format_math_prompt(example):
    # 使用与微调相同的模板
    alpaca_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
Let's think step by step.
"""
    return alpaca_template.format(example)

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None
    
invalid_outputs = []
def process_results(doc, completion, answer):
    split_ans = completion.split('The answer is: ')
    if len(split_ans) > 1:
        ans = split_ans[-1]
        extract_ans_temp = ans.split('.\n')[0]
        extract_ans_temp = extract_ans_temp.strip()
        if len(extract_ans_temp)>0 and extract_ans_temp[-1] == '.':
            extract_ans = extract_ans_temp[0:-1]
        else:
            extract_ans = extract_ans_temp
        extract_ans = extract_ans.strip()
        if math_test.is_equiv(extract_ans, answer):
            # print(extract_ans, answer)
            return True
        else:
            return False
    else:
        temp = {'question': doc, 'output': completion, 'answer': answer}
        invalid_outputs.append(temp)
        return False

def get_dataset_from_file():
    dataset = load_dataset('json',data_files="/root/llm/dataset/MATH_test.jsonl")
    return dataset["train"]

def test(model):
    stop_tokens = ["Instruction:", "Instruction", "Response:", "Response"]
    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=512, stop=stop_tokens)
    dataset = get_dataset_from_file()
    llm = LLM(model=model, tensor_parallel_size=8,gpu_memory_utilization=0.7)
    batch_size = 128
    generate_res = []
    print("generating responses...")

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i: i+batch_size]
        prompts = [format_math_prompt(q) for q in batch["instruction"]]
        outputs = llm.generate(prompts, sampling_params)
        for output in outputs:
            generated_text = output.outputs[0].text.strip()
            generated_text = generated_text[:-4]
            # print(generated_text)
            generate_res.append(generated_text)
        # break
    print("evaluating responses...")
    correct = 0
    for i, ex in enumerate(dataset):
        prediction = generate_res[i]
        ground_truth = ex['output']
        ground_truth = remove_boxed(math_test.last_boxed_only_string(ground_truth))
        if process_results(None, prediction, ground_truth):
            correct += 1
    accuracy = correct / len(dataset)
    return accuracy

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",type=str,default='math_model')
    try: parsed = parser.parse_args()
    except IOError as msg: parser.error(str(msg))
    model = parsed.model
    print(test(model=model))
    print("Testing completed.")
