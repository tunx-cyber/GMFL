import re
import re
from vllm import LLM, SamplingParams
from datasets import load_dataset
import argparse
from . import math_test
from fraction import Fraction


def format_gsm8k_prompt(example):
    # 使用与微调相同的模板
    alpaca_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
Let's think step by step.
"""
    return alpaca_template.format(example)

def is_digit(s):
    try:
        float(str(s).replace(",", ""))
        return True
    except ValueError:
        return False

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def extract_answer_number(completion):
    text = completion.split('The answer is: ')
    if len(text) > 1:
        extract_ans = text[-1].strip()
        match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
        if match:
            if '/' in match.group():
                denominator = match.group().split('/')[1]
                numerator = match.group().split('/')[0]
                if is_number(denominator) == True and is_number(numerator) == True:
                    if denominator == '0':
                        return round(float(numerator.replace(',', '')))
                    else:
                        frac = Fraction(match.group().replace(',', ''))
                        num_numerator = frac.numerator
                        num_denominator = frac.denominator
                        return round(float(num_numerator / num_denominator))
                else:
                    return None
            else:
                if float(match.group().replace(',', '')) == float('inf'):
                    return None
                return round(float(match.group().replace(',', '')))
        else:
            return None
    else:
        return None

def get_dataset_from_file():
    dataset = load_dataset("openai/gsm8k","main", split="test")
    # dataset = load_dataset("json",data_files="dataset/gsm8k_test.jsonl",split="train")
    return dataset

def test(model):
    stop_tokens = ["Instruction:", "Instruction", "Response:", "Response"]
    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=256, stop=stop_tokens)
    dataset = get_dataset_from_file()
    llm = LLM(model=model, tensor_parallel_size=8,gpu_memory_utilization=0.6)
    batch_size = 128
    generate_res = []
    print("generating responses...")

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i: i+batch_size]
        prompts = [format_gsm8k_prompt(q) for q in batch["question"]]
        outputs = llm.generate(prompts, sampling_params)
        for output in outputs:
            generated_text = output.outputs[0].text.strip()
            generate_res.append(generated_text)

    print("evaluating responses...")
    correct = 0
    for i, ex in enumerate(dataset):
        prediction = extract_answer_number(generate_res[i])
        ground_truth = ex['answer'].split('#### ')[1]
        ground_truth = int(ground_truth.replace(',', ''))
        if prediction and float(prediction) == float(ground_truth) or math_test.is_equiv(prediction, ground_truth):
            # print(prediction, ground_truth)
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
