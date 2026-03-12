import re
from vllm import LLM, SamplingParams
from datasets import load_dataset
import numpy as np
import argparse
def get_few_shot_prompt(examples):
    s = ""
    for example in examples:
        options = "\n".join([f"{chr(65+i)}. {option}" for i, option in enumerate(example["choices"])])
        formatted = f"{example['question']}\n\nOptions:\n{options}\nAnswer:"
        # 添加正确答案（0->A, 1->B, 等）
        answer_letter = chr(65 + example["answer"])
        formatted += f" {answer_letter}"
        s += formatted
    return s

def format_question(instruction,choices):
    options = ", ".join([f'{chr(65+i)}. "{option}"' for i, option in enumerate(choices)])
    formatted = f'The question is "{instruction}" and the optional answers are [{options}]'
    return formatted

def get_dataset_from_file():
    dataset = load_dataset("cais/mmlu", "all", split="test")
    return dataset


def generate_prompt(instruction, choices):
    instruction = format_question(instruction,choices)

    # 使用Alpaca模板包装
    alpaca_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
There is a multiple choice question with optional answers.  {}. Choose the right answer. Answer it with only "A" or "B" or "C" or "D"

### Response:
"""
    # print(alpaca_template.format(instruction))
    # exit()
    return alpaca_template.format(instruction)

def extract_answer(sentence: str) -> str:

    sentence_ = sentence.strip()
    match = re.search(r'\b([ABCD])\b', sentence_)
    return match.group(1) if match else ""

def test(model):
    stop_tokens = ["Instruction:", "Instruction", "Response:", "Response"]
    sampling_params = SamplingParams(temperature=0.1, top_p=0.75, top_k=40, max_tokens=32, stop=stop_tokens)
    dataset = get_dataset_from_file()
    # few_shot_examples = np.random.choice(range(len(dataset)),5)
    # few_shot_examples = dataset.select([0,1,7,8,11])
    # few_shot_prompt = get_few_shot_prompt(few_shot_examples)
    llm = LLM(model=model, tensor_parallel_size=4,gpu_memory_utilization=0.7)
    batch_size = 128
    generate_res = []
    print("generating responses...")

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i: i+batch_size]
        prompts = [generate_prompt(q, c) for q,c in zip(batch["question"],batch["choices"])]
        outputs = llm.generate(prompts, sampling_params)
        for output in outputs:
            generated_text = output.outputs[0].text.strip()
            generate_res.append(generated_text)
    print("evaluating responses...")

    correct = 0
    mis = 0
    for i, ex in enumerate(dataset):
        pred_answer = extract_answer(generate_res[i])
        gold_answer = chr(65 + int(ex["answer"]))  # 转成 A/B/C/D
        if pred_answer == "":
            mis +=1
        if pred_answer == gold_answer:
            correct += 1
    accuracy = correct / len(dataset)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"missing is {mis}")
    print(len(dataset))
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",type=str,default='chat_model')
    try: parsed = parser.parse_args()
    except IOError as msg: parser.error(str(msg))
    model = parsed.model
    test(model=model)
    print("Testing completed.")