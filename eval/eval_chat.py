import argparse
from openai import OpenAI, OpenAIError
import os
import json
import re
import ast
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

one_score_pattern = re.compile("\[\[(\d+\.?\d*)\]\]")
one_score_pattern_backup = re.compile("\[(\d+\.?\d*)\]")

alpaca_one = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{} 

### Response:
"""

aplaca_two = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{} 

### Response:
{}

### Instruction:
{} 

### Response:
"""

def gen_mt_1(question, answer, ref_answer=None):
    if ref_answer is not None:
        sys_prompt = "You are a helpful assistant."
        prompt_template = (
            "[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. "
            "Your evaluation should consider correctness and helpfulness. You will be given a reference answer and the assistant's answer. "
            "Begin your evaluation by comparing the assistant's answer with the reference answer. Identify and correct any mistakes. Be as objective as possible. "
            "After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", "
            "for example: \"Rating: [[5]]\".\n\n"
            "[Question]\n{question}\n\n[The Start of Reference Answer]\n{ref_answer_1}\n[The End of Reference Answer]\n\n"
            "[The Start of Assistant's Answer]\n{answer}\n[The End of Assistant's Answer]"
        )
        user_prompt = prompt_template.format(question=question,answer=answer,ref_answer_1=ref_answer)
    else:
        sys_prompt = "You are a helpful assistant."
        prompt_template = (
            "[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. "
            "Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. "
            "Begin your evaluation by providing a short explanation. Be as objective as possible. "
            "After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", "
            "for example: \"Rating: [[5]]\".\n\n"
            "[Question]\n{question}\n\n[The Start of Assistant's Answer]\n{answer}\n[The End of Assistant's Answer]"
        )
        user_prompt = prompt_template.format(question=question,answer=answer)
    messages = []
    messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages

def gen_mt_2(questions, answers, ref_answers=None):
    if ref_answers is not None:
        sys_prompt = (
            "Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question. "
            "Your evaluation should consider correctness and helpfulness. You will be given a reference answer and the assistant's answer. "
            "Your evaluation should focus on the assistant's answer to the second question. "
            "Begin your evaluation by comparing the assistant's answer with the reference answer. Identify and correct any mistakes. Be as objective as possible. "
            "After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", "
            "for example: \"Rating: [[5]]\".\n\n"
        )
        prompt_template = (
            "<|The Start of Reference Answer|>\n\n"
            "### User:\n{question_1}\n\n### Reference answer:\n{ref_answer_1}\n\n"
            "### User:\n{question_2}\n\n### Reference answer:\n{ref_answer_2}\n\n"
            "<|The End of Reference Answer|>\n\n\n"
            "<|The Start of Assistant A's Conversation with User|>\n\n"
            "### User:\n{question_1}\n\n### Assistant A:\n{answer_1}\n\n"
            "### User:\n{question_2}\n\n### Assistant A:\n{answer_2}\n\n"
            "<|The End of Assistant A's Conversation with User|>"
        )
        user_prompt = prompt_template.format(
            question_1=questions[0],
            ref_answer_1=ref_answers[0],
            question_2=questions[1],
            ref_answer_2=ref_answers[1],
            answer_1=answers[0],
            answer_2=answers[1],
        )
    else:
        sys_prompt = (
            "Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. "
            "Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. "
            "Your evaluation should focus on the assistant's answer to the second user question. "
            "Begin your evaluation by providing a short explanation. Be as objective as possible. "
            "After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", "
            "for example: \"Rating: [[5]]\".\n\n"
        )
        prompt_template = (
            "<|The Start of Assistant A's Conversation with User|>\n\n"
            "### User:\n{question_1}\n\n### Assistant A:\n{answer_1}\n\n"
            "### User:\n{question_2}\n\n### Assistant A:\n{answer_2}\n\n"
            "<|The End of Assistant A's Conversation with User|>"
        )
        user_prompt = prompt_template.format(
            question_1=questions[0],
            answer_1=answers[0],
            question_2=questions[1],
            answer_2=answers[1],
        )
    messages = []
    messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages
'''

'''
def chat_completion_openai(model, messages, temperature, max_tokens, api_dict={"api_key":"", "base_url": ""}):
    if api_dict is None:
        client = OpenAI(
            api_key=os.environ.get('DEEPSEEK_API_KEY'),
            base_url=""
        )
    else:
        client = OpenAI(
            api_key=api_dict['api_key'],
            base_url=api_dict['base_url']
        )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=42
        )
    except OpenAIError as e:
        print(f"OpenAI API error: {e}")
        return -1
    if hasattr(response.choices[0].message,"reasoning_content") and response.choices[0].message.reasoning_content:
        return str(response.choices[0].message.reasoning_content)+str(response.choices[0].message.content)
    else:
        return response.choices[0].message.content
def get_score_from_judgment(judgment):
    match = re.search(one_score_pattern, judgment)
    if not match:
        match = re.search(one_score_pattern_backup, judgment)

    if match:
        rating = ast.literal_eval(match.groups()[0])
    else:
        rating = -1
    return rating
import time
def gen_judgement(question_file, answer_file, ref_answer_file, output_file=None):
    questions = []
    with open(question_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()  # 去除换行符和空白
            if line:  # 跳过空行
                questions.append(json.loads(line))

    ans = []
    with open(answer_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()  # 去除换行符和空白
            if line:  # 跳过空行
                ans.append(json.loads(line))
    
    ref_answers = []
    with open(ref_answer_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()  # 去除换行符和空白
            if line:  # 跳过空行
                ref_answers.append(json.loads(line))
    ref_answers_dict = {item['question_id']: item for item in ref_answers}
    # print(ref_answers_dict[101]["choices"][0]['turns'][0])
    # print(ref_answers_dict[101]["choices"][0]['turns'][1])
    # exit()
    mt_1 = []
    mt_2 = []
    for question, answer in tqdm(zip(questions, ans),total=len(questions)):
        if question["question_id"] <= 154:
            continue
        assert question['question_id'] == answer['question_id'], "Mismatched question and answer IDs"
        q_1 = question['turns'][0]
        a_1 = answer['choices'][0]['turns'][0]
        q_2 = question['turns'][1]
        a_2 = answer['choices'][0]['turns'][1]
        ref_answer_item = ref_answers_dict.get(int(question['question_id']), None)
        if ref_answer_item:
            ref_a_1 = ref_answer_item["choices"][0]['turns'][0]
            ref_a_2 = ref_answer_item["choices"][0]['turns'][1]
            ref_answers_pair = [ref_a_1, ref_a_2]
        else:
            ref_answers_pair = None
        mt_1_msg = gen_mt_1(q_1, a_1, ref_answer=ref_a_1 if ref_answer_item else None)
        mt_2_msg = gen_mt_2([q_1, q_2],[a_1, a_2],ref_answers=ref_answers_pair)
        model_name = "openai/gpt-oss-120b"
        mt_1_judge = chat_completion_openai(model_name, mt_1_msg, temperature=0, max_tokens=2048)
        mt_2_judge = chat_completion_openai(model_name, mt_2_msg, temperature=0, max_tokens=2048)
        mt_1.append(get_score_from_judgment(mt_1_judge))
        mt_2.append(get_score_from_judgment(mt_2_judge))
        result = {
            "question_id": question['question_id'],
            "mt_1_judge": mt_1_judge,
            "mt_2_judge": mt_2_judge,
            "mt_1_score": mt_1[-1],
            "mt_2_score": mt_2[-1]
        }
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "a") as fout:
                fout.write(json.dumps(result) + "\n")
        time.sleep(10)  # 添加1秒的延迟，避免请求过于频繁
        
    mt_1 = [score for score in mt_1 if score != -1]
    mt_2 = [score for score in mt_2 if score != -1]
    print("MT-1 Scores:", mt_1)
    print("MT-2 Scores:", mt_2)
    print("MT-1 Average Score:", sum(mt_1) / len(mt_1))
    print("MT-2 Average Score:", sum(mt_2) / len(mt_2))
    print("MT-Avg Score:", (sum(mt_1) + sum(mt_2)) / (len(mt_1) + len(mt_2)))


def show_results(result_file):
    scores = []
    with open(result_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()  # 去除换行符和空白
            if line:  # 跳过空行
                scores.append(json.loads(line))
    mt_1 = [item['mt_1_score'] for item in scores]
    mt_2 = [item['mt_2_score'] for item in scores]
    mt_1 = [score for score in mt_1 if score != -1]
    mt_2 = [score for score in mt_2 if score != -1]
    mt_1_avg = sum(mt_1) / len(mt_1)
    mt_2_avg = sum(mt_2) / len(mt_2)
    # print("MT-1 Average Score:", mt_1_avg)
    # print("MT-2 Average Score:", mt_2_avg)
    # print("MT-Avg Score:", (mt_1_avg + mt_2_avg) / 2)
    return {
        "mt_1_avg": mt_1_avg,
        "mt_2_avg": mt_2_avg,
        "mt_avg": (mt_1_avg + mt_2_avg) / 2
    }

def generate_answers(model_name, lora_path, question_file, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    llm = LLM(model=model_name, enable_lora=True, tensor_parallel_size=1,gpu_memory_utilization=0.7,max_lora_rank=32)
    sampling_params = SamplingParams(temperature=0.7, top_p=1.0, max_tokens=1024)
    questions = []
    with open(question_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()  # 去除换行符和空白
            if line:  # 跳过空行
                questions.append(json.loads(line))
    
    for question in tqdm(questions):
        ans1 = llm.generate(alpaca_one.format(question['turns'][0]),
                            sampling_params=sampling_params,
                            lora_request=LoRARequest("lora", 1, lora_path))
        ans1 = ans1[0].outputs[0].text.strip()
        # second turn inference
        try:
            ans2 = llm.generate(aplaca_two.format(question['turns'][0], ans1, question['turns'][1]),
                                sampling_params=sampling_params,
                                lora_request=LoRARequest("lora", 1, lora_path))
            ans2 = ans2[0].outputs[0].text.strip()
        except ValueError as e:
            print(e)
            print("We skip")
            continue
        result = {
            "question_id": question['question_id'],
            "choices": [{
                "turns": [ans1, ans2]
            }]
        }
        
        with open(output_file, "a") as fout:
            fout.write(json.dumps(result) + "\n")


import argparse
if __name__ == "__main__":
    # gen_judgement("/home/tunx/MyLLM/FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl",
    #               "/home/tunx/MyLLM/ans_file.jsonl",
    #               "/home/tunx/MyLLM/ref_ans.jsonl",
    #               "/home/tunx/MyLLM/result.jsonl")
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='huggyllama/llama-7b', type=str, help='Model name or path')
    parser.add_argument('--lora_path',type=str,default="", help='LoRA path')
    parser.add_argument('--question_file', type=str, default="/root/llm/eval/open_ended/data/mtbench/question.jsonl",help='Path to question file')
    parser.add_argument('--answer_file', type=str, default="",help='Path to save generated answers')
    parser.add_argument('--ref_answer_file', type=str, default="/root/llm/eval/open_ended/data/mtbench/reference_answer/gpt-4.jsonl",help='Path to reference answer file')
    parser.add_argument('--result_file', type=str,default="", help='Path to save judgement results')
    args = parser.parse_args()
    # generate_answers(args.model, args.lora_path, args.question_file, args.answer_file)
    # gen_judgement(args.question_file, args.answer_file, args.ref_answer_file, args.result_file)
    print(show_results(args.result_file))

    