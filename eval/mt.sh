export PYTHONPATH=/root/llm/eval
alg=$1
seed=$2
CUDA_VISIBLE_DEVICES=7 python open_ended/gen_model_answer_mt.py --base_model_path huggyllama/llama-7b --lora_path /root/llm/output/$alg/"${alg}_${seed}" --template alpaca

python open_ended/gen_judge_mtbench.py --judge_model gpt-4o --model_list "${alg}_${alg}_${seed}"

# python open_ended/show_results_mt.py --model_list vicgalle_alpaca-gpt4_  --judge_model gpt-4o