# Official code for ACL 2026 "GMFL: Efficient Global Masking for Federated LLM Fine-tuning"
## how to run

example
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --fed_alg gmfl --dataset_name zwhe99/commonsense_170k --seed 42 --c 0.1 --beta 0.05
```

For FedEx-LoRA with GMFL
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --fed_alg gmfl --dataset_name zwhe99/commonsense_170k --seed 42 --c 0.1 --beta 0.05 --ex_style
```

`fed_alg` could be one of ` ["gmfl","fedit", "fedex","ffalora", "ffaloragm", "lora_a2", "lora_a2gm"]`
For FedIT with GMFL.

`dataset_name`  could be one of `["zwhe99/commonsense_170k", "vicgalle/alpaca-gpt4", "sst2", "mnli_matched", "qqp", "mrpc", "rte", "cola", "meta-math/MetaMathQA"]`


attention:

The seeds used in my paper are [42,123,456]. If you use seed 456, you will find the performance of all methods is severely degraded.
