from utils.utils import read_options,setup_seed
#export HF_ENDPOINT="https://hf-mirror.com"
def run_fed(args):
    if args.fed_alg == "fedit":
        from fed.FedIT import FedIT
        fed = FedIT(args)
        fed.run()
    elif args.fed_alg == "ffalora":
        from fed.FFALora import FFALora
        fed = FFALora(args)
        fed.run()
    elif args.fed_alg == "flora":
        from fed.FLora import FLora
        fed = FLora(args)
        fed.run()
    elif args.fed_alg == "gmfl":
        from fed.GMFL import GMFL
        fed = GMFL(args)
        fed.run()
    elif args.fed_alg == "fedex":
        from fed.FedExLora import FedEx
        fed = FedEx(args)
        fed.run()
    elif args.fed_alg == "ffaloragm":
        from fed.FFALoRAGM import FFALoraGM
        fed = FFALoraGM(args)
        fed.run()
    # elif args.fed_alg == "ffalorals":
    #     from fed.FFALoRALS import FFALoraLS
    #     fed = FFALoraLS(args)
    #     fed.run()
    elif args.fed_alg == "lora_a2":
        from fed.LoRA_A2 import LoRA_A2
        fed = LoRA_A2(args)
        fed.run()
    elif args.fed_alg == "lora_a2gm":
        from fed.LoRA_A2GM import LoRA_A2GM
        fed = LoRA_A2GM(args)
        fed.run()
    else:
        raise ValueError(f"Unknown federated learning algorithm: {args.fed_alg}")

if __name__ == "__main__":
    args = read_options()
    setup_seed(args.seed)
    import time
    start_time = time.time()
    run_fed(args)
    end_time = time.time()
    print(f"Total training time: {end_time - start_time} seconds")
