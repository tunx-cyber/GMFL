
import subprocess
import os

seeds=[123, 456]
algs = ["fedit", "fedex","ffalora","feda2", "fedrand"]

for alg in algs:
    dicts = []
    for seed in seeds:
        try:
            result = subprocess.run(
                "bash mt.sh {} {}".format(alg, seed),
                shell=True
            )
        except subprocess.CalledProcessError as e:
            print(f"❌ 命令执行失败，返回码: {e.returncode}")
            print(f"错误输出:\n{e.stderr}")
            exit(1)
        