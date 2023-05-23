from multiprocessing import Process, Queue
import sys, os
import time

import argparse

sys.path.append(os.path.abspath("."))


def kwargs_to_cmd(kwargs):
    cmd = "python3 ctc_mnist.py "
    for flag, val in kwargs.items():
        cmd += f"--{flag}={val} "

    return cmd


def run_exp(gpu_num, in_queue):
    while not in_queue.empty():
        try:
            experiment = in_queue.get(timeout=3)
        except:
            return

        before = time.time()

        experiment["gpu"] = gpu_num
        print(f"==> Starting experiment {kwargs_to_cmd(experiment)}")
        os.system(kwargs_to_cmd(experiment))

        with open("output.txt", "a+") as f:
            f.write(
                f"Finished experiment {experiment} in {str((time.time() - before) / 60.0)}."
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default=0, type=lambda x: [a for a in x.split("|") if a])
    parser.add_argument('--seeds', default=1, type=lambda x: [a for a in x.split("|") if a])
    args = parser.parse_args()

    # gpus = [1, 2, 3]
    gpus = args.gpus
    # seeds = [1, 2, 3]
    seeds = args.seeds

    experiments = []
    for seed in seeds:
        kwargs = {
            "learning_rate": 0.0004,
            "max_epochs": 10,
            "warmup": 20,
            "batch_size": 32,
            "expl_lambda": 2.0,
            "seed": seed,
        }

        experiments.append(kwargs)

    print(experiments)
    queue = Queue()

    for e in experiments:
        queue.put(e)

    processes = []
    for gpu in gpus:
        p = Process(target=run_exp, args=(gpu, queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
