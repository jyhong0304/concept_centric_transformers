from multiprocessing import Process, Queue
import sys, os
import time
from itertools import product

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

        with open("slotctc_training_samples_output.txt", "a+") as f:
            f.write(
                f"Finished experiment {experiment} in {str((time.time() - before) / 60.0)}."
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default=0, type=lambda x: [a for a in x.split("|") if a])
    parser.add_argument('--seeds', default=1, type=lambda x: [a for a in x.split("|") if a])
    args = parser.parse_args()

    gpus = args.gpus
    seeds = args.seeds
    n_training_samples = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000]

    experiments = []
    for seed, num_sample in product(seeds, n_training_samples):
        kwargs = {
            "learning_rate": 0.001,
            "max_epochs": 150,
            "warmup": 20,
            "batch_size": 32,
            "expl_lambda": 2.0,
            "n_train_samples": num_sample,
            "model": "mnist_slotctc",
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
