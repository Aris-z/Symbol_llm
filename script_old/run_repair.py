import subprocess
import os
import argparse

def parse_args():
        parser = argparse.ArgumentParser(description="self-training framework combining symbolic solver and llms")
        parser.add_argument(
                "--cur_iter",
                type=int,
        )
        args = parser.parse_args()
        return args

args = parse_args()

processes = []
for part_id in range(1,4):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(part_id - 1)
    process = subprocess.Popen(['python', 'generate_symbol_output/generate_repaired_math_vLLM_self_training.py', \
                                "--cur_iter", str(args.cur_iter), "--part_id", str(part_id), "--task_prefix", "gsm_math_full_llama2chat", "--base_model", "llama2chat", \
                                "--model_size", "7B"], env=env)
    processes.append(process)
# wait for all the process to be completed
for process in processes:
    process.wait()
