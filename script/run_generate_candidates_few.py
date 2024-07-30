import os
import subprocess


processes = []
for part_id in range(1,5):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(part_id - 1)
        process = subprocess.Popen(['python', 'generate_symbol_output/generate_candidates_code_agent.py', \
                                        "--part_id", str(part_id), "--task_prefix", "mbpp_full_llama2chat", \
                                    "--base_model", "llama2chat", "--vllm_batchsize", "1", "--model_size", "7B", "--few_shot"], env=env)            
        processes.append(process)
for process in processes:
    process.wait()


processes = []
for part_id in range(1,5):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(part_id - 1)
        process = subprocess.Popen(['python', 'generate_symbol_output/generate_candidates_vali_agent.py', \
                                     "--part_id", str(part_id), "--task_prefix", "mbpp_full_llama2chat", \
                                    "--base_model", "llama2chat", "--vllm_batchsize", "1", "--model_size", "7B", "--few_shot"], env=env)            
        processes.append(process)
for process in processes:
    process.wait()
