import os
import subprocess


# processes = []
# for part_id in range(1,5):
#         env = os.environ.copy()
#         env['CUDA_VISIBLE_DEVICES'] = str(part_id - 1)
#         process = subprocess.Popen(['python', 'generate_symbol_output/generate_candidates_label_code.py', \
#                                         "--part_id", str(part_id), "--task_prefix", "label_unit_test", \
#                                     "--base_model", "llama2chat", "--vllm_batchsize", "1", "--model_size", "7B", "--few_shot"], env=env)            
#         processes.append(process)
# for process in processes:
#     process.wait()

# try:
#     subprocess.call(["python", "unit_test_label_preference.py", \
#                     "--task_prefix", "label_unit_test", "--nodes", "4", "--cur_iter", "0", "--few_shot"])
# except:
#     raise Exception("code Label preference failed")

try:

    for iter_num in range(4, 6):
        #######
        # Data organization
        #######
        try:
            subprocess.call(["python", "self-training/organize_preference_data_unit_test_code_agent.py", \
                            "--task_prefix", "label_unit_test", "--cur_iter", str(iter_num), "--model_size", "7B"])
        except:
            raise Exception("data organization failed")
        #######
        # train model
        #######
        try:
            subprocess.call(["bash", "open-instruct/scripts/finetune_lora_with_accelerate_agent_training.sh", \
                            "label_unit_test", str(iter_num), "llama2chat", "7B", "test"])
        except:
            raise Exception("agent training failed")
        #######
        # generate candidates
        #######
        processes = []
        for part_id in range(1,5):
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = str(part_id - 1)
                process = subprocess.Popen(['python', 'generate_symbol_output/generate_candidates_label_code.py', \
                                            "--cur_iter", str(iter_num), "--part_id", str(part_id), "--task_prefix", "label_unit_test", \
                                            "--base_model", "llama2chat", "--vllm_batchsize", "1", "--model_size", "7B"], env=env)            
                processes.append(process)
        for process in processes:
            process.wait()
        for process in processes:
            if process.returncode != 0:
                raise Exception("generate candidates failed")
        #######
        # generate repaired candidates
        #######
        processes = []
        for part_id in range(1,5):
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = str(part_id - 1)
                process = subprocess.Popen(['python', 'generate_symbol_output/generate_repaired_label_code.py', \
                                            "--cur_iter", str(iter_num), "--part_id", str(part_id), "--task_prefix", "label_unit_test", \
                                            "--base_model", "llama2chat", "--vllm_batchsize", "1", "--model_size", "7B"], env=env)            
                processes.append(process)
        for process in processes:
            process.wait()
        for process in processes:
            if process.returncode != 0:
                raise Exception("generate repaired candidates failed")
        #######
        # label preference
        #######
        try:
            subprocess.call(["python", "unit_test_label_preference.py", \
                            "--task_prefix", "label_unit_test", "--nodes", "4", "--cur_iter", str(iter_num)])
        except:
            raise Exception("code Label preference failed")
        
        try:
            subprocess.call(["python", "unit_test_label_preference.py", \
                            "--task_prefix", "label_unit_test", "--nodes", "4", "--cur_iter", str(iter_num), "--repaired"])
        except:
            raise Exception("code Label repaired preference failed")
except:
    subprocess.run("python /mnt/nas/data/yihan/get_gpu/get_gpu.py", shell=True)

subprocess.run("python /mnt/nas/data/yihan/get_gpu/get_gpu.py", shell=True)