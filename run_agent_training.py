import subprocess

# Define your commands
commands = [
    "bash script/run_organize_preference.sh {iter_num}",
    "bash script/run_train.sh {iter_num} code",
    "bash script/run_train.sh {iter_num} validation",
    # "rm output/code_agent/mbpp_full_llama2chat_sft_iter{iter_num}_sft_tune_llama2chat_7B/model.safetensors",
    # "rm output/validation_agent/mbpp_full_llama2chat_sft_iter{iter_num}_sft_tune_llama2chat_7B/model.safetensors",
    "python script/run_generate_candidates.py --cur_iter {iter_num}",
    "python script/run_repair.py --cur_iter {iter_num}",
    "bash script/run_label_preference.sh {iter_num}"
]

# Function to run a command and check its outcome
def run_command(command):
    try:
        result = subprocess.run(command, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: Command '{command}' failed with return code {e.returncode}.")
        return False

# Execute commands in order
flag = False
for iter_num in range(1,6):
    for cmd in commands:
        #if iter_num == 5 and (cmd == commands[0] or cmd == commands[1] or cmd == commands[2]):
        #   continue
        print(cmd.format(iter_num=iter_num))
        if not run_command(cmd.format(iter_num=iter_num)):
            print("Stopping script due to error.")
            flag = True
            break
        else:
            print("All commands executed successfully.")
    if flag:
        break
# for i in range(9):
#     if i != 0 or i != 5 or i != 10:
#         print("Deleting checkpoint file...")
#         try:
#             subprocess.run(f"rm -r /mnt/nas/data/yihan/Code/symbol-llm-v2/output/gsm_math_full_llama2chat_sft_iter{i}_sft_tune_llama2chat_7B", shell=True)
#         except:
#             pass
subprocess.run("python /mnt/nas/data/yihan/get_gpu/get_gpu.py", shell=True)
