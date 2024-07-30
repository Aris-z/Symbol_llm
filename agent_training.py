import json
import numpy as np
import pandas as pd
import os
import jsonlines
import random
import argparse
import logging
import subprocess


logger = logging.getLogger('self_training_logger')
# logger.setLevel(logging.DEBUG)



def parse_args():
    parser = argparse.ArgumentParser(description="self-training framework combining symbolic solver and llms")
    parser.add_argument(
        "--task_name",
        type=str,
        default="mbpp",
    )
    parser.add_argument(
        "--iter_num",
        type=int,
        default=10,
        help="The number of iteration for the self-training.",
    )
    parser.add_argument(
        "--vllm_batchsize",
        type=int,
        default=1,
        help="batchsize for vllm",
    )
    parser.add_argument(
        "--task_prefix",
        type=str,
        default="gsm_math_full_llama2chat",
        help="The prefix for the dataset name, directory name and save path",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="7B",
        help="Model size",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="llama2chat",
        help="base model",
    )
    parser.add_argument(
        "--node",
        type=int,
        default=1,
        help="node numbers",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    base_model = args.base_model

    # ======================================================================== #
    #                     Generate the initial steps
    # ======================================================================== #
    processes = []
    for part_id in range(1,args.node + 1):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(part_id - 1)
        process = subprocess.Popen(
            ['python', 'generate_symbol_output/generate_candidates_code_agent.py', \
             "--cur_iter", "0", "--few_shot", "--part_id", str(part_id), "--task_prefix", args.task_prefix, "--base_model",
             base_model, "--model_size", args.model_size], env=env)
        processes.append(process)
    # wait for all the process to be completed
    for process in processes:
        process.wait()
    # ensure candidates are generated successfully
    for i in range(args.node):
        assert os.path.exists(
            f"score_memory/code_agent/{args.task_prefix}/{args.task_prefix}_part{i + 1}_iter0.json") == True, "generated code candidates file does not exist..."
    
    # validation agent
    processes = []
    for part_id in range(1,args.node + 1):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(part_id - 1)
        process = subprocess.Popen(
            ['python', 'generate_symbol_output/generate_candidates_vali_agent.py', \
             "--cur_iter", "0", "--few_shot", "--part_id", str(part_id), "--task_prefix", args.task_prefix, "--base_model",
             base_model, "--model_size", args.model_size], env=env)
        processes.append(process)
    # wait for all the process to be completed
    for process in processes:
        process.wait()
    # ensure
    for i in range(args.node):
        assert os.path.exists(
            f"score_memory/code_agent/{args.task_prefix}/{args.task_prefix}_part{i + 1}_iter0.json") == True, "generated code candidates file does not exist..."
    
    
    # label preferences for the data before
    try:
        subprocess.call(["python", "agent_label_preference.py", \
                        "--task_prefix", args.task_prefix, "--cur_iter", "0", "--few_shot"])
    except:
        pass
    for i in range(args.node):
        assert os.path.exists(f"score_memory/code_agent/{args.task_prefix}/scores_{args.task_prefix}_part{i+1}_iter0.npy") == True
    for i in range(args.node):
        assert os.path.exists(f"score_memory/validation_agent/{args.task_prefix}/scores_{args.task_prefix}_part{i+1}_iter0.npy") == True


    # ======================================================================== #
    #                     Start the self-training loops
    # ======================================================================== #

    for cur_iter in range(0,args.iter_num):
        logger.info(f"Current iteration: {cur_iter}")
        # ======================================================================== #
        #                     Step 1: Generate Samples
        # ======================================================================== #
        logger.info(f"Start to generate samples for iteration-{cur_iter}")
        subprocess.call(["python", f"self-training/organize_preference_data_code_agent.py", \
                         "--task_prefix", args.task_prefix, "--cur_iter", str(cur_iter), \
                         "--model_size", args.model_size])
        subprocess.call(["python", f"self-training/organize_preference_data_vali_agent.py", \
                         "--task_prefix", args.task_prefix, "--cur_iter", str(cur_iter), \
                         "--model_size", args.model_size])

        assert os.path.exists(f"open-instruct/data/code_{args.task_prefix}_sft_iter{cur_iter}.jsonl") == True, "code training set does not exist..."
        assert os.path.exists(f"open-instruct/data/vali_{args.task_prefix}_sft_iter{cur_iter}.jsonl") == True, "validation training set does not exist..."

        # ======================================================================== #
        #                     Step 2: Training LLM (call open-instruct)
        # ======================================================================== #
        logger.info(f"Start to train LLM for iteration-{cur_iter}")
        training_bash_script = "open-instruct/scripts/finetune_with_accelerate_self_training.sh"
        # call the training scripts
        subprocess.call(["bash", training_bash_script, args.task_prefix, str(cur_iter), base_model, args.model_size])

        assert len(os.listdir((f"open-instruct/output/{args.task_prefix}_sft_iter{cur_iter}_sft_tune_{base_model}_{args.model_size}"))) != 0, "The checkpoint does not exist"


        # ======================================================================== #
        #                     Step 3: Generate Candidates with vLLM
        # ======================================================================== #
        logger.info(f"Start to generate candidates for iteration-{cur_iter}")
        processes = []
        for part_id in range(1,args.node + 1):
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(part_id - 1)
            process = subprocess.Popen(['python', 'generate_symbol_output/generate_candidates_code_agent.py', \
                                        "--cur_iter", str(cur_iter), "--part_id", str(part_id), "--task_prefix", args.task_prefix, \
                                        "--base_model", base_model,"--vllm_batchsize", str(args.vllm_batchsize), "--model_size", args.model_size], env=env)            
            processes.append(process)
        # wait for all the process to be completed
        for process in processes:
            process.wait()


        processes = []
        for part_id in range(1,args.node + 1):
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(part_id - 1)
            process = subprocess.Popen(['python', 'generate_symbol_output/generate_candidates_vali_agent.py', \
                                        "--cur_iter", str(cur_iter), "--part_id", str(part_id), "--task_prefix", args.task_prefix, \
                                        "--base_model", base_model,"--vllm_batchsize", str(args.vllm_batchsize), "--model_size", args.model_size], env=env)            
            processes.append(process)
        # wait for all the process to be completed
        for process in processes:
            process.wait()


        # ensure candidates are generated successfully
        for i in range(args.node):
            assert os.path.exists(f"new_generated_data/code_agent/{args.task_prefix}_part{i+1}_iter{cur_iter+1}.json") == True, "generated candidates file does not exist..."
        for i in range(args.node):
            assert os.path.exists(f"new_generated_data/validation_agent/{args.task_prefix}_part{i+1}_iter{cur_iter+1}.json") == True, "generated candidates file does not exist..."


        # ======================================================================== #
        #                     Step 4: Repair Candidates with vLLM
        # ======================================================================== #
        logger.info(f"Start to repair candidates for iteration-{cur_iter}")
        processes = []
        for part_id in range(1,args.node + 1):
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(part_id - 1)
            process = subprocess.Popen(['python', 'generate_symbol_output/generate_repaired_code_agent.py', \
                                        "--cur_iter", str(cur_iter), "--part_id", str(part_id), "--task_prefix", args.task_prefix, "--base_model", base_model, \
                                        "--model_size", args.model_size], env=env)
            processes.append(process)
        # wait for all the process to be completed
        for process in processes:
            process.wait()

        processes = []
        for part_id in range(1,args.node + 1):
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(part_id - 1)
            process = subprocess.Popen(['python', 'generate_symbol_output/generate_repaired_vali_agent.py', \
                                        "--cur_iter", str(cur_iter), "--part_id", str(part_id), "--task_prefix", args.task_prefix, "--base_model", base_model, \
                                        "--model_size", args.model_size], env=env)
            processes.append(process)
        # wait for all the process to be completed
        for process in processes:
            process.wait()

        # ensure repaired candidates are generated successfully
        for i in range(args.node):
            assert os.path.exists(f"new_generated_data/code_agent/{args.task_prefix}_part{i+1}_iter{cur_iter+1}_repaired.json") == True, "generated candidates file does not exist..."
        for i in range(args.node):
            assert os.path.exists(f"new_generated_data/validation_agent/{args.task_prefix}_part{i+1}_iter{cur_iter+1}_repaired.json") == True, "generated candidates file does not exist..."


        # ======================================================================== #
        #                     Step 5: Check the correctness of candidates
        # ======================================================================== #
        # label preferences for the data before
        try:
            subprocess.call(["python", "agent_label_preference.py", \
                            "--task_prefix", args.task_prefix, "--cur_iter", str(cur_iter)])
        except:
            pass
        for i in range(args.node):
            assert os.path.exists(f"score_memory/code_agent/{args.task_prefix}/scores_{args.task_prefix}_part{i+1}_iter{cur_iter+1}.npy") == True
            assert os.path.exists(f"score_memory/validation_agent/{args.task_prefix}/scores_{args.task_prefix}_part{i+1}_iter{cur_iter+1}.npy") == True

        # label perference for the repaired data
        try:
            subprocess.call(["python", "agent_label_preference.py", \
                            "--task_prefix", args.task_prefix, "--cur_iter", str(cur_iter), "--repaired",])
        except:
            pass
        for i in range(args.node):
            assert os.path.exists(f"score_memory/code_agent/{args.task_prefix}/scores_{args.task_prefix}_part{i+1}_iter{cur_iter+1}_repaired.npy") == True
            assert os.path.exists(f"score_memory/validation_agent/{args.task_prefix}/scores_{args.task_prefix}_part{i+1}_iter{cur_iter+1}_repaired.npy") == True

    logger.info("Self-Training process has finished successfully ! ! !")

if __name__ == "__main__":
    main()