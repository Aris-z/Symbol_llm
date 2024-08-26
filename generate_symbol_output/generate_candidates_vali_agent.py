# To do: Prompt for code agent candidate generation
import torch
from transformers import AutoTokenizer, LlamaForCausalLM
import json
from tqdm import tqdm
from vllm import LLM
from vllm import SamplingParams
import os
import sys
import argparse
import logging
import subprocess
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from prompt import math_prompt, code_prompt
from use_datasets import load_data
from vali_agent_label_preference import parse_code_block
logger = logging.getLogger('self_training_logger')
logger.setLevel(logging.DEBUG)


VALI_REFLECT_NUM = 5
def parse_args():
    parser = argparse.ArgumentParser(description="self-training framework combining symbolic solver and llms")
    parser.add_argument(
        "--cur_iter",
        type=int,
        default=0,
        help="The index of the current iteration",
    )
    parser.add_argument(
        "--vllm_batchsize",
        type=int,
        default=1,
        help="batchsize for vllm",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="mbpp",
    )
    parser.add_argument(
        "--task_prefix",
        type=str,
        default="mbpp_full_llama2chat",
        help="The prefix for the dataset name, directory name and save path",
    )
    parser.add_argument(
        "--part_id",
        type=int,
        default=1,
        help="Original datasets are split into several parts, this argument returns the part index",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="llama2chat",
        help="base model",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="7B",
        help="Model size",
    )
    parser.add_argument(
        "--code_repaired",
        action="store_true",
        help="the solution code is repaired code",
    )
    parser.add_argument(
        "--few_shot",
        action="store_true",
        help="Whether to use few-shot prompting",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.few_shot and args.base_model=="llama2chat":
        PATH_TO_CONVERTED_WEIGHTS = f"/mnt/nas/data/yihan/Code/share_model/CodeLlama-7b-Instruct-hf"
        PATH_TO_CODE_DATA = "/mnt/nas/data/yihan/Code/symbol-llm-v2/score_memory/code_agent/mbpp_full_llama2chat"
    else:
        PATH_TO_CONVERTED_WEIGHTS=f"/mnt/nas/data/yihan/Code/symbol-llm-v2/output/validation_agent/{args.task_prefix}_sft_iter{args.cur_iter}_sft_tune_{args.base_model}_{args.model_size}_merged/"
        PATH_TO_CODE_DATA = "/mnt/nas/data/yihan/Code/symbol-llm-v2/new_generated_data/code_agent"
    
    llm = LLM(model=PATH_TO_CONVERTED_WEIGHTS, tensor_parallel_size=1, trust_remote_code=True)
    tokenizer = llm.get_tokenizer()
    tokenizer.pad_token = tokenizer.eos_token

    for part in [f"part{args.part_id}"]:
        if args.few_shot:
            with open(f"{PATH_TO_CODE_DATA}/{args.task_prefix}_part{args.part_id}_iter{args.cur_iter}.json", "r") as f:
                data_test_solution = json.load(f)
        else:
            with open(f"{PATH_TO_CODE_DATA}/{args.task_prefix}_part{args.part_id}_iter{args.cur_iter+1}.json", "r") as f:
                data_test_solution = json.load(f)
        
        result = []
        for i in range(0,len(data_test_solution), args.vllm_batchsize):
            result_dict = {}

            empty_dict = {               
                'code_id': i,
                'validation_id': -1,
                'question': data_test_solution[i]['question'],
                'solution_code': "",
                'response': "",
                'target': "",
                'logprobs': -99999
                }
            sampling_params = SamplingParams(max_tokens=2200, n=VALI_REFLECT_NUM)
            # prompts = [prompt]
            # instruction = "Write Python code to solve the question. "
            instruction = "Write Python unit test code to validate the solution code for given question.\nYou should try your best to generate two unit tests that detect bugs in this given code. If the solution code is not given, you should create the most difficult unit tests you can think of for this problem. Just output the code directly. DO NOT add additional explanations or introducement in the answer unless you are asked to.\n"
            try:
                solution_code = parse_code_block(data_test_solution[i]["response"])
            except:
                # result += [empty_dict for _ in range(VALI_REFLECT_NUM)]
                # continue
                solution_code = data_test_solution[i]['target']
            if args.few_shot:
                prompts = [code_prompt.VALI_INSTRUCTION + "\n" + code_prompt.VALI_PROMPT_FS + "\nProblem:\n" + data_test_solution[j]['question'] + "\nTest:\n" + "\n".join(data_test_solution[j]["test_list"])
                            + "\nSolution:\n" + solution_code + "\nThe unit test is:\n" for j in range(i,min(i+args.vllm_batchsize, len(data_test_solution)))]
            else:
                prompts = [code_prompt.VALI_INSTRUCTION + "\nProblem:\n" + data_test_solution[j]['question'] + "\n".join(data_test_solution[j]["test_list"])
                            + "\nSolution:\n" + solution_code + "\nThe unit test is:\n" for j in range(i,min(i+args.vllm_batchsize, len(data_test_solution)))]

            try:
                outputs = llm.generate(prompts, sampling_params)
            except:
                print("error")
                prompts = []
                outputs = []


            # for _ in range(5):
            if outputs:
                for j, output in enumerate(outputs):
                    # print(output)
                    response_list = output.outputs
                    for idex, response in enumerate(response_list):
                        response_text = response.text
                        result_dict = {}
                        response_text = response_text.strip()
                        result_dict['code_id'] = i
                        result_dict['validation_id'] = idex
                        result_dict['question'] = data_test_solution[i]['question']
                        result_dict['solution_code'] = solution_code
                        result_dict['response'] = response_text
                        result_dict['target'] = data_test_solution[i]['target']
                        result_dict['logprobs'] = response.cumulative_logprob / (len(response.token_ids) + 1e-8)
                        result_dict['test_list'] = data_test_solution[i]['test_list']
                        result.append(result_dict)
                        # print(response)
            else:
                result += (empty_dict)
                print("The response is empty")


            # solve the mistakes
            if len(result) % VALI_REFLECT_NUM != 0:
                result += [result_dict for _ in range(VALI_REFLECT_NUM-len(result)%VALI_REFLECT_NUM)]

            print(f"====={i+j}/{len(data_test_solution)}=====", (i+j) / len(data_test_solution))

        if args.few_shot:
            test_result_folder = f"score_memory/validation_agent/{args.task_prefix}"
            if not os.path.exists(test_result_folder):
                os.system(f"mkdir -p {test_result_folder}")
            with open(f"{test_result_folder}/{args.task_prefix}_{part}_iter0.json", 'w') as file:
                json.dump(result, file, indent=4)
        else:
            test_result_folder = f"new_generated_data/validation_agent/"
            if not os.path.exists(test_result_folder):
                os.system(f"mkdir -p {test_result_folder}")
            # with open(f"{test_result_folder}/gsm_math_full_13b_{part}_iter0.json",'w') as file:
            if args.code_repaired:
                with open(f"{test_result_folder}/{args.task_prefix}_{part}_iter{args.cur_iter + 1}_codereparied.json", 'w') as file:
                    json.dump(result, file, indent=4)
            else:
                with open(f"{test_result_folder}/{args.task_prefix}_{part}_iter{args.cur_iter+1}.json", 'w') as file:
                    json.dump(result, file, indent=4)



        print(f"[info] the result file has been saved.")
        print("==========")


if __name__ == "__main__":
    main()
