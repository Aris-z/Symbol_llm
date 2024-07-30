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
from prompt import math_prompt,code_prompt
from use_datasets import load_data

logger = logging.getLogger('self_training_logger')
logger.setLevel(logging.DEBUG)



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
        "--few_shot",
        action="store_true",
        help="Whether to use few-shot prompting",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.few_shot and args.base_model=="llama2chat":
        PATH_TO_CONVERTED_WEIGHTS = f"/mnt/nas/data/yihan/Code/share_model/Llama-2-7b-chat-hf"
    else:
        PATH_TO_CONVERTED_WEIGHTS=f"/mnt/nas/data/yihan/Code/symbol-llm-v2/output/test_agent/{args.task_prefix}_sft_iter{args.cur_iter}_sft_tune_{args.base_model}_{args.model_size}_merged/"

    llm = LLM(model=PATH_TO_CONVERTED_WEIGHTS, tensor_parallel_size=1, trust_remote_code=True)
    tokenizer = llm.get_tokenizer()
    tokenizer.pad_token = tokenizer.eos_token

    for part in [f"part{args.part_id}"]:
        data_test = load_data(args.task_name, args.part_id)
        result = []
        for i in range(0,len(data_test),args.vllm_batchsize):
            result_dict = {}
            sampling_params = SamplingParams(max_tokens=2200,n=5)
            # prompts = [prompt]
            # instruction = "Write Python code to solve the question. "
            instruction = "Write Python code to solve the question.\nThe returned value of the program is supposed to be the right answer to any possible values for this question and bug free. Just ONLY output the code directly. DO NOT add additional explanations or introducement in the answer unless you are asked to.\n"

            if args.few_shot:
                prompts = [code_prompt.CODE_INSTRUCTION + "\n" + code_prompt.CODE_PROMPT_FS + "\nProblem:\n" + data_test[j]['input']
                            + "\nTest:\n" + data_test[j]["test_list"][0] + "\nThe solution code is:\n" for j in range(i,min(i+args.vllm_batchsize, len(data_test)))]
            else:
                prompts = [code_prompt.CODE_INSTRUCTION + "\n" + code_prompt.CODE_PROMPT_FS + "\nProblem:\n" + data_test[j]['input']
                            + "\nTest:\n" + data_test[j]["test_list"][0] + "\nThe solution code is:\n" for j in range(i,min(i+args.vllm_batchsize, len(data_test)))]

            try:
                # print(prompts)
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
                    for response in response_list:
                        response_text = response.text
                        result_dict = {}
                        response_text = response_text.strip()
                        result_dict['id'] = i
                        result_dict['question'] = data_test[i]['input']
                        result_dict['response'] = response_text
                        result_dict['target'] = data_test[i]['label']
                        result_dict['logprobs'] = response.cumulative_logprob / (len(response.token_ids)+1e-8)
                        result_dict['test_list'] = data_test[i]['test_list']
                        result.append(result_dict)
                        # print(response)
            else:
                result_dict = {}

                result_dict['id'] = i
                result_dict['question'] = data_test[i]['input']
                result_dict['response'] = ""
                result_dict['target'] = data_test[i]['label']
                result_dict['logprobs'] = -99999
                result_dict['test_list'] = data_test[i]['test_list']
                result.append(result_dict)
                print("The response is empty")


            # solve the mistakes
            if len(result) % 5 != 0:
                result += [result_dict for _ in range(5-len(result)%5)]

            print(f"====={i+j}/{len(data_test)}=====", (i+j) / len(data_test))

        if args.few_shot:
            test_result_folder = f"score_memory/test_agent/{args.task_prefix}"
            if not os.path.exists(test_result_folder):
                os.system(f"mkdir -p {test_result_folder}")
            with open(f"{test_result_folder}/{args.task_prefix}_{part}_iter0.json", 'w') as file:
                json.dump(result, file, indent=4)
        else:
            test_result_folder = f"new_generated_data/test_agent"
            if not os.path.exists(test_result_folder):
                os.system(f"mkdir -p {test_result_folder}")
            # with open(f"{test_result_folder}/gsm_math_full_13b_{part}_iter0.json",'w') as file:
            with open(f"{test_result_folder}/{args.task_prefix}_{part}_iter{args.cur_iter+1}.json", 'w') as file:
                json.dump(result, file, indent=4)


        print(f"[info] the result file has been saved.")
        print("==========")


if __name__ == "__main__":
    main()