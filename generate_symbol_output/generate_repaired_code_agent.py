import torch
from transformers import AutoTokenizer, LlamaForCausalLM
import json
import re
from tqdm import tqdm
from vllm import LLM, SamplingParams
import os
import sys
import argparse
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from prompt import code_prompt


logger = logging.getLogger('self_training_logger')
logger.setLevel(logging.DEBUG)


CODE_REFLECT_NUM = 5
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

def extract_code_blocks(text):
    text = text.strip()
    pattern = r"\[PYTHON\](.*?)\[PYTHON\]"
    code_blocks = re.findall(pattern, text, re.DOTALL)
    if code_blocks:
        if code_blocks[0].startswith("[PYTHON]"):
            return "[PYTHON]\n" + code_blocks[0].split("[PYTHON]")[1].strip() + "\n[PYTHON]"
        else:
            return "[PYTHON]\n" + code_blocks[0].strip() + "\n[PYTHON]"
    else:
        return "[PYTHON]\n" + text.strip() + "\n[PYTHON]"


def main():
    args = parse_args()

    PATH_TO_CONVERTED_WEIGHTS = f"output/code_agent/{args.task_prefix}_sft_iter{args.cur_iter}_sft_tune_{args.base_model}_{args.model_size}_merged/"
    llm = LLM(model=PATH_TO_CONVERTED_WEIGHTS, tensor_parallel_size=1)

    for part in [f"part{args.part_id}"]:
        with open(f"new_generated_data/code_agent/{args.task_prefix}_{part}_iter{args.cur_iter+1}.json") as file:
            data_before = json.load(file)

        result = []
        for i in range(0,len(data_before)//CODE_REFLECT_NUM,1):
            result_dict = {}
            empty_dict ={
                    "id": data_before[i*CODE_REFLECT_NUM]["id"],
                    "question": data_before[i*CODE_REFLECT_NUM]["question"],
                    "response": "",
                    "target": "",
                    "logprobs": -99999
                }
            prompt = data_before[i*CODE_REFLECT_NUM]['question']
            test_list = data_before[i*CODE_REFLECT_NUM]['test_list']
            sampling_params = SamplingParams(max_tokens=4096,n=1)
            # prompts = [prompt]
            # instruction = "Repair the provided Python code to solve the given problem."
            instruction = "You are provided with a code for given problem. You can either repair and refine this code, or simply return the original solution. Just output the code directly. DO NOT add additional explanations or introducement in the answer unless you are asked to.\n"
            prompts = [code_prompt.CODE_REPAIR_INSTRUCTION + "\nProblem:\n" + prompt + "\nTest:\n" + "\n".join(test_list) \
                        + "\nThe current Python code is:\n" + extract_code_blocks(data_before[i*CODE_REFLECT_NUM+j]['response']) \
                        + "\nThe repaired code is:\n" for j in range(CODE_REFLECT_NUM)]
            try:
                outputs = llm.generate(prompts, sampling_params)
            except:
                outputs = []
            # for _ in range(5):
            if outputs:
                outputs = outputs[:CODE_REFLECT_NUM]  # trunct to 5
                for _, output in enumerate(outputs):
                    # print(output)
                    response_list = output.outputs
                    for response in response_list:
                        response_text = response.text
                        result_dict = {}
                        # response = response.split(prompt)[1].strip()
                        response_text = response_text.strip()
                        result_dict['id'] = data_before[i*CODE_REFLECT_NUM]['id']
                        result_dict['question'] = data_before[i*CODE_REFLECT_NUM]['question']
                        result_dict['response'] = response_text
                        result_dict['target'] = data_before[i*CODE_REFLECT_NUM]['target']
                        result_dict['logprobs'] = response.cumulative_logprob / (len(response.token_ids)+1e-8)
                        result_dict['test_list'] = data_before[i*CODE_REFLECT_NUM]['test_list']
                        result.append(result_dict)
                        # print(response)
            else:
                result.append(empty_dict)
                print("The response is empty")

                    # print("-----")
                    # print(data_test[i]['output'])

            # solve the mistakes
            if len(result) % 5 != 0:
                result += [result_dict for _ in range(5-len(result)%5)]
            print(f"====={i}/{len(data_before)}=====", (i) / len(data_before))


        test_result_folder = f"new_generated_data/code_agent/"
        if not os.path.exists(test_result_folder):
            os.system(f"mkdir -p {test_result_folder}")
        # with open(f"{test_result_folder}/gsm_math_full_13b_{part}_iter0.json",'w') as file:
        with open(f"{test_result_folder}/{args.task_prefix}_{part}_iter{args.cur_iter+1}_repaired.json", 'w') as file:
        # with open(f"{test_result_folder}/theoremqa_v2_iter2_repaired.json", 'w') as file:
            json.dump(result, file, indent=4)

        print(f"[info] the result file has been saved.")
        print("==========")

if __name__ == "__main__":
    main()
