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
from prompt import math_prompt, theoremqa_prompt

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
        "--task_prefix",
        type=str,
        default="gsm_math_full_llama2chat",
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
        PATH_TO_CONVERTED_WEIGHTS=f"/mnt/nas/data/yihan/Code/symbol-llm-v2/output/{args.task_prefix}_sft_iter{args.cur_iter}_sft_tune_{args.base_model}_{args.model_size}/"

    llm = LLM(model=PATH_TO_CONVERTED_WEIGHTS, tensor_parallel_size=1, trust_remote_code=True)
    tokenizer = llm.get_tokenizer()
    tokenizer.pad_token = tokenizer.eos_token

    for part in [f"part{args.part_id}"]:
        test_path = f"open-instruct/data/gsm_math_full_{part}.json"
        # test_path = f"symbol-llm-v2/open-instruct/data/theoremqa_train.json"
        with open(test_path) as file:
            data_test = json.load(file)
        print(test_path)
        result = []
        for i in range(0,len(data_test),args.vllm_batchsize):
            result_dict = {}
            prompt = data_test[i]['input']
            sampling_params = SamplingParams(max_tokens=2200,n=5)
            # prompts = [prompt]
            # instruction = "Write Python code to solve the question. "
            instruction = "Write Python code to solve the question.\nThe returned value of the program is supposed to be the answer. It should be integer or float or list of integer/float.\n"

            if args.few_shot:
                prompts = [instruction + "\n" + math_prompt.MATH_PROMPT_FS + "\nThe question is:\n" + data_test[j]['input']
                            + "\nThe solution code is:\n" for j in range(i,min(i+args.vllm_batchsize, len(data_test)))]
            else:
                prompts = [instruction + "\nThe question is:\n" + data_test[j]['input']
                            + "\nThe solution code is:\n" for j in range(i,min(i+args.vllm_batchsize, len(data_test)))]

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
                        result.append(result_dict)
                        # print(response)
            else:
                result_dict = {}

                result_dict['id'] = i
                result_dict['question'] = data_test[i]['input']
                result_dict['response'] = ""
                result_dict['target'] = data_test[i]['label']
                result_dict['logprobs'] = -99999
                result.append(result_dict)
                print("The response is empty")


            # solve the mistakes
            if len(result) % 5 != 0:
                result += [result_dict for _ in range(5-len(result)%5)]

            print(f"====={i+j}/{len(data_test)}=====", (i+j) / len(data_test))

        if args.few_shot:
            test_result_folder = f"score_memory/{args.task_prefix}"
            if not os.path.exists(test_result_folder):
                os.system(f"mkdir -p {test_result_folder}")
            with open(f"{test_result_folder}/{args.task_prefix}_{part}_iter0.json", 'w') as file:
                json.dump(result, file, indent=4)
        else:
            test_result_folder = f"new_generated_data/"
            if not os.path.exists(test_result_folder):
                os.system(f"mkdir -p {test_result_folder}")
            # with open(f"{test_result_folder}/gsm_math_full_13b_{part}_iter0.json",'w') as file:
            with open(f"{test_result_folder}/{args.task_prefix}_{part}_iter{args.cur_iter+1}.json", 'w') as file:
                json.dump(result, file, indent=4)


        print(f"[info] the result file has been saved.")
        print("==========")


if __name__ == "__main__":
    main()