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
from prompt import math_prompt, theoremqa_prompt


logger = logging.getLogger('self_training_logger')
logger.setLevel(logging.DEBUG)



def parse_args():
    parser = argparse.ArgumentParser(description="self-training framework combining symbolic solver and llms")
    parser.add_argument(
        "--domain",
        type=str,
        default="math",
        help="The name of the domain [math,agent,logic].",
    )
    parser.add_argument(
        "--cur_iter",
        type=int,
        default=0,
        help="The index of the current iteration",
    )
    parser.add_argument(
        "--vllm_batchsize",
        type=int,
        default=4,
        help="batchsize for vllm",
    )
    parser.add_argument(
        "--task_prefix",
        type=str,
        default="gsm_math_full_v15",
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
        help="Base Model",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="7B",
        help="Model size",
    )
    args = parser.parse_args()
    return args


def extract_code_blocks(text):
    text = text.strip()
    pattern = r"```(.*?)```"
    code_blocks = re.findall(pattern, text, re.DOTALL)
    if code_blocks:
        if code_blocks[0].startswith("python\n"):
            return "```python\n"+code_blocks[0].split("python\n")[1].strip()+"\n```"
        else:
            return "```python\n"+code_blocks[0]+"\n```"
    else:
        return "```python\n"+text+"\n```".strip()


def main():
    args = parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.part_id-1)

    # PATH_TO_CONVERTED_WEIGHTS="/cpfs01/user/xufangzhi/symbol-llm-v2/open-instruct/output/theoremqa_v2_sft_iter1_sft_tune_llama2chat_7B"
    PATH_TO_CONVERTED_WEIGHTS = f"/cpfs01/user/xufangzhi/symbol-llm-v2/open-instruct/output/{args.task_prefix}_sft_iter{args.cur_iter}_sft_tune_{args.base_model}_{args.model_size}/"
    # PATH_TO_CONVERTED_WEIGHTS = "/cpfs01/user/xufangzhi/symbol-llm-v2/open-instruct/output/sft_iter0_sft_tune_dpo_iter0_7B"
    # PATH_TO_CONVERTED_WEIGHTS="/cpfs01/shared/public/public_hdd/llmeval/model_weights/llama2/model_weights_hf/llama-2-13b-chat-hf"
    # PATH_TO_CONVERTED_TOKENIZER="/cpfs01/shared/NLP-A100/NLP-A100_hdd/symbol-llm/symbol-llm_7b_instruct"
    PATH_TO_CONVERTED_TOKENIZER = "/cpfs01/shared/public/public_hdd/llmeval/model_weights/llama2/model_weights_hf/llama-2-7b-chat-hf"
    # available_gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    llm = LLM(model=PATH_TO_CONVERTED_WEIGHTS, tensor_parallel_size=1)

    batch_size = 1


    for beam in [1]:
        for part in [f"part{args.part_id}"]:
        # for part in ["part20"]:

            # part = "part1"
            # test_path = f"test_dataset/{dataset}/{dataset}_test.json"
            # test_path = f"symbol-llm-v2/open-instruct/data/metamathqa_{part}.json"
            test_path = f"symbol-llm-v2/open-instruct/data/gsm_math_full_{part}.json"
            # test_path = f"symbol-llm-v2/open-instruct/data/theoremqa_train.json"
            with open(test_path) as file:
                data_test = json.load(file)
            print(test_path)

            # with open(f"new_generated_data/theoremqa_v2_iter2.json") as file:
            with open(f"new_generated_data/{args.task_prefix}_{part}_iter{args.cur_iter+1}.json") as file:
            # with open(f"new_generated_data/gsm_math_full_v14_{part}_iter{args.cur_iter + 1}.json") as file:
                data_before = json.load(file)

            assert len(data_test)==len(data_before)//5
            result = []
            for i in range(0,len(data_test),1):
                result_dict = {}
                prompt = data_test[i]['input']
                sampling_params = SamplingParams(max_tokens=4096,n=1)
                # prompts = [prompt]
                # instruction = "Repair the provided Python code to solve the given problem."
                instruction = "You are provided with a Python code to solve the given problem. You can either repair and refine it, or simply return the original solution.\n"
                prompts = [instruction + "\nThe question is:\n" + data_test[i]['input'] \
                            + "\nThe current Python code is:\n" + extract_code_blocks(data_before[i*5+j]['response']) \
                            + "\nThe solution code is:\n"
                           for j in range(5)]
                try:
                    outputs = llm.generate(prompts, sampling_params)
                except:
                    outputs = []
                # for _ in range(5):
                if outputs:
                    outputs = outputs[:5]  # trunct to 5
                    for j, output in enumerate(outputs):
                        # print(output)
                        response_list = output.outputs
                        for response in response_list:
                            response_text = response.text
                            # response = llm.generate(prompts, sampling_params)[0].outputs[0].text
                            # generate_ids = model.generate(inputs.input_ids, max_length=1200, num_beams=beam, do_sample=True)
                            # response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                            result_dict = {}
                            # response = response.split(prompt)[1].strip()
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
                    # response = response.split(prompt)[1].strip()
                    result_dict['id'] = i
                    result_dict['question'] = data_test[i]['input']
                    result_dict['response'] = ""
                    result_dict['target'] = data_test[i]['label']
                    result_dict['logprobs'] = -99999
                    result.append(result_dict)
                    print("The response is empty")

                        # print("-----")
                        # print(data_test[i]['output'])

                # solve the mistakes
                if len(result) % 5 != 0:
                    result += [result_dict for _ in range(5-len(result)%5)]
                print(f"====={i+j}/{len(data_test)}=====", (i+j) / len(data_test))


            test_result_folder = f"new_generated_data/"
            if not os.path.exists(test_result_folder):
                os.system(f"mkdir -p {test_result_folder}")
            # with open(f"{test_result_folder}/gsm_math_full_13b_{part}_iter0.json",'w') as file:
            with open(f"{test_result_folder}/{args.task_prefix}_{part}_iter{args.cur_iter+1}_repaired.json", 'w') as file:
            # with open(f"{test_result_folder}/theoremqa_v2_iter2_repaired.json", 'w') as file:
                json.dump(result, file, indent=4)

            # with open(f"{test_result_folder}/prediction.txt",'w') as file:
            #     for i in range(len(result)):
            #         if result[i]['response']=="":
            #             file.write("wrong")
            #         else:
            #             file.write(result[i]['response'])
            #         file.write('\n')
            print(f"[info] the result file has been saved.")
            print("==========")

if __name__ == "__main__":
    main()