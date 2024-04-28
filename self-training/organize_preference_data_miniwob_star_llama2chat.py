"""
input: generated predictions, ground-truth label, executed results via symbolic solver
output: organized training data, including dpo samples and sft samples
[current version] add organized self-repair data, only SFT
"""
import collections
import json
import numpy as np
import pandas as pd
import os
import jsonlines
import random
import re
from collections import defaultdict

import re
import math
import argparse
import logging
import subprocess

logger = logging.getLogger('self_training_logger')
logger.setLevel(logging.DEBUG)


def calculate_code_bleu(reference, candidate):
    """
    :param reference:
    :param candidate:
    :return: calculated metric
    """
    # 清理代码中的空格和换行符
    reference = re.sub(r'\s', '', reference)
    candidate = re.sub(r'\s', '', candidate)
    # 创建n-gram字典
    reference_ngrams = create_ngram_dict(reference, 4)
    candidate_ngrams = create_ngram_dict(candidate, 4)
    # 计算n-gram匹配数
    matching_ngrams = 0
    for ngram in candidate_ngrams:
        if ngram in reference_ngrams:
            matching_ngrams += min(candidate_ngrams[ngram], reference_ngrams[ngram])
    # 计算候选翻译的长度
    candidate_length = len(candidate)
    # 计算参考翻译的长度
    reference_length = len(reference)
    # 计算精确度
    precision = matching_ngrams / candidate_length
    # 计算召回率
    recall = matching_ngrams / reference_length
    # 计算CodeBLEU
    code_bleu = math.exp(0.5 * math.log(precision * recall))
    return code_bleu


def create_ngram_dict(code, n):
    ngrams = {}
    for i in range(len(code) - n + 1):
        ngram = code[i:i + n]
        if ngram in ngrams:
            ngrams[ngram] += 1
        else:
            ngrams[ngram] = 1
    return ngrams


def count_effective_samples(scores):
    num = 0
    for i in range(len(scores)):
        for j in range(5):
            if scores[i][j]==1:
                num += 1
                break
    return num


def extract_code_blocks(text):
    """
    :param text: original response from the model
    :return: parsed code form
    """
    text = text.strip()
    pattern = r"```(.*?)```"
    code_blocks = re.findall(pattern, text, re.DOTALL)
    if code_blocks:
        if code_blocks[0].startswith("python\n"):
            return "```python\n" + code_blocks[0].split("python\n")[1].strip() + "\n```"
        else:
            return "```python\n" + code_blocks[0] + "\n```"
    else:
        return "```python\n" + text + "\n```".strip()

def extract_action_blocks(text):
    pattern = r"```(.*?)```"
    code_blocks = re.findall(pattern, text, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()
    else:
        return text.strip()


def parse_args():
    parser = argparse.ArgumentParser(description="self-training framework combining symbolic solver and llms")
    parser.add_argument(
        "--domain",
        type=str,
        default="math",
        help="The name of the domain [math,agent,logic].",
    )
    parser.add_argument(
        "--iter_num",
        type=int,
        default=0,
        help="The number of iteration for the self-training.",
    )
    parser.add_argument(
        "--task_prefix",
        type=str,
        default="miniwob_v17_llama2chat",
        help="The prefix for the dataset name, directory name and save path",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="7B",
        help="Model size",
    )
    parser.add_argument(
        "--cur_iter",
        type=int,
        default=2,
        help="The index of the current iteration",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    """
    load original ground-truth data
    """
    part_num = 8
    ground_truth = []
    for i in range(1, part_num+1):
        part_name = f"part{i}"
        with open(f"symbol-llm-v2/open-instruct/data/miniwob_{part_name}.json", 'r') as file:
            data = json.load(file)
        ground_truth += data


    preference_data = []
    preference_data_sft = []
    preference_data_filtered = []
    preference_data_best = []
    repair_id = set()
    include_id = set()
    freq_id = defaultdict(int)
    explored_id = set()
    for iter_idx in range(args.cur_iter+1):
        response_pool = set()
        chosen_pool = collections.defaultdict(list)
        rejected_pool = collections.defaultdict(list)
        # [Data Load] read scores for each iteration
        scores, candidates = [], []
        scores_repaired, candidates_repaired = [], []
        for i in range(1, part_num + 1):
            if iter_idx == 0:
                part_name = f"part{i}"
                score = np.load(f"symbol-llm-v2/score_memory/{args.task_prefix}/scores_{args.task_prefix}_{part_name}_iter{iter_idx}.npy").tolist()
                with open(f"symbol-llm-v2/score_memory/{args.task_prefix}/{args.task_prefix}_{part_name}_iter{iter_idx}.json",'r') as file:
                    data = json.load(file)
                scores += score
                candidates += data
                print(len(candidates))
                print(len(scores))

            else:
                part_name = f"part{i}"
                score = np.load(f"symbol-llm-v2/score_memory/{args.task_prefix}/scores_{args.task_prefix}_{part_name}_iter{iter_idx}.npy").tolist()
                with open(f"new_generated_data/{args.task_prefix}_{part_name}_iter{iter_idx}.json", 'r') as file:
                    data = json.load(file)
                scores += score
                candidates += data

                # load self-repaired samples
                score_repaired = np.load(f"symbol-llm-v2/score_memory/{args.task_prefix}/scores_{args.task_prefix}_{part_name}_iter{iter_idx}_repaired.npy").tolist()
                with open(f"new_generated_data/{args.task_prefix}_{part_name}_iter{iter_idx}_repaired.json", 'r') as file:
                    data_repaired = json.load(file)
                scores_repaired += score_repaired
                candidates_repaired += data_repaired
        print(f"Original samples: {len(scores)}")

        # merge self-repaired samples
        effective_samples = 0
        if iter_idx >= 1:
            print(f"orginal effective samples: {count_effective_samples(scores)}")
            effective_samples = count_effective_samples(scores)
            for i in range(len(scores)):
                for j in range(5):
                    if scores[i][j] != 1 and scores_repaired[i][j] == 1 or \
                            (scores[i][j]==1 and scores_repaired[i][j]==1 and candidates[i*5+j]['logprobs']<candidates_repaired[i*5+j]['logprobs']):
                        scores[i][j] = 1
                        candidates[i * 5 + j] = candidates_repaired[i * 5 + j]
            print(f"orginal effective samples after self-repair: {count_effective_samples(scores)}")


        # [All] get chosen pool and rejected pool
        for i in range(len(scores)):  # for each origin sample, origin id: i
            chosen_candidates_idx_list = []
            rejected_candidates_idx_list = []
            for j in range(5):  # generate 5 code for each (x,y)
                # if scores[i][j] == 1 and response not in response_pool:
                if scores[i][j] == 1:
                    chosen_candidates_idx_list.append(i * 5 + j)
                else:
                    rejected_candidates_idx_list.append(i * 5 + j)

            # update chosen pool for each sample
            chosen_candidates_list, rejected_candidates_list = [], []
            for m in range(len(chosen_candidates_idx_list)):
                chosen_candidates_list.append(candidates[chosen_candidates_idx_list[m]])
            for m in range(len(rejected_candidates_idx_list)):
                rejected_candidates_list.append(candidates[rejected_candidates_idx_list[m]])

            chosen_pool[str(i)] = chosen_candidates_list
            random.shuffle(chosen_pool[str(i)])


            # [SFT] normal sft data
            explore_num = 0
            for m in range(min(2, len(chosen_pool[str(i)]))):  # choose at most 10 candidates
                data_dict = {}
                data_dict['origin_id'] = str(i)
                # data_dict['source'] = ground_truth[i]['source']
                data_dict['type'] = "self-explore"
                data_dict['prompt'] = "You are required to navigate the web. To accomplish the task, use methods in Agent class to generate actions, with the following functions. type(characters: str): Type a string via the keyboard. click_xpath(xpath: str): Click an HTML element with a valid XPath. press(key_type: str): Press a key on the keyboard (enter, space, arrowleft, arrowright, backspace, arrowup, arrowdown, command+a, command+c, command+v). click_option(xpath: str): Click an option HTML element in a list with a valid XPath. movemouse(xpath: str): Move the mouse cursor on an HTML element with a valid XPath.\n" \
                                      + "\nThe observation is:" + candidates[i*5+j]['question'] + "\nThe action is:\n"
                data_dict['completion'] = "```\n" + extract_action_blocks(chosen_pool[str(i)][m]['response']).strip() + "\n```"
                explore_num += 1
                explored_id.add((iter_idx, str(i)))
                include_id.add((iter_idx, str(i)))
                preference_data_sft.append(data_dict)



        print(f"The iteration {iter_idx} has: DPO data {len(preference_data_filtered)}, SFT data {len(preference_data_sft)}")
        print(f"The current normal sft data: {len(explored_id)}, self-repair sft data: {len(repair_id)}")
        print(f"The number of used samples: {len(include_id)}")
        #     print(f"Current iteration contains: {len(include_id_gsm)} gsm samples, and {len(include_id_math)} math samples, {len(include_id_gsm)/len(include_id_math)}")
        print("-" * 30)

    with open(f"symbol-llm-v2/logs/{args.task_prefix}_log.txt","a") as file:
        file.write(f"orginal effective samples before self-repair: {effective_samples}\n")
        file.write(f"orginal effective samples after self-repair: {count_effective_samples(scores)}\n")
        file.write(f"The iteration {iter_idx} has: SFT data {len(preference_data_sft)}\n")
        file.write(f"The current normal sft data: {len(explored_id)}, self-repair sft data: {len(repair_id)}\n")
        file.write(f"The number of used samples: {len(include_id)}\n")
        file.write("-" * 30)
        file.write("\n")


    with jsonlines.open(f"symbol-llm-v2/open-instruct/data/{args.task_prefix}_sft_iter{args.cur_iter}.jsonl",'w') as file:
        if len(preference_data_sft)<3000:
            preference_data_sft = [item for item in preference_data_sft for _ in range(2)]
        else:
            preference_data_sft = [item for item in preference_data_sft for _ in range(2)]
        random.shuffle(preference_data_sft)
        for i in range(len(preference_data_sft)):
            file.write(preference_data_sft[i])


if __name__ == "__main__":
    main()