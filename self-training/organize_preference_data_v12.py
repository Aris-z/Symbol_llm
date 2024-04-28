"""
input: generated predictions, ground-truth label, executed results via symbolic solver
output: organized training data, including dpo samples and sft samples
[current version] add organized self-repair data, only SFT
"""

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
        default=1,
        help="The number of iteration for the self-training.",
    )
    parser.add_argument(
        "--task_prefix",
        type=str,
        default="gsm_math_full_v11",
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
        default=0,
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
        with open(f"symbol-llm-v2/open-instruct/data/gsm_math_full_{part_name}.json", 'r') as file:
            data = json.load(file)
        ground_truth += data


    include_id = set()
    freq_id = defaultdict(int)
    explored_id = set()
    preference_data = []
    preference_data_sft = []
    preference_data_filtered = []
    for iter_idx in range(args.cur_iter+1):
        # [Data Load] read scores for each iteration
        scores, candidates = [], []
        scores_repaired, candidates_repaired = [], []
        for i in range(1, part_num + 1):
            if iter_idx == 0:
                part_name = f"part{i}"
                score = np.load(f"symbol-llm-v2/score_memory/{args.task_prefix}/scores_gsm_math_full_{part_name}_iter{iter_idx}.npy").tolist()
                with open(f"symbol-llm-v2/score_memory/{args.task_prefix}/gsm_math_full_{part_name}_iter{iter_idx}.json",'r') as file:
                    data = json.load(file)
                scores += score
                candidates += data
            else:
                part_name = f"part{i}"
                score = np.load(f"symbol-llm-v2/score_memory/{args.task_prefix}/scores_{args.task_prefix}_{part_name}_iter{iter_idx}.npy").tolist()
                with open(f"new_generated_data/{args.task_prefix}_{part_name}_iter{iter_idx}.json",
                          'r') as file:
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
        if iter_idx >= 1:
            print(f"orginal effective samples: {count_effective_samples(scores)}")
            for i in range(len(scores)):
                for j in range(5):
                    if scores[i][j] != 1 and scores_repaired[i][j] == 1:
                        scores[i][j] = 1
                        candidates[i * 5 + j] = candidates_repaired[i * 5 + j]
            print(f"orginal effective samples after self-repair: {count_effective_samples(scores)}")

        # [All] compostional pool
        hard_flag = []
        explored_id_cur = set()
        repair_id_cur = set()
        for i in range(len(scores)):  # for each origin sample, origin id: i
            chosen_candidates_set = set()
            rejected_candidates_set = set()
            for j in range(5):  # generate 5 code for each (x,y)
                if scores[i][j] == 1:
                    chosen_candidates_set.add(i * 5 + j)
                else:
                    rejected_candidates_set.add(i * 5 + j)

            if chosen_candidates_set:
                chosen_candidates_list = list(chosen_candidates_set)
                rejected_candidates_list = list(rejected_candidates_set)
                chosen_gold_index = []
                has_sft = False
                for m in range(len(chosen_candidates_list)):
                    # [SFT] normal sft data
                    data_dict = {}
                    data_dict['origin_id'] = str(i)
                    data_dict['source'] = ground_truth[i]['source']
                    data_dict['prompt'] = "Write a Python code to solve the problem.\n" + "\nThe question is:" + \
                                          ground_truth[i]['input'] + "\nThe solution code is:\n"
                    data_dict['completion'] = extract_code_blocks(candidates[chosen_candidates_list[m]]['response'])

                    if ground_truth[i]['input'].strip() in data_dict['completion']:
                        include_id.add(str(i))
                        explored_id_cur.add(str(i))
                        freq_id[str(i)] += 1
                        has_sft = True
                        chosen_gold_index.append(m)

                if has_sft:
                    data_dict = {}
                    data_dict['origin_id'] = str(i)
                    data_dict['source'] = ground_truth[i]['source']
                    data_dict['prompt'] = "Write a Python code to solve the problem.\n" + "\nThe question is:" + \
                                          ground_truth[i]['input'] + "\nThe solution code is:\n"
                    data_dict['completion'] = extract_code_blocks(candidates[chosen_candidates_list[chosen_gold_index[0]]]['response'])
                    preference_data_sft.append(data_dict)

                # [SFT] self-repair sft data
                if 1<len(chosen_candidates_list)<5 and len(chosen_gold_index)>1 and has_sft:
                    data_dict = {}
                    data_dict['origin_id'] = str(i)
                    data_dict['source'] = ground_truth[i]['source']
                    data_dict['prompt'] = "Repair the provided Python code to solve the given problem.\n" + "\nThe question is:" + \
                                          ground_truth[i]['input'] + "\nThe current Python code is:\n" + \
                                          extract_code_blocks(candidates[rejected_candidates_list[-1]]['response']) + \
                                          "\nThe repaired code is as follows:\n"
                    data_dict['completion'] = extract_code_blocks(candidates[chosen_candidates_list[chosen_gold_index[-1]]]['response'])
                    repair_id_cur.add(str(i))
                    preference_data_sft.append(data_dict)

        print(f"The iteration {iter_idx} has: DPO data {len(preference_data_filtered)}, SFT data {len(preference_data_sft)}")
        print(f"The current normal sft data: {len(explored_id_cur)}, self-repair sft data: {len(repair_id_cur)}")
        print(f"The number of used samples: {len(include_id)}")
        #     print(f"Current iteration contains: {len(include_id_gsm)} gsm samples, and {len(include_id_math)} math samples, {len(include_id_gsm)/len(include_id_math)}")
        print("-" * 30)
        #     with open(f"../SymbolLLMv2/open-instruct/data/preference_dpo_iter{iter_idx}.json",'w') as file:
        #         json.dump(preference_data_filtered,file,indent=4)
        # with open(f"symbol-llm-v2/open-instruct/data/preference_sft_iter{iter_idx}.json", 'w') as file:
        #     json.dump(preference_data_sft, file, indent=4)

    with open(f"symbol-llm-v2/logs/{args.task_prefix}_log.txt","a") as file:
        file.write(f"The iteration {iter_idx} has: DPO data {len(preference_data_filtered)}, SFT data {len(preference_data_sft)}\n")
        file.write(f"The current normal sft data: {len(explored_id_cur)}, self-repair sft data: {len(repair_id_cur)}\n")
        file.write(f"The number of used samples: {len(include_id)}\n")
        file.write("-" * 30)
        file.write("\n")


    with jsonlines.open(f"symbol-llm-v2/open-instruct/data/{args.task_prefix}_sft_iter{args.cur_iter}.jsonl",'w') as file:
        random.shuffle(preference_data_sft)
        for i in range(len(preference_data_sft)):
            file.write(preference_data_sft[i])

if __name__ == "__main__":
    main()