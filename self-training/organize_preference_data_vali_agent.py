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
from use_datasets import load_data
from prompt import code_prompt

logger = logging.getLogger('self_training_logger')
logger.setLevel(logging.DEBUG)


VALI_REFLECT_NUM = 5
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
            if scores[i][j]>=1:
                num += 1
                break
    return num


def extract_code_blocks(text):
    """
    :param text: original response from the model
    :return: parsed code form
    """
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
        default=1,
        help="The number of iteration for the self-training.",
    )
    parser.add_argument(
        "--task_prefix",
        type=str,
        default="mbpp_full_llama2chat",
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
    PART_NUM = 4

    response_pool = set()
    chosen_pool = collections.defaultdict(list)
    rejected_pool = collections.defaultdict(list)
    for iter_idx in range(args.cur_iter+1):
        # [Data Load] read scores for each iteration
        scores, candidates = [], []
        scores_repaired, candidates_repaired = [], []
        for i in range(1, PART_NUM + 1):
            if iter_idx == 0:
                part_name = f"part{i}"
                score = np.load(f"score_memory/validation_agent/{args.task_prefix}/scores_{args.task_prefix}_{part_name}_iter{iter_idx}.npy").tolist()
                with open(f"score_memory/validation_agent/{args.task_prefix}/{args.task_prefix}_{part_name}_iter{iter_idx}.json",'r') as file:
                    data = json.load(file)
                scores += score
                candidates += data
            else:
                part_name = f"part{i}"
                score = np.load(f"score_memory/validation_agent/{args.task_prefix}/scores_{args.task_prefix}_{part_name}_iter{iter_idx}.npy").tolist()
                with open(f"new_generated_data/validation_agent/{args.task_prefix}_{part_name}_iter{iter_idx}.json", 'r') as file:
                    data = json.load(file)
                scores += score
                candidates += data

                # load self-repaired samples
                score_repaired = np.load(f"score_memory/validation_agent/{args.task_prefix}/scores_{args.task_prefix}_{part_name}_iter{iter_idx}_repaired.npy").tolist()
                with open(f"new_generated_data/validation_agent/{args.task_prefix}_{part_name}_iter{iter_idx}_repaired.json", 'r') as file:
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
                if candidates[i * VALI_REFLECT_NUM]["validation_id"] == -1:
                    continue
                for j in range(VALI_REFLECT_NUM):
                    if scores[i][j] < scores_repaired[i][j] or \
                            (scores[i][j] == scores_repaired[i][j] and candidates[i * VALI_REFLECT_NUM+j]['logprobs'] < candidates_repaired[i * VALI_REFLECT_NUM+j]['logprobs']):
                        scores[i][j] = scores_repaired[i][j]
                        candidates[i * VALI_REFLECT_NUM + j] = candidates_repaired[i * VALI_REFLECT_NUM + j]
            print(f"orginal effective samples after self-repair: {count_effective_samples(scores)}")



        include_id = set()
        freq_id = defaultdict(int)
        explored_id = set()
        preference_data = []
        preference_data_sft = []
        preference_data_filtered = []
        preference_data_best = []
        # [All] get chosen pool and rejected pool
        explored_id_cur = set()
        repair_id_cur = set()
        for i in range(len(scores)):  # for each origin sample, origin id: i
            chosen_candidates_idx_list = []
            rejected_candidates_idx_list = []
            if scores[i][0] < 0:
                chosen_candidates_list = []
                rejected_candidates_list = []
            else:
                for j in range(VALI_REFLECT_NUM):  # generate VALI_REFLECT_NUM code for each (x,y)
                    response = extract_code_blocks(candidates[i * VALI_REFLECT_NUM+j]['response'])
                    # choose_flag = 1 if iter_idx == 0 else 2
                    choose_flag = 2
                    if scores[i][j] >= choose_flag and response not in response_pool and (re.search(r'assert\s\w*\(.*\)\s*==\s*.*', response) and "def " not in response):
                    # if scores[i][j] == 1 and response not in response_pool:
                        chosen_candidates_idx_list.append(i * VALI_REFLECT_NUM + j)
                        response_pool.add(response)
                    elif response not in response_pool:
                        rejected_candidates_idx_list.append(i * VALI_REFLECT_NUM + j)
                        response_pool.add(response)

                # update chosen pool for each sample
                chosen_candidates_list, rejected_candidates_list = [], []
                for m in range(len(chosen_candidates_idx_list)):
                    chosen_candidates_list.append(candidates[chosen_candidates_idx_list[m]])
                for m in range(len(rejected_candidates_idx_list)):
                    rejected_candidates_list.append(candidates[rejected_candidates_idx_list[m]])

            chosen_pool[str(i)] = sorted(chosen_pool[str(i)]+chosen_candidates_list, key=lambda x: x.get('logprobs', float('-inf')), reverse=True)
            rejected_pool[str(i)] = sorted(rejected_pool[str(i)]+rejected_candidates_list, key=lambda x: x.get('logprobs', float('-inf')), reverse=True)

            if chosen_pool[str(i)]:
                # [SFT] normal sft data
                explore_num = 0
                for m in range(min(10, max(1,len(chosen_pool[str(i)])-1))):  # choose at most 10 candidates
                    data_dict = {}
                    data_dict['origin_id'] = str(i)
                    # data_dict['source'] = ground_truth[i]['source']
                    data_dict['type'] = "self-explore"
                    data_dict['prompt'] = code_prompt.VALI_INSTRUCTION + "\nProblem:" + \
                                          chosen_pool[str(i)][m]['question'] + "\nTest:\n" + chosen_pool[str(i)][m]['test_list'][0] + "\nThe solution code is:\n" + chosen_pool[str(i)][m]['solution_code'] + "\nThe unit test code is:\n"
                    data_dict['completion'] = extract_code_blocks(chosen_pool[str(i)][m]['response'])
                    explore_num += 1
                    explored_id_cur.add(str(i))
                    include_id.add(str(i))
                    preference_data_sft.append(data_dict)
                    if m==0:
                        preference_data_best.append(data_dict)


                # [SFT] self-repair sft data
                repair_num = max(0, min(2, len(chosen_pool[str(i)])-explore_num, len(rejected_pool[str(i)])))
                for m in range(repair_num):
                    data_dict = {}
                    data_dict['origin_id'] = str(i)
                    # data_dict['source'] = ground_truth[i]['source']
                    data_dict['type'] = "self-repair"
                    data_dict['prompt'] = code_prompt.VALI_REPAIR_INSTRUCTION + \
                                          "\nProblem:" + rejected_pool[str(i)][m]['question'] + "\nTest:\n" + rejected_pool[str(i)][m]['test_list'][0] + "\nThe current Python code is:\n" + rejected_pool[str(i)][m]["solution_code"] + "\nThe current unit test code is:\n" + \
                                          extract_code_blocks(rejected_pool[str(i)][m]['response']) + \
                                          "\nThe repaired unit test code is:\n"
                    data_dict['completion'] = extract_code_blocks(chosen_pool[str(i)][explore_num+m]['response'])
                    repair_id_cur.add(str(i))
                    preference_data_sft.append(data_dict)


        print(f"The iteration {iter_idx} has: DPO data {len(preference_data_filtered)}, SFT data {len(preference_data_sft)}")
        print(f"The current normal sft data: {len(explored_id_cur)}, self-repair sft data: {len(repair_id_cur)}")
        print(f"The number of used samples: {len(include_id)}")
        #     print(f"Current iteration contains: {len(include_id_gsm)} gsm samples, and {len(include_id_math)} math samples, {len(include_id_gsm)/len(include_id_math)}")
        print("-" * 30)

    with open(f"logs/{args.task_prefix}_validation_log.txt","a+") as file:
        file.write(f"orginal effective samples before self-repair: {effective_samples}\n")
        file.write(f"orginal effective samples after self-repair: {count_effective_samples(scores)}\n")
        file.write(f"The iteration {iter_idx} has: SFT data {len(preference_data_sft)}\n")
        file.write(f"The current normal sft data: {len(explored_id_cur)}, self-repair sft data: {len(repair_id_cur)}\n")
        file.write(f"The number of used samples: {len(include_id)}\n")
        file.write("-" * 30)
        file.write("\n")


    with jsonlines.open(f"open-instruct/data/{args.task_prefix}_sft_iter{args.cur_iter}_validation_agent.jsonl",'w') as file:
        random.shuffle(preference_data_sft)
        for i in range(len(preference_data_sft)):
            file.write(preference_data_sft[i])

if __name__ == "__main__":
    main()
