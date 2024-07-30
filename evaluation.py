import os
import json
from code_agent_label_preference import *
from vllm import LLM, SamplingParams
from prompt import math_prompt
from use_datasets import load_data

# PATH_TO_CODE = "score_memory/code_agent/mbpp_full_llama2chat"
# PATH_TO_CODE = "open-instruct/data"
PATH_TO_CODE = "new_generated_data/code_agent"

part_num = 4
def main():
    data_code = []
    for part in range(1, part_num + 1):
        with open(f"{PATH_TO_CODE}/mbpp_full_llama2chat_part{part}_iter6.json", "r") as f:
            data_code += json.load(f)
    # with open(f"{PATH_TO_CODE}/label_unit_test_part1_iter1.json", "r") as f:
    #     for line in f:
    #         data_code.append(json.loads(line))
    pass_5 = []
    acc = []
    for i in range(len(data_code)//5):
        # if i > 1:
        #     exit()
        flag = False            
        # print(data_label[i]['test_list'])
        for j in range(5):
            code = data_code[i * 5 + j]["response"]
            # try:
            #     code = re.sub(r'\bdef\s+\w+\s*\(', "def solution(", code)
            #     start_token = re.escape("def solution(")
            #     end_token = re.escape(re.search(r'return\s+.+\n', code).group())
            #     pattern = fr"{start_token}(.*?){end_token}"
            #     parse_code = re.search(pattern, code, re.DOTALL).group()
            # except:
            #     print(code)
            #     acc.append(False)
            #     continue
            # print(code)
            target = ""
            for item in data_code[i * 5 + j]["test_list"]:
                target += item + "\n"
            try:
                parse_code = parse_code_block(code)
            except ValueError:
                # print(code)
                acc.append(False)
                continue
            exc_codes = parse_code + "\n" + target
            # print(code)
            # print(sample["test_code"])
            output = exc_code(exc_codes, validation="evaluation")
            acc.append(output)
            flag = flag or output
            # print(f"{output}" * 50)
        pass_5.append(flag)
    print(f"Overall: {sum(acc)/len(acc)}")
    print(f"Pass@5: {sum(pass_5)/len(pass_5)}")

def test():
    data_code = []
    with open(f"{PATH_TO_CODE}/mbpp_full_llama2chat_sft_iter6_code_agent.jsonl", "r") as f:
        for line in f:
            data_code.append(json.loads(line))
    acc = []
    for i in range(len(data_code)):
        # if i > 1:
        #     exit()          
        targets = data_code[i]['test_list']
        # print(data_label[i]['test_list'])
        target = ''
        for k in range(len(targets)):
            target += (targets[k] + "\n")
        code = data_code[i]["completion"]
        try:
            parse_code = parse_code_block(code)
        except ValueError:
            acc.append(False)
            continue
        exc_codes = parse_code + "\n" + target
        output = exc_code(exc_codes, validation="evaluation")
        acc.append(output)
    print(f"Overall: {sum(acc)/len(acc)}")
    # print(f"Pass@5: {sum(pass_5)/len(pass_5)}")

if __name__ == "__main__":
    main()
    # test()

