import os
import json
import argparse

def load_data(task_name, part_id):
    datas = []
    if task_name == "mbpp":
        with open("/mnt/nas/data/yihan/Code/symbol-llm-v2/use_datasets/mbpp/data_mbpp.jsonl", "r", encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                data["input"] = data.pop("text")
                data["label"] = data.pop("code")
                datas.append(data)
        piece = int(len(datas)//4)+1
        return datas[(part_id-1) * piece: part_id * piece]
if __name__ == "__main__":
    pass