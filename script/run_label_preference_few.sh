echo "run label_preference.py"
python code_agent_label_preference.py --task_prefix mbpp_full_llama2chat --nodes 4 --few_shot
python vali_agent_label_preference.py --task_prefix mbpp_full_llama2chat --nodes 4 --few_shot
