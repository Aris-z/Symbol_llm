ITER_NUM=$1

echo "run label_preference.py"
python code_agent_label_preference.py --task_prefix mbpp_full_llama2chat --cur_iter $ITER_NUM --nodes 4
python vali_agent_label_preference.py --task_prefix mbpp_full_llama2chat --cur_iter $ITER_NUM --nodes 4

echo "run label_preference.py repaired"
python code_agent_label_preference.py --task_prefix mbpp_full_llama2chat --cur_iter $ITER_NUM --repaired --nodes 4
python vali_agent_label_preference.py --task_prefix mbpp_full_llama2chat --cur_iter $ITER_NUM --repaired --nodes 4
