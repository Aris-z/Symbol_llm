ITER_NUM=$1
python self-training/organize_preference_data_code_agent.py --task_prefix mbpp_full_llama2chat --cur_iter $ITER_NUM --model_size 7B
python self-training/organize_preference_data_vali_agent.py --task_prefix mbpp_full_llama2chat --cur_iter $ITER_NUM --model_size 7B