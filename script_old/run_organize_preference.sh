ITER_NUM=$1
python self-training/organize_preference_data_llama2chat.py --task_prefix gsm_math_full_llama2chat --cur_iter $ITER_NUM --model_size 7B