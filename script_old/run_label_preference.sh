ITER_NUM=$1

echo "run label_preference.py"
python pal/scripts/label_preference.py --task_prefix gsm_math_full_llama2chat --cur_iter $ITER_NUM --nodes 3
echo "run label_preference.py repaired"
python pal/scripts/label_preference.py --task_prefix gsm_math_full_llama2chat --cur_iter $ITER_NUM --repaired --nodes 3
