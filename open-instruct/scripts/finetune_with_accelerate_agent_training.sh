# source /cpfs01/user/xufangzhi/anaconda3/bin/activate /cpfs01/user/xufangzhi/anaconda3/envs/flashattv2
# cd symbol-llm-v2/open-instruct
# echo "[INFO] We have successfully activate the environment."
# echo "[INFO] Start to run the shell."
export CUDA_VISIBLE_DEVICES=0,1,2,3

TASK_PREFIX=$1
ITER_NUM=$2
BASE_MODEL=$3


MODEL_SIZE=$4
AGENT_TYPE=$5

DS_FILE=/mnt/nas/data/yihan/Code/symbol-llm-v2/open-instruct/ds_configs/stage3_no_offloading_accelerate.conf
NUM_GPUS=4
BATCH_SIZE_PER_GPU=2
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training model using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

echo "$BASE_MODEL"
echo "$ITER_NUM"

if [ "$BASE_MODEL" = "llama2chat" ] || [ "$ITER_NUM" = "0" ]; then
  if [ "$MODEL_SIZE" = "7B" ]; then
    MODEL_DIR=/mnt/nas/data/yihan/Code/share_model/Llama-2-7b-chat-hf/
  elif [ "$MODEL_SIZE" = "13B" ]; then
    MODEL_DIR=/cpfs01/shared/public/public_hdd/llmeval/model_weights/llama2/model_weights_hf/llama-2-13b-chat-hf
    DS_FILE=ds_configs/stage3_offloading_accelerate.conf
  fi
elif [ "$BASE_MODEL" = "continue" ]; then
  PREV_ITER_NUM=$((ITER_NUM - 1))
  MODEL_DIR=/cpfs01/user/xufangzhi/symbol-llm-v2/open-instruct/output/${TASK_PREFIX}_sft_iter${PREV_ITER_NUM}_sft_tune_${BASE_MODEL}_${MODEL_SIZE}
fi



echo "$MODEL_DIR"

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file ${DS_FILE} \
    /mnt/nas/data/yihan/Code/symbol-llm-v2/open-instruct/open_instruct/finetune.py \
    --model_name_or_path ${MODEL_DIR} \
    --use_flash_attn \
    --tokenizer_name ${MODEL_DIR} \
    --use_slow_tokenizer \
    --train_file open-instruct/data/${TASK_PREFIX}_sft_iter${ITER_NUM}_${AGENT_TYPE}_agent.jsonl \
    --max_seq_length 2048 \
    --preprocessing_num_workers 32 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 1 \
    --output_dir ./output/${AGENT_TYPE}_agent/${TASK_PREFIX}_sft_iter${ITER_NUM}_sft_tune_${BASE_MODEL}_${MODEL_SIZE} \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1
