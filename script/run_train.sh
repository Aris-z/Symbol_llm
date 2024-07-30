ITER_NUM=$1
AGENT_TYPE=$2
bash open-instruct/scripts/finetune_lora_with_accelerate_agent_training.sh mbpp_full_llama2chat ${ITER_NUM} llama2chat 7B $AGENT_TYPE