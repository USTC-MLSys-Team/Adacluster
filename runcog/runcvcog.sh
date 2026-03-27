#!/bin/bash
# 批量测试 prompts.txt 并保存到 /workspace/SpargeAttn/finalcog
# 自动选择显存最空闲的 GPU

# 输出目录
OUTPUT_DIR="/workspace/SpargeAttn/finalcogwithqueryskip0.3"

mkdir -p "$OUTPUT_DIR"

# 模型路径
MODEL_PATH="/workspace/SpargeAttn/CogVideoX-2b"

# prompt 文件
PROMPT_FILE="/workspace/SpargeAttn/Wan2.2/evaluate/datasets/video/prompts.txt"

# 选择显存最空闲的 GPU
FREE_GPU=$(nvidia-smi --query-gpu=memory.free,index --format=csv,noheader,nounits \
           | sort -nr | head -n 1 | awk '{print $2}')
echo "使用 GPU $FREE_GPU"

export CUDA_VISIBLE_DEVICES=$FREE_GPU
export C_RATIO=64
export TOPK_NUM=64
export TOP_P=0.9

i=0
while IFS= read -r PROMPT || [[ -n "$PROMPT" ]]; do
  # 跳过空行
  if [[ -z "$PROMPT" ]]; then
    continue
  fi

  echo "生成第 $i 个视频: $PROMPT"

  /usr/bin/python /workspace/SpargeAttn/cv_utills/run_cogvideo.py \
    --prompt "$PROMPT" \
    --model_path "$MODEL_PATH" \
    --output_path "$OUTPUT_DIR/tmp_${i}.mp4" \
    --num_frames 81 \
    --num_inference_steps 3 \
    --method kvclus \
    --use_spas_sage_attn \



  # 统一改名为 0.mp4, 1.mp4 ...
  if [[ -f "$OUTPUT_DIR/tmp_${i}.mp4" ]]; then
    mv "$OUTPUT_DIR/tmp_${i}.mp4" "$OUTPUT_DIR/${i}.mp4"
    echo "第 $i 个视频生成成功，保存为 $OUTPUT_DIR/${i}.mp4\n"
  else
    echo "第 $i 个视频生成失败"
  fi

  i=$((i+1))
done < "$PROMPT_FILE"

echo "全部完成，结果在 $OUTPUT_DIR"
