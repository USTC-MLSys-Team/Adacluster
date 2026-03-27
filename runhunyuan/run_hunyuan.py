import gc
import os
import subprocess
from typing import Iterable, List, Sequence, Tuple
import torch
from diffusers import HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video
from modify_hunyuan_video.pipeline_hunyuan_video import HunyuanVideoPipeline
from modify_hunyuan_video.hunyuan_video_attn_processor_kvclus_withrightclusmaxclus import HunyuanVideoAttnProcessor2_0_kvclus

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ENABLE_PCA_ANALYSIS"] = "0"

PROMPT_FILE = "/workspace/SpargeAttn/evaluate/datasets/video/prompts.txt"
OUTPUT_DIR = "/workspace/SpargeAttn/benchmark_results/maxclus"
MODEL_ID = "/workspace/SpargeAttn/HunyuanVideo1"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_prompts():
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()][:4]

def build_pipeline(device: str) -> HunyuanVideoPipeline:
    print(f"[{device}] Loading model with maxclus processor...")
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        MODEL_ID, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    pipe = HunyuanVideoPipeline.from_pretrained(
        MODEL_ID, transformer=transformer, torch_dtype=torch.float16
    )
    processor = HunyuanVideoAttnProcessor2_0_kvclus()
    pipe.set_sparse_attn_processor(processor)
    pipe.set_skip_ratio(0)
    pipe.vae.enable_tiling()
    pipe.enable_model_cpu_offload(device=device)
    return pipe

def generate_videos_on_gpu(worker_name: str, gpu_id: int, tasks):
    if not tasks:
        return
    device_str = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    pipe = build_pipeline(device_str)
    
    for global_idx, prompt in tasks:
        print(f"\n[{worker_name}] Generating video {global_idx+1}: {prompt[:50]}...")
        import time
        start = time.time()
        generator = torch.Generator(device=device_str).manual_seed(global_idx)
        try:
            output = pipe(
                prompt=prompt,
                height=720, width=1280,
                num_frames=81, num_inference_steps=30,
                generator=generator,
            ).frames[0]
            elapsed = time.time() - start
            save_path = os.path.join(OUTPUT_DIR, f"{global_idx}.mp4")
            export_to_video(output, save_path, fps=15)
            print(f"[{worker_name}] Done in {elapsed:.1f}s: {save_path}")
        except Exception as e:
            print(f"[{worker_name}] ERROR: {e}")
        finally:
            del output
            torch.cuda.empty_cache()
            gc.collect()

def main():
    import sys
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    prompts = load_prompts()
    start_idx = gpu_id
    tasks = [(start_idx, prompts[start_idx])]
    generate_videos_on_gpu(f"GPU-{gpu_id}", gpu_id, tasks)

if __name__ == "__main__":
    main()
