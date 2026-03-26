import subprocess
import os
import sys
sys.path.append('/workspace/SpargeAttn/cv_utills')

workdir = "/workspace/SpargeAttn/cv_utills"
os.chdir(workdir)

generate_py = "/workspace/SpargeAttn/cv_utills/Wan2.1/generate.py"
prompts_file = "/workspace/SpargeAttn/Wan2.2/evaluate/datasets/video/prompts.txt"
ckpt_dir = "/workspace/SpargeAttn/cv_utills/Wan2.1/Wan2.1-T2V-1.3B"
task = "t2v-1.3B"
size = "832*480"
output_dir = "/workspace/SpargeAttn/cv_utills/Wan2.1/video_wan"

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 读取 prompts.txt
with open(prompts_file, "r", encoding="utf-8") as f:
    prompts = [line.strip() for line in f if line.strip()][0]

# 循环生成视频
for i, prompt in enumerate(prompts):
    output_path = os.path.join(output_dir, f"{i}.mp4")
    print(f"生成第 {i+1}/{len(prompts)} 个视频: {prompt} -> {output_path}")
    # from wan.modules.model import WanModel  # 已改用 model_kvclus
    # 设置 PYTHONPATH，保证能找到 triton_kernel
    env = os.environ.copy()
    env["PYTHONPATH"] = "/workspace/SpargeAttn/cv_utils:" + env.get("PYTHONPATH", "")

    subprocess.run([
        "python", generate_py,
        "--task", task,
        "--ckpt_dir", ckpt_dir,
        "--prompt", prompt,
        "--save_file", output_path,
        "--size", size,
        "--base_seed", "0"    
        ], env=env)

print("所有视频已生成完成。")
