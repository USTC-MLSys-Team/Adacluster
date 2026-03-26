# 读取文件
with open("model_flashinfer_v2.py", "r") as f:
    lines = f.readlines()

# 找到 flashinfer 调用的位置并修改
new_lines = []
for i, line in enumerate(lines):
    new_lines.append(line)
    if "q_flashinfer = q.flatten(0, 1).permute(1, 0, 2).contiguous()" in line:
        # 添加类型转换
        new_lines.append("        # FlashInfer 需要 float16/bfloat16\n")
        new_lines.append("        dtype = torch.float16 if q.dtype == torch.float32 else q.dtype\n")
        new_lines.append("        q_flashinfer = q_flashinfer.to(dtype)\n")
        new_lines.append("        k_flashinfer = k.flatten(0, 1).permute(1, 0, 2).contiguous().to(dtype)\n")
        new_lines.append("        v_flashinfer = v.flatten(0, 1).permute(1, 0, 2).contiguous().to(dtype)\n")
        # 跳过原来的 k 和 v 赋值
        for j in range(i+1, min(i+5, len(lines))):
            if "k_flashinfer" in lines[j] or "v_flashinfer" in lines[j]:
                continue
        # 修改下一行
        idx = i + 2
        lines[idx] = lines[idx].replace("k_flashinfer = k.flatten", "# k_flashinfer = k.flatten")
        lines[idx + 1] = lines[idx + 1].replace("v_flashinfer = v.flatten", "# v_flashinfer = v.flatten")

# 写回
with open("model_flashinfer_v2.py", "w") as f:
    f.writelines(new_lines)

print("已修复数据类型问题")
