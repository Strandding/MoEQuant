import os

# 需要修复的文件及其对应的正确 DATASET_PATH
fixes = {
    "lm_eval/tasks/superglue.py": ('DATASET_PATH = "datasets/super_glue/super_glue.py"', 'DATASET_PATH = "super_glue"'),
    "lm_eval/tasks/hellaswag.py": ('DATASET_PATH = "datasets/hellaswag/hellaswag.py"', 'DATASET_PATH = "hellaswag"'),
    "lm_eval/tasks/winogrande.py": ('DATASET_PATH = "datasets/winogrande/winogrande.py"', 'DATASET_PATH = "winogrande"'),
    "lm_eval/tasks/openbookqa.py": ('DATASET_PATH = "datasets/openbookqa/openbookqa.py"', 'DATASET_PATH = "openbookqa"')
}

base_dir = os.getcwd() # 确保你在 code 目录下运行

for rel_path, (old_str, new_str) in fixes.items():
    file_path = os.path.join(base_dir, rel_path)
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if old_str in content:
            new_content = content.replace(old_str, new_str)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"[已修复] {rel_path}")
        elif new_str in content:
            print(f"[已是最新] {rel_path}")
        else:
            print(f"[警告] 在 {rel_path} 中未找到目标字符串，可能需要手动检查。")
    else:
        print(f"[错误] 找不到文件: {file_path}")

print("所有路径修复完成！")