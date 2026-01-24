import json

def merge_jsonl(input_file, output_file, group_size=32):
    """
    将jsonl文件每group_size行合并为一行，主要合并text字段。
    """
    print(f"正在处理文件: {input_file} ...")
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        chunk = []
        count = 0
        
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                chunk.append(data)
            except json.JSONDecodeError:
                print(f"跳过无效的 JSON 行: {line[:50]}...")
                continue
            
            # 当收集满8条时，进行合并
            if len(chunk) == group_size:
                # 提取所有 text 内容并用换行符拼接
                # 注意：这里假设你的字段名是 "text"，如截图所示
                merged_text = "\n".join([item.get("text", "") for item in chunk])
                
                # 创建新的一行数据
                new_record = {"text": merged_text}
                
                # 写入新文件
                f_out.write(json.dumps(new_record, ensure_ascii=False) + '\n')
                
                # 重置缓冲区
                chunk = []
                count += 1
        
        # 处理剩余不足8条的数据（如果有的话）
        if chunk:
            merged_text = "\n".join([item.get("text", "") for item in chunk])
            new_record = {"text": merged_text}
            f_out.write(json.dumps(new_record, ensure_ascii=False) + '\n')
            count += 1

    print(f"处理完成！")
    print(f"原始条目数: 2400 (预期)")
    print(f"合并后条目数: {count}")
    print(f"结果已保存至: {output_file}")

# 配置参数
input_filename = 'ebss_qwen3next_selfbuild.jsonl'
output_filename = 'ebss_qwen3next_selfbuild_merged.jsonl'

# 执行函数
if __name__ == "__main__":
    merge_jsonl(input_filename, output_filename)