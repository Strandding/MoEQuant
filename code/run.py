import torch
import sys
import time
import argparse
import multiprocessing

def occupy_gpu_task(device_id, size_gb):
    """单个 GPU 占用的子进程任务"""
    try:
        # 核心：在子进程内部确保 CUDA 环境正确
        device = torch.device(f"cuda:{device_id}")
        
        # 计算需要的浮点数数量 (float32 占用 4 bytes)
        n_elements = (size_gb * 1024 * 1024 * 1024) // 4
        
        print(f"进程启动：正在尝试在 GPU {device_id} 上申请 {size_gb}GB 显存...")
        
        # 申请显存
        dummy_tensor = torch.empty(n_elements, dtype=torch.float32, device=device).fill_(0)
        
        print(f"成功！已占用 GPU {device_id} 约 {size_gb}GB 显存。")
        
        # 保持运行
        while True:
            time.sleep(60)
            
    except torch.cuda.OutOfMemoryError:
        print(f"错误：GPU {device_id} 显存不足，无法申请 {size_gb}GB。")
    except Exception as e:
        print(f"GPU {device_id} 发生错误: {e}")

if __name__ == "__main__":
    # 【修复关键】强制使用 spawn 启动方式
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="多 GPU 显存占用脚本")
    parser.add_argument("devices", type=int, nargs='+', help="GPU 编号列表")
    parser.add_argument("--size", type=int, default=38, help="每张卡的 GB 大小")
    
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("未检测到 CUDA 环境。")
        sys.exit(1)

    print(f"计划在 GPU {args.devices} 上各占用 {args.size}GB...")
    print("按 Ctrl+C 停止所有占用并释放显存。")

    processes = []
    for dev_id in args.devices:
        p = multiprocessing.Process(target=occupy_gpu_task, args=(dev_id, args.size))
        p.start()
        processes.append(p)

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\n正在释放显存并退出...")
        for p in processes:
            p.terminate()
            p.join()
        print("已停止。")