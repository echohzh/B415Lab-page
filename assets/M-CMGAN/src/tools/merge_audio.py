#!/usr/bin/env python3
"""
音频分段文件拼接脚本
将目录下的音频分段文件按顺序拼接成一个完整的音频文件
"""

import os
import re
import argparse
from pathlib import Path
import glob

try:
    import soundfile as sf
    import numpy as np
except ImportError:
    print("请安装必要的库: pip install soundfile numpy")
    exit(1)

def natural_sort_key(s):
    """自然排序key函数，用于正确排序带数字的文件名"""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

def get_audio_files(directory, extensions=['.wav', '.flac', '.mp3', '.m4a']):
    """获取目录下所有音频文件，并按文件名自然排序"""
    audio_files = []
    for ext in extensions:
        audio_files.extend(glob.glob(os.path.join(directory, f'*{ext}')))
        audio_files.extend(glob.glob(os.path.join(directory, f'*{ext.upper()}')))
    
    # 去重并排序
    audio_files = list(set(audio_files))
    audio_files.sort(key=natural_sort_key)
    
    return audio_files

def merge_audio_files(input_dir, output_file, sample_rate=None):
    """
    拼接音频文件
    
    Args:
        input_dir: 输入目录路径
        output_file: 输出文件路径
        sample_rate: 目标采样率（None表示使用第一个文件的采样率）
    """
    # 获取所有音频文件
    audio_files = get_audio_files(input_dir)
    
    if not audio_files:
        print(f"错误: 在目录 {input_dir} 中没有找到音频文件")
        return False
    
    print(f"找到 {len(audio_files)} 个音频文件:")
    for f in audio_files:
        print(f"  - {os.path.basename(f)}")
    
    # 读取所有音频数据
    audio_data_list = []
    target_sr = sample_rate
    
    for i, file_path in enumerate(audio_files, 1):
        try:
            data, sr = sf.read(file_path)
            
            # 设置目标采样率
            if target_sr is None:
                target_sr = sr
            elif sr != target_sr:
                print(f"警告: 文件 {os.path.basename(file_path)} 采样率不同 ({sr} vs {target_sr})")
                # 重采样（可选，需要安装resampy或librosa）
                # 这里简单提示，实际可能需要重采样
            
            audio_data_list.append(data)
            print(f"  [{i}/{len(audio_files)}] 读取完成: {os.path.basename(file_path)} (时长: {len(data)/sr:.2f}s)")
            
        except Exception as e:
            print(f"错误: 读取文件 {file_path} 失败: {e}")
            return False
    
    # 拼接音频
    print("正在拼接音频...")
    merged_audio = np.concatenate(audio_data_list)
    
    # 保存拼接后的音频
    try:
        sf.write(output_file, merged_audio, target_sr)
        total_duration = len(merged_audio) / target_sr
        print(f"\n成功! 拼接完成:")
        print(f"  输出文件: {output_file}")
        print(f"  总时长: {total_duration:.2f} 秒")
        print(f"  采样率: {target_sr} Hz")
        print(f"  总样本数: {len(merged_audio)}")
        return True
    except Exception as e:
        print(f"错误: 保存文件失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='将分段的音频文件拼接成完整音频')
    parser.add_argument('save_dir', nargs='?', 
                       default='/home/xyj/极嘈杂语音增强结果/分段增强结果/PrimeK-Td/split_clwdwbl/',
)
    parser.add_argument('-o', '--output', 
                       help='输出文件路径（默认: 输入目录父目录下的 merged_audio.wav）')
    # parser.add_argument('output', nargs='?',
    #                    default='/home/xyj/极嘈杂语音增强结果/分段增强结果/Td-SENetsplit_jg20190311/',)
    parser.add_argument('-sr', '--sample_rate', type=int, default=None,
                       help='目标采样率（默认: 使用第一个文件的采样率）')
    
    args = parser.parse_args()
    
    # 处理输入目录
    input_dir = Path(args.save_dir)
    if not input_dir.exists():
        print(f"错误: 目录不存在 - {input_dir}")
        return
    
    if not input_dir.is_dir():
        print(f"错误: 路径不是目录 - {input_dir}")
        return
    
    # 处理输出文件路径
    if args.output:
        output_file = Path(args.output)
    else:
        # 默认输出到输入目录的父目录
        parent_dir = input_dir.parent
        output_file = parent_dir / f"{input_dir.name}_merged.wav"
    
    # 确保输出目录存在
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"输入目录: {input_dir}")
    print(f"输出文件: {output_file}")
    print("-" * 50)
    
    # 执行拼接
    success = merge_audio_files(str(input_dir), str(output_file), 16000)
    
    if not success:
        print("\n拼接失败!")

if __name__ == "__main__":
    main()