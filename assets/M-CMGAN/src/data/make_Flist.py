import os


def generate_file_mapping_txt(directory, output_txt_path):
    """
    遍历目录下的文件，生成 `文件名|路径` 格式的 `.txt` 文件

    Args:
        directory (str): 要遍历的目录路径
        output_txt_path (str): 输出的 `.txt` 文件路径
        prefix_path (str): 路径前缀（如 "VoiceBank+DEMAND/wav_clean/"）
    """
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        for filename in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, filename)):
                # 获取不带后缀的文件名
                name_without_ext = os.path.splitext(filename)[0]
                # 构造完整路径（如 "p232_001|VoiceBank+DEMAND/wav_clean/p232_001.wav"）
                line = f"{name_without_ext}|{directory}{filename}\n"
                f.write(line)

    print(f"TXT 文件已生成：{output_txt_path}")


# 示例用法
if __name__ == "__main__":
    input_dir = "/home/xjulsk/xyj/TIMIT+CHiME3/noisy/"  # 替换为你的目录路径（包含 .wav 文件的目录）
    output_txt = "TIMIT+CHiME3_noisy.txt"  # 输出的 TXT 文件名
    generate_file_mapping_txt(input_dir, output_txt)