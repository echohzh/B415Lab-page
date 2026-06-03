import os
import glob
from pydub import AudioSegment
from pydub.utils import make_chunks
import argparse


def pcm_to_wav_pydub(input_dir, output_dir_original, output_dir_16k):
    """
    使用pydub将PCM文件转换为WAV格式

    Args:
        input_dir: 输入PCM文件目录
        output_dir_original: 原采样率输出目录
        output_dir_16k: 16kHz重采样输出目录
    """

    # 创建输出目录
    os.makedirs(output_dir_original, exist_ok=True)
    os.makedirs(output_dir_16k, exist_ok=True)

    # 查找所有PCM文件
    pcm_files = glob.glob(os.path.join(input_dir, "*.pcm"))

    if not pcm_files:
        print("未找到PCM文件")
        return

    print(f"找到 {len(pcm_files)} 个PCM文件")

    for pcm_file in pcm_files:
        try:
            filename = os.path.basename(pcm_file)
            name_without_ext = os.path.splitext(filename)[0]

            print(f"处理文件: {filename}")

            # 从文件名推断参数（根据你的实际文件名调整）
            # 48k1c表示48kHz, 1声道
            sample_rate = 48000  # 根据文件名调整
            channels = 1
            sample_width = 2  # 16-bit = 2字节

            # 读取PCM文件
            with open(pcm_file, 'rb') as f:
                pcm_data = f.read()

            # 创建AudioSegment对象
            audio = AudioSegment(
                data=pcm_data,
                sample_width=sample_width,
                frame_rate=sample_rate,
                channels=channels
            )

            # 输出原采样率WAV
            output_file_original = os.path.join(output_dir_original, f"{name_without_ext}.wav")
            audio.export(output_file_original, format="wav")
            print(f"  原采样率: {output_file_original}")

            # 输出16kHz重采样WAV
            output_file_16k = os.path.join(output_dir_16k, f"{name_without_ext}_16k.wav")
            audio_16k = audio.set_frame_rate(16000)
            audio_16k.export(output_file_16k, format="wav")
            print(f"  16kHz: {output_file_16k}")

        except Exception as e:
            print(f"处理文件 {pcm_file} 时出错: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PCM转WAV转换器')
    parser.add_argument('--input', '-i', default='/home/xjulsk/xyj/微弱语音/', help='输入目录（默认当前目录）')
    parser.add_argument('--output_original', '-oo', default='/home/xjulsk/xyj/微弱语音/output_original', help='原采样率输出目录')
    parser.add_argument('--output_16k', '-o16', default='/home/xjulsk/xyj/微弱语音/output_16k', help='16kHz输出目录')

    args = parser.parse_args()

    pcm_to_wav_pydub(args.input, args.output_original, args.output_16k)


