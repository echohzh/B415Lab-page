# from compute_metrics import *
import json
import os

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf
from natsort import natsorted
from tqdm import tqdm
import csv
import pandas as pd

class ComputeScore:
    def __init__(self, primary_model_path='/home/xyj/Experiments/CMG-v1/src/scores/dnsmos/DNSMOS/sig_bak_ovr.onnx',
                 p808_model_path='/home/xyj/Experiments/CMG-v1/src/scores/dnsmos/DNSMOS/model_v8.onnx'):
        self.onnx_sess = ort.InferenceSession(primary_model_path)
        self.p808_onnx_sess = ort.InferenceSession(p808_model_path)

    def audio_melspec(self, audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True):
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=frame_size + 1, hop_length=hop_length,
                                                  n_mels=n_mels)
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
        return mel_spec.T

    def cal_mos(self, audio, sampling_rate):
        fs = sampling_rate
        actual_length = len(audio)
        INPUT_LENGTH = 9.01
        len_samples = int(INPUT_LENGTH * fs)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)

        num_hops = int(np.floor(len(audio) / fs) - INPUT_LENGTH) + 1
        hop_len_samples = fs
        predicted_p808_mos = []

        for idx in range(num_hops):
            audio_seg = audio[int(idx * hop_len_samples): int((idx + INPUT_LENGTH) * hop_len_samples)]
            if len(audio_seg) < len_samples:
                continue

            input_features = np.array(audio_seg).astype('float32')[np.newaxis, :]
            p808_input_features = np.array(self.audio_melspec(audio=audio_seg[:-160])).astype('float32')[
                np.newaxis, :, :]
            oi = {'input_1': input_features}
            p808_oi = {'input_1': p808_input_features}
            p808_mos = self.p808_onnx_sess.run(None, p808_oi)[0][0][0]
            predicted_p808_mos.append(p808_mos)

        P808_MOS = np.mean(predicted_p808_mos)
        return [P808_MOS]

    def cal_dnsmos808(self, file_path, save_path, save_per_audio_csv=True):
        """
        计算DNSMOS分数并保存结果

        Args:
            file_path: 音频文件目录
            save_path: 保存结果的目录
            save_per_audio_csv: 是否保存每个音频的分数到CSV文件
        """
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        audio_list = os.listdir(file_path)
        audio_list = [audio for audio in audio_list if audio.endswith('.wav')]
        audio_list = natsorted(audio_list)

        metrics_total = np.zeros(1)
        num = len(audio_list)

        # 用于保存每个音频的分数
        per_audio_results = []

        # 用于保存每个指标的单次结果，计算标准差
        metrics_list = []

        for audio in tqdm(audio_list, desc="Evaluating DNSMOS"):
            est_path = os.path.join(file_path, audio)
            # 使用 librosa 读取音频，并重采样到 16kHz
            est_audio, sr = librosa.load(est_path, sr=16000)  # sr=16000 会自动重采样到16kHz

            # 确保音频是单声道（如果是立体声，librosa会自动转换为单声道）
            if len(est_audio.shape) > 1:
                est_audio = librosa.to_mono(est_audio)
            metrics = self.cal_mos(est_audio, sr)

            # 保存每个音频的分数
            per_audio_results.append({
                'audio_name': audio,
                'P808_MOS': metrics[0]
            })

            metrics_total += metrics
            metrics_list.append(metrics)

        # 方法1：使用csv模块保存每个音频的分数
        if save_per_audio_csv:
            csv_path = os.path.join(save_path, 'per_audio_scores.csv')
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['audio_name', 'P808_MOS']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(per_audio_results)
            print(f"每个音频的分数已保存到: {csv_path}")

        # 方法2：使用pandas保存（更美观，推荐）
        # 如果安装了pandas，可以使用这个方法
        try:
            import pandas as pd
            df = pd.DataFrame(per_audio_results)
            excel_path = os.path.join(save_path, 'per_audio_scores.xlsx')
            df.to_excel(excel_path, index=False)
            print(f"每个音频的分数已保存到Excel: {excel_path}")
        except ImportError:
            print("提示: 安装pandas可以保存为Excel格式: pip install pandas openpyxl")

        print()

        metrics_avg = metrics_total / num
        metrics_std = np.std(metrics_list, axis=0)

        print(f'P808_MOS: {metrics_avg[0]:.4f} ± {metrics_std[0]:.4f}')

        # 保存平均值结果
        result_file_path = os.path.join(save_path, 'dnsmos_avg.json')
        results = {
            'P808_MOS': float(metrics_avg[0]),
            'P808_MOS_std': float(metrics_std[0]),
            'num_samples': num,
            'csv_file': 'per_audio_scores.csv' if save_per_audio_csv else None
        }

        # 如果文件存在，加载已有数据并追加新结果
        if os.path.exists(result_file_path):
            with open(result_file_path, 'r') as json_file:
                all_results = json.load(json_file)
        else:
            all_results = []

        all_results.append(results)

        with open(result_file_path, 'w') as json_file:
            json.dump(all_results, json_file, indent=4)

        return per_audio_results, results


if __name__ == '__main__':

    file_path = '/home/xyj/极嘈杂语音增强结果/PrimeK-Td-96/'
    save_path = '/home/xyj/极嘈杂语音增强结果/PrimeK-Td-96/'
    ComputeScore().cal_dnsmos808(file_path, save_path)
    # noisy_dir_list = [d for d in os.listdir('/home/xyj/result_DyTMamba-zh/DNS400/') if os.path.isdir(os.path.join('/home/xyj/result_DyTMamba-zh/DNS400/', d))]
    #
    # for noisy_dir in noisy_dir_list:
    #     noisy_dir = os.path.join('/home/xyj/result_DyTMamba-zh/DNS400/', noisy_dir)
    #     output_dir = os.path.join('/home/xyj/result_DyTMamba-zh/DNS400/', os.path.split(noisy_dir)[-1])
    #     ComputeScore().cal_dnsmos808(file_path, save_path)