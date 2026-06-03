import argparse
import os
import json,sys
import numpy as np
import soundfile as sf
import torch
import torchaudio
from natsort import natsorted
from tools.gtcrn_compute_metrics import *
from utils import *
from tqdm import tqdm
from rich.progress import track
from tools.cal_dnsmos808 import ComputeScore
@torch.no_grad()
def noisy_evaluation(noisy_dir, clean_dir, saved_dir):
    # if not os.path.exists(saved_dir):
    #     os.mkdir(saved_dir)

    audio_list = os.listdir(noisy_dir)
    # 仅保留以 .wav 结尾的文件
    audio_list = [audio for audio in audio_list if audio.endswith('.wav')]
    audio_list = natsorted(audio_list)

    metrics_total = np.zeros(8)
    num = len(audio_list)
    # 用于保存每个指标的单次结果，计算标准差
    metrics_list = []
    for i, audio in enumerate(audio_list):
        est_path = os.path.join(noisy_dir, audio)
        est_audio, sr = sf.read(est_path)

        clean_path = os.path.join(clean_dir, audio)
        clean_audio, sr = sf.read(clean_path)

        assert sr == 16000
        metrics = compute_all_metrics(clean_audio, est_audio, sr, 0)
        metrics = np.array(metrics)

        metrics_total += metrics
        metrics_list.append(metrics)  # 添加指标到列表

        # 显示当前进度百分比
        progress_percentage = (i + 1) / num * 100
        sys.stdout.write(f'\rEvaluation Processing {i + 1}/{num} ({progress_percentage:.2f}%): {audio}')
        sys.stdout.flush()

    print()

    metrics_avg = metrics_total / num
    metrics_std = np.std(metrics_list, axis=0)  # 计算标准差

    # 打印平均值和标准差
    print(f'pesq: {metrics_avg[0]:.4f} ± {metrics_std[0]:.4f}')
    print(f'csig: {metrics_avg[1]:.4f} ± {metrics_std[1]:.4f}')
    print(f'cbak: {metrics_avg[2]:.4f} ± {metrics_std[2]:.4f}')
    print(f'covl: {metrics_avg[3]:.4f} ± {metrics_std[3]:.4f}')
    print(f'ssnr: {metrics_avg[4]:.4f} ± {metrics_std[4]:.4f}')
    print(f'sisnr: {metrics_avg[5]:.4f} ± {metrics_std[5]:.4f}')
    print(f'sisdr: {metrics_avg[6]:.4f} ± {metrics_std[6]:.4f}')
    print(f'stoi: {metrics_avg[7]:.4f} ± {metrics_std[7]:.4f}')

    # 保存结果
    # result_file_path = os.path.join(saved_dir, 'metrics_avg.json')
    # results = {
    #     'pesq': metrics_avg[0],
    #     'csig': metrics_avg[1],
    #     'cbak': metrics_avg[2],
    #     'covl': metrics_avg[3],
    #     'ssnr': metrics_avg[4],
    #     'sisnr': metrics_avg[5],
    #     'sisdr': metrics_avg[6],
    #     'stoi': metrics_avg[7]
    # }

    # 如果文件存在，加载已有数据并追加新结果
    # if os.path.exists(result_file_path):
    #     with open(result_file_path, 'r') as json_file:
    #         all_results = json.load(json_file)
    # else:
    #     all_results = []

    # 追加当前实验的结果
    # all_results.append(results)

    # 保存更新后的结果
    # with open(result_file_path, 'w') as json_file:
    #     json.dump(all_results, json_file, indent=4)


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default='/home/pod/shared-nvme/CMG-v1/src/ckpt/Td_chinese/epoch67-pesq:3.261-loss_sum:0.5355:-g:0.533-d:0.001',
                    help="the path where the model is saved")

parser.add_argument("--test_dir", type=str, default='/home/dataset/Uighur-Chinese-English/Chinese/test/babble/',
                    help="noisy tracks dir to be enhanced")
parser.add_argument("--save_tracks", type=str, default=True, help="save predicted tracks or not")
parser.add_argument("--save_dir", type=str, default='/home/dataset/Uighur-Chinese-English/Chinese/test/babble/', help="where enhanced tracks to be saved")
args = parser.parse_args()


if __name__ == '__main__':
    noisy_dir_list = [d for d in os.listdir(args.test_dir) if os.path.isdir(os.path.join(args.test_dir, d))]
    for noisy_dir in noisy_dir_list:
        # noisy_dir = '/home/xyj/datasets/chinese/dev_noisy_db/SNR_-3dB'
        print(noisy_dir)
        noisy_dir = os.path.join(args.test_dir, noisy_dir)
        output_dir = os.path.join(args.save_dir, os.path.split(noisy_dir)[-1])
        noisy_evaluation( noisy_dir=noisy_dir, clean_dir = '/home/dataset/Voicebank/noisy-vctk-16k/clean_testset_wav_16k/',
                saved_dir=output_dir)

        ComputeScore().cal_dnsmos808(noisy_dir, output_dir)