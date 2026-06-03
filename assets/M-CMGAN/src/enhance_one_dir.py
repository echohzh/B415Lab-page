import argparse
import os

import numpy as np
import soundfile as sf
import torch
import torchaudio
from natsort import natsorted
from models import generator_td as generator
from tools.compute_metrics import compute_metrics
from utils import *


@torch.no_grad()
def enhance_one_track(model, audio_path, saved_dir, cut_len, n_fft=400, hop=100, save_tracks=False,device='cuda:1'):

    name = os.path.split(audio_path)[-1]
    noisy, sr = torchaudio.load(audio_path)
    # assert sr == 16000
    noisy = noisy.to(device)
#归一化
    c = torch.sqrt(noisy.size(-1) / torch.sum((noisy ** 2.0), dim=-1))
    noisy = torch.transpose(noisy, 0, 1)
    noisy = torch.transpose(noisy * c, 0, 1)

    length = noisy.size(-1)     #320
    # print(length)
    frame_num = int(np.ceil(length / 100))  # 4
    padded_len = frame_num * 100  # 400    padding_len = padded_len - length  # 400-320=80
    noisy = torch.cat([noisy, noisy[:, :padding_len]], dim=-1)
    if padded_len > cut_len:
        batch_size = int(np.ceil(padded_len / cut_len))
        while 100 % batch_size != 0:
            batch_size += 1
        noisy = torch.reshape(noisy, (batch_size, -1))
    # frame_num = int(np.ceil(length / cut_len))# 4
    # padded_len = frame_num * cut_len  #400
    # padding_len = padded_len - length  # 400-320=80
    # noisy = torch.cat([noisy, noisy[:, :padding_len]], dim=-1)
    # noisy = torch.reshape(noisy, (-1, cut_len))
    noisy_spec = torch.stft(noisy, n_fft, hop, window=torch.hamming_window(n_fft).to(device), onesided=True)
    noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
    est_real, est_imag,_ = model(noisy_spec)
    est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)

    est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
    est_audio = torch.istft(est_spec_uncompress, n_fft, hop, window=torch.hamming_window(n_fft).to(device),
                            onesided=True)
    est_audio = est_audio/c
    est_audio = torch.flatten(est_audio)[:length].cpu().numpy()
    # assert len(est_audio) == length
    if save_tracks:
        saved_path = os.path.join(saved_dir, name)
        sf.write(saved_path, est_audio, sr)

    return est_audio, length


def enhanced_one_dir(model_path, noisy_dir, save_tracks, saved_dir,device='cuda:1'):
    # cuda_device = '0'
    # os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    torch.device(device)
    map_location = torch.device(device)
    n_fft = 400
    model = generator.TSCNet(num_channel=64, num_features=n_fft//2+1).to(device)
    model.load_state_dict((torch.load(model_path,map_location=device)))
    model.eval()
    if not os.path.exists(saved_dir):
        os.mkdir(saved_dir)

    audio_list = os.listdir(noisy_dir)
    audio_list = natsorted(audio_list)
    for audio in audio_list:
        noisy_path = os.path.join(noisy_dir, audio)
        est_audio, length = enhance_one_track(model, noisy_path, saved_dir, 16000*6, n_fft, n_fft//4, save_tracks)



def evaluation( noisy_dir, clean_dir,  saved_dir):
    # n_fft = 400
    # model = generator.TSCNet(num_channel=64, num_features=n_fft//2+1).cuda()
    # model.load_state_dict((torch.load(model_path)))
    # model.eval()

    if not os.path.exists(saved_dir):
        os.mkdir(saved_dir)

    audio_list = os.listdir(noisy_dir)
    audio_list = natsorted(audio_list)
    num = len(audio_list)
    metrics_total = np.zeros(6)
    for audio in audio_list:
        est_path = os.path.join(saved_dir, audio)
        est_audio, sr = sf.read(est_path)
        clean_path = os.path.join(clean_dir, audio)
        clean_audio, sr = sf.read(clean_path)
        assert sr == 16000
        metrics = compute_metrics(clean_audio, est_audio, sr, 0)
        metrics = np.array(metrics)
        metrics_total += metrics

    metrics_avg = metrics_total / num
    print('pesq: ', metrics_avg[0], 'csig: ', metrics_avg[1], 'cbak: ', metrics_avg[2], 'covl: ',
          metrics_avg[3], 'ssnr: ', metrics_avg[4], 'stoi: ', metrics_avg[5])


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default='/home/xyj/Experience/ckpt-TdSENet_C/epoch93-pesq:3.363-loss_sum:0.5242:-g:0.521-d:0.003',
                    help="the path where the model is saved")

parser.add_argument("--test_dir", type=str, default='/home/xyj/Experience/CMG-v1/src/noisy_audios/split_audio2/',
                    help="noisy tracks dir to be enhanced")

parser.add_argument("--save_tracks", type=str, default=True, help="save predicted tracks or not")
parser.add_argument("--save_dir", type=str, default='/home/xyj/Enh_audio/Noisy-C/Td_audio2', help="where enhanced tracks to be saved")
args = parser.parse_args()


if __name__ == '__main__':


    noisy_dir = os.path.join(args.test_dir)

    #增强算法1处理
    enhanced_one_dir(model_path = args.model_path, noisy_dir = noisy_dir,
               save_tracks = args.save_tracks, saved_dir = args.save_dir)

    #得到增强算法1的评价指标
    # evaluation( noisy_dir=noisy_dir, clean_dir = clean_dir,
    #            saved_dir=args.save_dir)

