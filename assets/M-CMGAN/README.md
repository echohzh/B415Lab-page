## M-CMGAN模型架构

```
输入: 带噪语音 STFT → 实部 + 虚部 + 幅度 (3 通道)

[DenseEncoder]
  ├── Conv2d(3→64) + InstanceNorm + PReLU
  ├── DilatedDenseNet (4 层, dilation=1,2,4,8)
  └── Conv2d(64→64, stride=2)  ← 频率下采样

[CFB]  ← 通道特征分支（门控 + 倒谱 LSTM 单元）

[TMFC]  ← 核心模块：Time-Mamba-Frequency-Conformer
  ├── TimeRecursive_Mamba（迭代 Mamba 处理时间维度）
  ├── GlobleCModule（全局通道调制）
  ├── MamformerBlock（Mamba + Attention 混合频率建模）
  └── Fmodule（频率维度调制）

[TSCB]  ← Time-Spectral Conformer Block
  ├── Time Conformer + Fmodule
  └── Freq Conformer + Fmodule

[CFB_d5]  ← 跳跃连接融合

├── [MaskDecoder]    → 幅度掩膜 → 增强幅度谱
└── [ComplexDecoder] → 复数残差 → 增强复数谱

输出: 增强语音（幅度掩膜 × 相位 + 复数残差 → ISTFT）
```

## 目录结构

```
M-CMGAN/
├── README.md
└── src/
    ├── train.py                    # 主训练脚本（GAN + Metric Discriminator）
    ├── evaluation.py               # 评估脚本（增强 + 8 项指标计算）
    ├── infer_cmgan.py              # 推理脚本（librosa 多格式音频加载）
    ├── dns400_eval.py              # DNS Challenge 批量评估
    ├── enhance_one_dir.py          # 单目录增强
    ├── eval_noisy_dirs.py          # 带噪音频评估 + DNSMOS
    ├── env.py                      # AttrDict 配置工具
    ├── utils.py                    # 信号处理工具（power compress、LearnableSigmoid 等）
    ├── pakages.txt                 # 依赖包列表
    ├── models/
    │   ├── generator_cmgan.py      # 原版 CMGAN 生成器（TSCNet, 4×TSCB）
    │   ├── gen_mcmgan1.py          # M-CMGAN 生成器（TSCNet, TMFC + TSCB + CFB + FiLM）
    │   ├── conformer.py            # Conformer/Mamba 模块库（含 MamformerBlock 等）
    │   ├── conformer_back.py       # 旧版 Conformer 实现（备份）
    │   ├── transformer.py          # Transformer 变体（Transformamba, MambaFFN 等）
    │   ├── utils.py                # 模型共享工具（与 src/utils.py 相同）
    │   ├── plt.py                  # 波形绘图工具
    │   └── modules/
    │       └── mamba_simple.py     # Mamba SSM 实现（来自 mamba-ssm 库）
    ├── data/
    │   ├── dataloader.py           # 数据集类 + DataLoader 工厂函数
    │   ├── noisy_generator.py      # 合成带噪数据（按 SNR 混合）
    │   ├── noisy_mixer.py          # 带噪音频混合器
    │   ├── make_Flist.py           # 文件列表生成
    │   ├── file_coper.py           # 文件拷贝工具
    │   ├── filecounter.py          # 文件计数工具
    │   ├── randomly_data_selecter.py  # 随机子集选择
    │   └── *.txt                   # 数据集划分文件
    ├── tools/
    │   ├── gtcrn_compute_metrics.py   # 完整指标计算（PESQ/CSIG/CBAK/COVL/SSNR/SI-SNR/SI-SDR/STOI）
    │   ├── compute_metrics.py         # 备选指标计算（含 DNSMOS 类）
    │   ├── cal_dnsmos808.py           # DNSMOS P.808 评分（ONNX）
    │   ├── calculate_metrics.py       # 并行批量指标计算
    │   ├── dns_mos.py                 # DNSMOS Azure API 评分
    │   ├── conbine_tfevent.py         # TensorBoard 事件合并
    │   ├── merge_audio.py             # 音频拼接工具
    │   ├── pcm2wav.py                 # PCM 转 WAV
    │   ├── wav_spec_plot.py           # 波形/频谱图绘制
    │   └── whisper_script.py          # Whisper ASR 测试
    └── scores/
        ├── pesq.py, stoi.py, sisdr.py, snr.py, ssnr.py  # 客观指标
        ├── csig.py, cbak.py, covl.py                     # 复合指标
        ├── llr.py, lsd.py, mcd.py                        # 谱距离指标
        ├── fwsegsnr.py, bsseval.py                       # 频域/源分离指标
        ├── mosnet/, dnsmos/, srmr/                       # MOS 预测模块
        └── helper.py
```

## 环境配置

### 依赖安装

```bash
pip install -r src/pakages.txt
```



### 数据集准备

支持多种数据集格式。以 VCTK-DEMAND 为例，目录结构如下：

```
-VCTK-DEMAND/
  -train/
    -noisy/     # 带噪训练音频
    -clean/     # 干净训练音频
  -test/
    -noisy/     # 带噪测试音频
    -clean/     # 干净测试音频
```

所有音频需为 **16 kHz** 采样率，单通道 WAV 格式。

数据集文件列表（`src/data/*.txt`）格式为 `filename|path`。

## 训练

```bash
cd src

python3 train.py \
    --data_dir /path/to/dataset \
    --noisy_dir /path/to/noisy_data \
    --batch_size 4 \
    --epochs 100 \
    --init_lr 5e-4 \
    --decay_epoch 10 \
    --cut_len 32000 \
    --save_model_dir ./ckpt/
```

### 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 100 | 训练轮数 |
| `--batch_size` | 4 | 批次大小 |
| `--init_lr` | 5e-4 | 初始学习率（判别器为 2 倍） |
| `--decay_epoch` | 10 | 学习率衰减周期（StepLR, gamma=0.7） |
| `--cut_len` | 32000 | 音频截取长度（2 秒 @ 16kHz） |
| `--save_epoch` | 25 | 每隔多少 epoch 保存一次 checkpoint |
| `--loss_weights` | [0.1, 0.9, 0.2, 0.2, 0.05] | 损失权重：[RI, 幅度, 时域, (预留), GAN] |

### 损失函数

$$L_G = 0.1 \cdot L_{RI} + 0.9 \cdot L_{mag} + 0.2 \cdot L_{time} + 0.05 \cdot L_{GAN}$$

- **RI Loss**: 实部和虚部的 MSE 损失
- **Magnitude Loss**: 幅度谱的 MSE 损失
- **Time Loss**: 时域波形的 MAE 损失
- **GAN Loss**: 度量判别器的对抗损失（MSE）

### STFT 参数

- 窗长 (n_fft): 400 (25ms @ 16kHz)
- 帧移 (hop): 100 (6.25ms)
- 窗函数: Hamming 窗
- 压缩: power-law compress (指数 0.3)

## 评估

### 单模型增强 + 指标计算(推理)

```bash
python3 evaluation.py \
    --model_path ./ckpt/best_model \
    --test_dir /path/to/VCTK-DEMAND \
    --save_dir ./enhanced_audio
```
&&
```bash
python3 infer_cmgan.py \
    --model_path ./ckpt/best_model \
    --input_dir ./noisy_audio \
    --output_dir ./enhanced_audio
```
&&
```bash
python3 dns400_eval.py/enhance_one_dir.py
```

计算 8 项客观指标（均值 ± 标准差）：

| 指标 | 说明 |
|------|------|
| **PESQ** | 感知语音质量评估 |
| **CSIG** | 信号失真复合指标 |
| **CBAK** | 背景噪声干扰复合指标 |
| **COVL** | 整体质量复合指标 |
| **SSNR** | 分段信噪比 |
| **SI-SNR** | 尺度不变信噪比 |
| **SI-SDR** | 尺度不变信号失真比 |
| **STOI** | 短时客观可懂度 |

### DNSMOS 评分

```bash
python3 tools/cal_dnsmos808.py --data_dir ./enhanced_audio --output_dir ./dnsmos_results
```




