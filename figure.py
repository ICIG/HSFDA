import matplotlib.pyplot as plt
import librosa.display
import numpy as np

# 准备一些示例数据，这里用随机数代替真实的语音信号
clean_signal, sr_clean = librosa.load('/home/zt/ssl/base_demo_BSSE-SE/yuyin/clean.wav', sr=None)  # 假设clean.wav是干净的语音信号
noisy_signal, sr_noisy = librosa.load('/home/zt/ssl/base_demo_BSSE-SE/yuyin/noisy.wav', sr=None)  # 假设noisy.wav是带噪声的语音信号
noise_signal, sr_noise = librosa.load('/home/zt/ssl/base_demo_BSSE-SE/yuyin/cmgan.wav', sr=None)  # 假设noise.wav是噪声信号

# 绘制clean波形图
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
librosa.display.waveshow(clean_signal, sr=sr_clean)
plt.title('Clean Signal')

# 绘制noisy波形图
plt.subplot(3, 1, 2)
librosa.display.waveshow(noisy_signal, sr=sr_noisy)
plt.title('Noisy Signal')

# 绘制noise波形图
plt.subplot(3, 1, 3)
librosa.display.waveshow(noise_signal, sr=sr_noise)
plt.title('Noise Signal')

plt.tight_layout()
plt.show()
