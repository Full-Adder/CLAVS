import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram

# 加载音频文件
sample_rate, audio_data = wavfile.read('SaliencyDataFuc/V18_Soccer2.wav')

# 计算频谱图
frequencies, times, Sxx = spectrogram(audio_data, fs=sample_rate)

# 绘制频谱图
plt.figure(figsize=(10, 4))
plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx))  # 使用对数尺度绘制频谱图
plt.title('Spectrogram')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.colorbar(label='Intensity (dB)')
plt.ylim(0, 5000)  # 设置频率范围
plt.savefig('SaliencyDataFuc/spectrogram.png')