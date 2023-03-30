import os
import paddle

sample_rate = 16000
wav_duration = 0.5
num_channels = 1
num_frames = sample_rate * wav_duration
wav_data = paddle.linspace(-1.0, 1.0, num_frames) * 0.1
waveform = wav_data.tile([num_channels, 1])
base_dir = os.getcwd()
filepath = os.path.join(base_dir, "test.wav")

paddle.audio.save(filepath, waveform, sample_rate)
wav_info = paddle.audio.info(filepath)