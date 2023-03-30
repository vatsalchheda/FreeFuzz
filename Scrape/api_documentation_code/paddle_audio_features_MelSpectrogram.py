import paddle
from paddle.audio.features import MelSpectrogram

sample_rate = 16000
wav_duration = 0.5
num_channels = 1
num_frames = sample_rate * wav_duration
wav_data = paddle.linspace(-1.0, 1.0, num_frames) * 0.1
waveform = wav_data.tile([num_channels, 1])

feature_extractor = MelSpectrogram(sr=sample_rate, n_fft=512, window = 'hann', power = 1.0)
feats = feature_extractor(waveform)