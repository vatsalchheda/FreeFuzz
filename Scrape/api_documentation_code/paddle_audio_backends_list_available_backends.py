import paddle

sample_rate = 16000
wav_duration = 0.5
num_channels = 1
num_frames = sample_rate * wav_duration
wav_data = paddle.linspace(-1.0, 1.0, num_frames) * 0.1
waveform = wav_data.tile([num_channels, 1])
wav_path = "./test.wav"

current_backend = paddle.audio.backends.get_current_backend()
print(current_backend) # wave_backend, the default backend.
backends = paddle.audio.backends.list_available_backends()
# default backends is ['wave_backend']
# backends is ['wave_backend', 'soundfile'], if have installed paddleaudio >= 1.0.2
if 'soundfile' in backends:
    paddle.audio.backends.set_backend('soundfile')

paddle.audio.save(wav_path, waveform, sample_rate)