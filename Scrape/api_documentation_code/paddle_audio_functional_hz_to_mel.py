import paddle

val = 3.0
htk_flag = True
mel_paddle_tensor = paddle.audio.functional.hz_to_mel(
    paddle.to_tensor(val), htk_flag)