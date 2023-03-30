import paddle

val = 3.0
htk_flag = True
mel_paddle_tensor = paddle.audio.functional.mel_to_hz(
    paddle.to_tensor(val), htk_flag)