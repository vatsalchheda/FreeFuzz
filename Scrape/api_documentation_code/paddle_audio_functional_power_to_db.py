import paddle

val = 3.0
decibel_paddle = paddle.audio.functional.power_to_db(
    paddle.to_tensor(val))