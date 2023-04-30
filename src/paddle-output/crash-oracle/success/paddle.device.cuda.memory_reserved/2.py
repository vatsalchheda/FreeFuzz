import paddle
arg_1 = "gpu:0"
res = paddle.device.cuda.memory_reserved(arg_1,)
