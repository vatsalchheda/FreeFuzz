import paddle
arg_1 = 24377
arg_2 = -14
arg_class = paddle.audio.features.MFCC(sr=arg_1,n_mfcc=arg_2,)
arg_3_0_tensor = paddle.rand([1, 41402], dtype=paddle.float32)
arg_3_0 = arg_3_0_tensor.clone()
arg_3 = [arg_3_0,]
res = arg_class(*arg_3)
