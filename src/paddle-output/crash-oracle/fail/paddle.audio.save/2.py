import paddle
arg_1 = "E:\UIUC\Spring 2023\CS 527\FreeFuzz\Scrape\test.wav"
arg_2_tensor = paddle.rand([1, 8000], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = 15987
res = paddle.audio.save(arg_1,arg_2,arg_3,)
