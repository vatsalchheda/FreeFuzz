import paddle

x = paddle.to_tensor([[-1.5, 0., 2.], [3., 1.2, -2.4]])
out1 = paddle.logsumexp(x) # [3.4691226]
out2 = paddle.logsumexp(x, 1) # [2.15317821, 3.15684602]