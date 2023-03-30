import paddle

data = paddle.to_tensor([[0], [1]], dtype='float32')
res = paddle.log1p(data)
# [[0.], [0.6931472]]