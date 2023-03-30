import paddle

x = paddle.to_tensor([2, 3, 4], 'float64')
y = paddle.cast(x, 'uint8')