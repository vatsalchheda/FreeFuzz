import paddle

# example 1: x is a float
x_i = paddle.to_tensor([[1.0], [10.0]])
res = paddle.log10(x_i) # [[0.], [1.0]]

# example 2: x is float32
x_i = paddle.full(shape=[1], fill_value=10, dtype='float32')
paddle.to_tensor(x_i)
res = paddle.log10(x_i)
print(res) # [1.0]

# example 3: x is float64
x_i = paddle.full(shape=[1], fill_value=10, dtype='float64')
paddle.to_tensor(x_i)
res = paddle.log10(x_i)
print(res) # [1.0]