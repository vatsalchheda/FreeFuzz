import paddle

input = paddle.full(shape=[2, 3], fill_value=0.0, dtype='float32', name='input')
output = paddle.full_like(input, 2.0)
# [[2. 2. 2.]
#  [2. 2. 2.]]