import paddle

input = paddle.rand(shape=[4, 32, 32], dtype='float32')
res = paddle.is_empty(x=input)
print("res:", res)
# ('res:', Tensor: eager_tmp_1
#    - place: CPUPlace
#    - shape: [1]
#    - layout: NCHW
#    - dtype: bool
#    - data: [0])