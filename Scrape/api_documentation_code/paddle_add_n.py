import paddle

input0 = paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype='float32')
input1 = paddle.to_tensor([[7, 8, 9], [10, 11, 12]], dtype='float32')
output = paddle.add_n([input0, input1])
# [[8., 10., 12.],
#  [14., 16., 18.]]