# required: gpu
import paddle

input_tensor = paddle.to_tensor(paddle.ones((3, 3)), dtype="float32")
index = paddle.to_tensor([0, 2], dtype="int32")
value = paddle.to_tensor([[1, 1, 1], [1, 1, 1]], dtype="float32")
outplace_res = paddle.index_add(input_tensor, index, 0, value)
print(outplace_res)
# Tensor(shape=[3, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [[2., 2., 2.],
#         [1., 1., 1.],
#         [2., 2., 2.]])