import paddle

tensor = paddle.to_tensor([[[1,2,2],[2,3,3]],[[0,5,5],[9,9,0]]], dtype=paddle.float32)
res = paddle.mode(tensor, 2)
print(res)
# (Tensor(shape=[2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#   [[2., 3.],
#    [5., 9.]]), Tensor(shape=[2, 2], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
#   [[1, 1],
#    [1, 0]]))