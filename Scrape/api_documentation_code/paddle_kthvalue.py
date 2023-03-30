import paddle

x = paddle.randn((2,3,2))
# Tensor(shape=[2, 3, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#       [[[ 0.22954939, -0.01296274],
#         [ 1.17135799, -0.34493217],
#         [-0.19550551, -0.17573971]],
#
#        [[ 0.15104349, -0.93965352],
#         [ 0.14745511,  0.98209465],
#         [ 0.10732264, -0.55859774]]])
y = paddle.kthvalue(x, 2, 1)
# (Tensor(shape=[2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
# [[ 0.22954939, -0.17573971],
#  [ 0.14745511, -0.55859774]]), Tensor(shape=[2, 2], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
#  [[0, 2],
#  [1, 2]]))