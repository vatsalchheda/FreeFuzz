import paddle

paddle.set_device("cpu")
paddle.seed(1234)

x = paddle.rand(shape=[3, 3], dtype='float64')
# [[0.02773777, 0.93004224, 0.06911496],
#  [0.24831591, 0.45733623, 0.07717843],
#  [0.48016702, 0.14235102, 0.42620817]])

print(paddle.linalg.eigvals(x))
# [(-0.27078833542132674+0j), (0.29962280156230725+0j), (0.8824477020120244+0j)] #complex128