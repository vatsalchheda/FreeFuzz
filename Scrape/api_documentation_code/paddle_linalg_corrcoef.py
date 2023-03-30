import paddle

xt = paddle.rand((3,4))
print(paddle.linalg.corrcoef(xt))

# Tensor(shape=[3, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
# [[ 1.        , -0.73702252,  0.66228950],
# [-0.73702258,  1.        , -0.77104872],
# [ 0.66228974, -0.77104825,  1.        ]])