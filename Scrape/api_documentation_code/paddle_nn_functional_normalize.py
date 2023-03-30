import paddle
import paddle.nn.functional as F

paddle.disable_static()
x = paddle.arange(6, dtype="float32").reshape([2,3])
y = F.normalize(x)
print(y)
# Tensor(shape=[2, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [[0.        , 0.44721359, 0.89442718],
#         [0.42426404, 0.56568539, 0.70710671]])

y = F.normalize(x, p=1.5)
print(y)
# Tensor(shape=[2, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [[0.        , 0.40862012, 0.81724024],
#         [0.35684016, 0.47578689, 0.59473360]])

y = F.normalize(x, axis=0)
print(y)
# Tensor(shape=[2, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [[0.        , 0.24253564, 0.37139067],
#         [1.        , 0.97014254, 0.92847669]])