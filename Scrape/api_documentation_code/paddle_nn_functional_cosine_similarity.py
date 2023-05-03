import paddle
import paddle.nn as nn

paddle.seed(1)
x1 = paddle.randn(shape=[2, 3])
x2 = paddle.randn(shape=[2, 3])

result = paddle.nn.functional.cosine_similarity(x1, x2, axis=0)
print(result)
# [0.97689527,  0.99996042, -0.55138415]
