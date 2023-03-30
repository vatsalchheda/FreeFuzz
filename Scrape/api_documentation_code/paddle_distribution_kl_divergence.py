import paddle

p = paddle.distribution.Beta(alpha=0.5, beta=0.5)
q = paddle.distribution.Beta(alpha=0.3, beta=0.7)

print(paddle.distribution.kl_divergence(p, q))
# Tensor(shape=[1], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#        [0.21193528])