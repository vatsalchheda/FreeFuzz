import paddle
from paddle.distribution import independent

beta = paddle.distribution.Beta(paddle.to_tensor([0.5, 0.5]), paddle.to_tensor([0.5, 0.5]))
print(beta.batch_shape, beta.event_shape)
# (2,) ()
print(beta.log_prob(paddle.to_tensor(0.2)))
# Tensor(shape=[2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [-0.22843921, -0.22843921])
reinterpreted_beta = independent.Independent(beta, 1)
print(reinterpreted_beta.batch_shape, reinterpreted_beta.event_shape)
# () (2,)
print(reinterpreted_beta.log_prob(paddle.to_tensor([0.2,  0.2])))
# Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [-0.45687842])