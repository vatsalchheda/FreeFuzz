import paddle
from paddle.distribution import transformed_distribution

d = transformed_distribution.TransformedDistribution(
    paddle.distribution.Normal(0., 1.),
    [paddle.distribution.AffineTransform(paddle.to_tensor(1.), paddle.to_tensor(2.))]
)

print(d.sample([10]))
# Tensor(shape=[10], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [-0.10697651,  3.33609009, -0.86234951,  5.07457638,  0.75925219,
#         -4.17087793,  2.22579336, -0.93845034,  0.66054249,  1.50957513])
print(d.log_prob(paddle.to_tensor(0.5)))
# Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [-1.64333570])