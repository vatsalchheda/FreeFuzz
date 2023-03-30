import paddle
input = paddle.to_tensor([1.1, 1.9])
label = paddle.to_tensor([1.0, 2.0])
output = paddle.fluid.layers.mse_loss(input, label)
print(output.numpy())
# [0.01]