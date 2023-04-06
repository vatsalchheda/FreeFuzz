import paddle
mse_loss = paddle.nn.loss.MSELoss()
input = paddle.to_tensor(1.5)
label = paddle.to_tensor(1.7)
output = mse_loss(input, label)
print(output)
# [0.04000002]