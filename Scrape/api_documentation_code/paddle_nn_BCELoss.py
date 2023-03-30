import paddle

input = paddle.to_tensor([0.5, 0.6, 0.7])
label = paddle.to_tensor([1.0, 0.0, 1.0])
bce_loss = paddle.nn.BCELoss()
output = bce_loss(input, label)
print(output)
# Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [0.65537101])