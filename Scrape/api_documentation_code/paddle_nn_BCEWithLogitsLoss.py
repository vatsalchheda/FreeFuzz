import paddle
logit = paddle.to_tensor([5.0, 1.0, 3.0], dtype="float32")
label = paddle.to_tensor([1.0, 0.0, 1.0], dtype="float32")
bce_logit_loss = paddle.nn.BCEWithLogitsLoss()
output = bce_logit_loss(logit, label)
print(output.numpy())  # [0.45618808]