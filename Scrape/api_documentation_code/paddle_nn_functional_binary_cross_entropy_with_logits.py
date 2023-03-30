import paddle

logit = paddle.to_tensor([5.0, 1.0, 3.0])
label = paddle.to_tensor([1.0, 0.0, 1.0])
output = paddle.nn.functional.binary_cross_entropy_with_logits(logit, label)
print(output)  # [0.45618808]