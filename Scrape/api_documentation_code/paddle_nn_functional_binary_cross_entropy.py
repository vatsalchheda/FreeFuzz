import paddle

input = paddle.to_tensor([0.5, 0.6, 0.7], 'float32')
label = paddle.to_tensor([1.0, 0.0, 1.0], 'float32')
output = paddle.nn.functional.binary_cross_entropy(input, label)
print(output)  # [0.65537095]