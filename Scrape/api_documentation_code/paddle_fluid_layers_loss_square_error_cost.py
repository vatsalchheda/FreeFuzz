import paddle
input = paddle.to_tensor([1.1, 1.9])
label = paddle.to_tensor([1.0, 2.0])
output = paddle.nn.functional.square_error_cost(input, label)
print(output)
# [0.01, 0.01]