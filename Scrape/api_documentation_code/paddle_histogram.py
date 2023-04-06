import paddle

inputs = paddle.to_tensor([1, 2, 1])
result = paddle.histogram(inputs, bins=4, min=0, max=3)
print(result) # [0, 2, 1, 0]