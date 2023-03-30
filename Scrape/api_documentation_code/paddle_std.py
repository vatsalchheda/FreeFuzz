import paddle

x = paddle.to_tensor([[1.0, 2.0, 3.0], [1.0, 4.0, 5.0]])
out1 = paddle.std(x)
# [1.63299316]
out2 = paddle.std(x, axis=1)
# [1.       2.081666]