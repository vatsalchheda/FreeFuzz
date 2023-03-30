import paddle

x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
out = paddle.asin(x)
print(out)
# [-0.41151685 -0.20135792  0.10016742  0.30469265]