import paddle

x = paddle.to_tensor([[-1, 2], [3, -4]], dtype='float32')
y = paddle.fluid.layers.leaky_relu(x, alpha=0.1)
print(y) # [[-0.1, 2], [3, -0.4]]