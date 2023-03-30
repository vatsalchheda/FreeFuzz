import paddle

img1 = paddle.to_tensor([[1, 2], [3, 4]], dtype=paddle.float32)
img2 = paddle.to_tensor([[5, 6], [7, 8]], dtype=paddle.float32)
inputs = [img1, img2]
index = paddle.to_tensor([[1], [0]], dtype=paddle.int32)
res = paddle.multiplex(inputs, index)
print(res) # Tensor([[5., 6.], [3., 4.]], dtype=float32)