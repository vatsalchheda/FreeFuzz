import paddle

input = paddle.to_tensor([[1, 2], [3, 4]], dtype='float32')
other = paddle.to_tensor([[2, 1], [2, 4]], dtype='float32')
label = paddle.to_tensor([[1, -1], [-1, -1]], dtype='float32')
loss = paddle.nn.functional.margin_ranking_loss(input, other, label)
print(loss) # [0.75]