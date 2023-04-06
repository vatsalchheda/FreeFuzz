import paddle

input = paddle.to_tensor([[1, 2], [3, 4]], dtype="float32")
other = paddle.to_tensor([[2, 1], [2, 4]], dtype="float32")
label = paddle.to_tensor([[1, -1], [-1, -1]], dtype="float32")
margin_rank_loss = paddle.nn.MarginRankingLoss()
loss = margin_rank_loss(input, other, label)

print(loss)
# [0.75]