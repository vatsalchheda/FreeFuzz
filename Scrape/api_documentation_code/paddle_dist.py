import paddle

x = paddle.to_tensor([[3, 3],[3, 3]], dtype="float32")
y = paddle.to_tensor([[3, 3],[3, 1]], dtype="float32")
out = paddle.dist(x, y, 0)
print(out) # out = [1.]

out = paddle.dist(x, y, 2)
print(out) # out = [2.]

out = paddle.dist(x, y, float("inf"))
print(out) # out = [2.]

out = paddle.dist(x, y, float("-inf"))
print(out) # out = [0.]