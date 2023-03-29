import paddle

x = paddle.to_tensor([[3 + 4j, 7 - 24j, 0, 1 + 2j], [6 + 8j, 3, 0, -2]])
print(paddle.sgn(x))
#[[0.6+0.8j       0.28-0.96j      0.+0.j      0.4472136+0.8944272j]
# [0.6+0.8j       1.+0.j          0.+0.j      -1.+0.j]]