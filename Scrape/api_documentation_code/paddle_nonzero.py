import paddle

x1 = paddle.to_tensor([[1.0, 0.0, 0.0],
                       [0.0, 2.0, 0.0],
                       [0.0, 0.0, 3.0]])
x2 = paddle.to_tensor([0.0, 1.0, 0.0, 3.0])
out_z1 = paddle.nonzero(x1)
print(out_z1)
#[[0 0]
# [1 1]
# [2 2]]
out_z1_tuple = paddle.nonzero(x1, as_tuple=True)
for out in out_z1_tuple:
    print(out)
#[[0]
# [1]
# [2]]
#[[0]
# [1]
# [2]]
out_z2 = paddle.nonzero(x2)
print(out_z2)
#[[1]
# [3]]
out_z2_tuple = paddle.nonzero(x2, as_tuple=True)
for out in out_z2_tuple:
    print(out)
#[[1]
# [3]]