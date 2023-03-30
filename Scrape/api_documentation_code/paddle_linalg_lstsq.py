import paddle

paddle.set_device("cpu")
x = paddle.to_tensor([[1, 3], [3, 2], [5, 6.]])
y = paddle.to_tensor([[3, 4, 6], [5, 3, 4], [1, 2, 1.]])
results = paddle.linalg.lstsq(x, y, driver="gelsd")
print(results[0])
# [[ 0.78350395, -0.22165027, -0.62371236],
# [-0.11340097,  0.78866047,  1.14948535]]
print(results[1])
# [19.81443405, 10.43814468, 30.56185532])
print(results[2])
# 2
print(results[3])
# [9.03455734, 1.54167950]

x = paddle.to_tensor([[10, 2, 3], [3, 10, 5], [5, 6, 12.]])
y = paddle.to_tensor([[4, 2, 9], [2, 0, 3], [2, 5, 3.]])
results = paddle.linalg.lstsq(x, y, driver="gels")
print(results[0])
# [[ 0.39386186,  0.10230173,  0.93606132],
# [ 0.10741687, -0.29028133,  0.11892585],
# [-0.05115091,  0.51918161, -0.19948854]]
print(results[1])
# []