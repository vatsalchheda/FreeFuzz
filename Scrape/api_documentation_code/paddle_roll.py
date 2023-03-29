import paddle

x = paddle.to_tensor([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [7.0, 8.0, 9.0]])
out_z1 = paddle.roll(x, shifts=1)
print(out_z1)
#[[9. 1. 2.]
# [3. 4. 5.]
# [6. 7. 8.]]
out_z2 = paddle.roll(x, shifts=1, axis=0)
print(out_z2)
#[[7. 8. 9.]
# [1. 2. 3.]
# [4. 5. 6.]]
out_z3 = paddle.roll(x, shifts=1, axis=1)
print(out_z3)
#[[3. 1. 2.]
# [6. 4. 5.]
# [9. 7. 8.]]