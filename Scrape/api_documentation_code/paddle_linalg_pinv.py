import paddle

x = paddle.arange(15).reshape((3, 5)).astype('float64')
input = paddle.to_tensor(x)
out = paddle.linalg.pinv(input)
print(input)
print(out)

# input:
# [[0. , 1. , 2. , 3. , 4. ],
# [5. , 6. , 7. , 8. , 9. ],
# [10., 11., 12., 13., 14.]]

# out:
# [[-0.22666667, -0.06666667,  0.09333333],
# [-0.12333333, -0.03333333,  0.05666667],
# [-0.02000000,  0.00000000,  0.02000000],
# [ 0.08333333,  0.03333333, -0.01666667],
# [ 0.18666667,  0.06666667, -0.05333333]]

# one can verify : x * out * x = x ;
# or              out * x * out = x ;