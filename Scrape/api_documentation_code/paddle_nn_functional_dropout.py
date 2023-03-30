import paddle

x = paddle.to_tensor([[1,2,3], [4,5,6]]).astype(paddle.float32)
y_train = paddle.nn.functional.dropout(x, 0.5)
y_test = paddle.nn.functional.dropout(x, 0.5, training=False)
y_0 = paddle.nn.functional.dropout(x, axis=0)
y_1 = paddle.nn.functional.dropout(x, axis=1)
y_01 = paddle.nn.functional.dropout(x, axis=[0,1])
print(x)
# Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
#        [[1., 2., 3.],
#         [4., 5., 6.]])
print(y_train)
# Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
#        [[2. , 0. , 6. ],
#         [8. , 0. , 12.]])
print(y_test)
# Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
#        [[1., 2., 3.],
#         [4., 5., 6.]])
print(y_0)
# Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
#        [[0. , 0. , 0. ],
#         [8. , 10., 12.]])
print(y_1)
# Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
#        [[2. , 0. , 6. ],
#         [8. , 0. , 12.]])
print(y_01)
# Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
#        [[0. , 0. , 0. ],
#         [8. , 0. , 12.]])