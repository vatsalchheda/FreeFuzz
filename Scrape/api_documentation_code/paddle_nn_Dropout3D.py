import paddle

x = paddle.arange(24, dtype="float32").reshape((1, 2, 2, 2, 3))
print(x)
# Tensor(shape=[1, 2, 2, 2, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [[[[[0. , 1. , 2. ],
#            [3. , 4. , 5. ]],
#           [[6. , 7. , 8. ],
#            [9. , 10., 11.]]],

#          [[[12., 13., 14.],
#            [15., 16., 17.]],
#           [[18., 19., 20.],
#            [21., 22., 23.]]]]])

m = paddle.nn.Dropout3D(p=0.5)
y_train = m(x)
print(y_train)
# Tensor(shape=[1, 2, 2, 2, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [[[[[0. , 2. , 4. ],
#            [6. , 8. , 10.]],
#           [[12., 14., 16.],
#            [18., 20., 22.]]],

#          [[[0. , 0. , 0. ],
#            [0. , 0. , 0. ]],
#           [[0. , 0. , 0. ],
#            [0. , 0. , 0. ]]]]])

m.eval()  # switch the model to test phase
y_test = m(x)
print(y_test)
# Tensor(shape=[1, 2, 2, 2, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [[[[[0. , 1. , 2. ],
#            [3. , 4. , 5. ]],
#           [[6. , 7. , 8. ],
#            [9. , 10., 11.]]],

#          [[[12., 13., 14.],
#            [15., 16., 17.]],
#           [[18., 19., 20.],
#            [21., 22., 23.]]]]])