import paddle

x = paddle.rand([2, 2, 1, 3], dtype="float32")
print(x)
# Tensor(shape=[2, 2, 1, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [[[[0.10052059, 0.93890846, 0.45351565]],
#          [[0.47507706, 0.45021373, 0.11331241]]],

#         [[[0.53358698, 0.97375143, 0.34997326]],
#          [[0.24758087, 0.52628899, 0.17970420]]]])

m = paddle.nn.Dropout2D(p=0.5)
y_train = m(x)
print(y_train)
# Tensor(shape=[2, 2, 1, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [[[[0.        , 0.        , 0.        ]],
#          [[0.95015413, 0.90042746, 0.22662482]]],

#         [[[1.06717396, 1.94750285, 0.69994652]],
#          [[0.        , 0.        , 0.        ]]]])

m.eval()  # switch the model to test phase
y_test = m(x)
print(y_test)
# Tensor(shape=[2, 2, 1, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [[[[0.10052059, 0.93890846, 0.45351565]],
#          [[0.47507706, 0.45021373, 0.11331241]]],

#         [[[0.53358698, 0.97375143, 0.34997326]],
#          [[0.24758087, 0.52628899, 0.17970420]]]])