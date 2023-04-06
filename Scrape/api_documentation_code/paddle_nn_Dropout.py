import paddle

x = paddle.to_tensor([[1,2,3], [4,5,6]], dtype="float32")
m = paddle.nn.Dropout(p=0.5)

y_train = m(x)
print(y_train)
# Tensor(shape=[2, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [[2., 0., 6.],
#         [0., 0., 0.]])

m.eval()  # switch the model to test phase
y_test = m(x)
print(y_test)
# Tensor(shape=[2, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [[1., 2., 3.],
#         [4., 5., 6.]])