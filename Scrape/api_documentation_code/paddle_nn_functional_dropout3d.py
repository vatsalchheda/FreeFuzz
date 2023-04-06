import paddle

x = paddle.randn(shape=(2, 3, 4, 5, 6)).astype(paddle.float32)
y_train = paddle.nn.functional.dropout3d(x)  #train
y_test = paddle.nn.functional.dropout3d(x, training=False) #test
print(x[0,0,:,:,:])
print(y_train[0,0,:,:,:]) # may all 0
print(y_test[0,0,:,:,:])