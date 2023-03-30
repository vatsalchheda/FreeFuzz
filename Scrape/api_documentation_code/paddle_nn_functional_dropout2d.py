import paddle

x = paddle.randn(shape=(2, 3, 4, 5)).astype(paddle.float32)
y_train = paddle.nn.functional.dropout2d(x)  #train
y_test = paddle.nn.functional.dropout2d(x, training=False) #test
for i in range(2):
    for j in range(3):
        print(x[i,j,:,:])
        print(y_train[i,j,:,:]) # may all 0
        print(y_test[i,j,:,:])