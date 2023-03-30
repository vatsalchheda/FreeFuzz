import paddle

input1 = paddle.rand(shape=[2, 3, 5], dtype='float32')
check = paddle.is_tensor(input1)
print(check)  #True

input3 = [1, 4]
check = paddle.is_tensor(input3)
print(check)  #False