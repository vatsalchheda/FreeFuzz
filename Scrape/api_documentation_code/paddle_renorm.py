import paddle
input = [[[2.0,2,-2],[3,0.3,3]],[[2,-8,2],[3.1,3.7,3]]]
x = paddle.to_tensor(input,dtype='float32')
y = paddle.renorm(x, 1.0, 2, 2.05)
print(y)