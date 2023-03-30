import paddle

x = paddle.to_tensor([[5,8,9,5],
                     [0,0,1,7],
                     [6,9,2,4]])
out1 = paddle.argmax(x)
print(out1) # 2
out2 = paddle.argmax(x, axis=0)
print(out2)
# [2, 2, 0, 1]
out3 = paddle.argmax(x, axis=-1)
print(out3)
# [2, 3, 1]
out4 = paddle.argmax(x, axis=0, keepdim=True)
print(out4)
# [[2, 2, 0, 1]]