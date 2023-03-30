import paddle

x =  paddle.to_tensor([[5,8,9,5],
                         [0,0,1,7],
                         [6,9,2,4]])
out1 = paddle.argmin(x)
print(out1) # 4
out2 = paddle.argmin(x, axis=0)
print(out2)
# [1, 1, 1, 2]
out3 = paddle.argmin(x, axis=-1)
print(out3)
# [0, 0, 2]
out4 = paddle.argmin(x, axis=0, keepdim=True)
print(out4)
# [[1, 1, 1, 2]]