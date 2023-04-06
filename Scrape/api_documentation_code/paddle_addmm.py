import paddle

x = paddle.ones([2,2])
y = paddle.ones([2,2])
input = paddle.ones([2,2])

out = paddle.addmm( input=input, x=x, y=y, beta=0.5, alpha=5.0 )

print(out)
# [[10.5 10.5]
# [10.5 10.5]]