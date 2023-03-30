import paddle

x =  paddle.randn([3,3,3])

A = paddle.linalg.det(x)

print(A)

# [ 0.02547996,  2.52317095, -6.15900707])