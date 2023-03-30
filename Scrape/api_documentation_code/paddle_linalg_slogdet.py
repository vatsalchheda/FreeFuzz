import paddle

x =  paddle.randn([3,3,3])

A = paddle.linalg.slogdet(x)

print(A)

# [[ 1.        ,  1.        , -1.        ],
# [-0.98610914, -0.43010661, -0.10872950]])