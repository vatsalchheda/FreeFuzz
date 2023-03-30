import paddle

x = paddle.to_tensor([[1.0, 2.0], [1.0, 3.0], [4.0, 6.0]]).astype('float64')
x = x.reshape([3, 2])
u, s, vh = paddle.linalg.svd(x)
print (u)
#U = [[ 0.27364809, -0.21695147  ],
#      [ 0.37892198, -0.87112408 ],
#      [ 0.8840446 ,  0.44053933 ]]

print (s)
#S = [8.14753743, 0.78589688]
print (vh)
#VT= [[ 0.51411221,  0.85772294],
#     [ 0.85772294, -0.51411221]]

# one can verify : U * S * VT == X
#                  U * UH == I
#                  V * VH == I