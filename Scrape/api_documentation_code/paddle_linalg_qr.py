import paddle

x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).astype('float64')
q, r = paddle.linalg.qr(x)
print (q)
print (r)

# Q = [[-0.16903085,  0.89708523],
#      [-0.50709255,  0.27602622],
#      [-0.84515425, -0.34503278]])

# R = [[-5.91607978, -7.43735744],
#      [ 0.        ,  0.82807867]])

# one can verify : X = Q * R ;