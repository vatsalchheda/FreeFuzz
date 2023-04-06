import paddle

# example 1, default offset value
data1 = paddle.tril_indices(4,4,0)
print(data1)
# [[0, 1, 1, 2, 2, 2, 3, 3, 3, 3],
#  [0, 0, 1, 0, 1, 2, 0, 1, 2, 3]]

# example 2, positive offset value
data2 = paddle.tril_indices(4,4,2)
print(data2)
# [[0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
#  [0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]]

# example 3, negative offset value
data3 = paddle.tril_indices(4,4,-1)
print(data3)
# [[ 1, 2, 2, 3, 3, 3],
#  [ 0, 0, 1, 0, 1, 2]]