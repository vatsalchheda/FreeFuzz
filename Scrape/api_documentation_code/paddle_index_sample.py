import paddle

x = paddle.to_tensor([[1.0, 2.0, 3.0, 4.0],
                      [5.0, 6.0, 7.0, 8.0],
                      [9.0, 10.0, 11.0, 12.0]], dtype='float32')
index = paddle.to_tensor([[0, 1, 2],
                          [1, 2, 3],
                          [0, 0, 0]], dtype='int32')
target = paddle.to_tensor([[100, 200, 300, 400],
                           [500, 600, 700, 800],
                           [900, 1000, 1100, 1200]], dtype='int32')
out_z1 = paddle.index_sample(x, index)
print(out_z1)
#[[1. 2. 3.]
# [6. 7. 8.]
# [9. 9. 9.]]

# Use the index of the maximum value by topk op
# get the value of the element of the corresponding index in other tensors
top_value, top_index = paddle.topk(x, k=2)
out_z2 = paddle.index_sample(target, top_index)
print(top_value)
#[[ 4.  3.]
# [ 8.  7.]
# [12. 11.]]

print(top_index)
#[[3 2]
# [3 2]
# [3 2]]

print(out_z2)
#[[ 400  300]
# [ 800  700]
# [1200 1100]]