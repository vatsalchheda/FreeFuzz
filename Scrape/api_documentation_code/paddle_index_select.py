import paddle

x = paddle.to_tensor([[1.0, 2.0, 3.0, 4.0],
                      [5.0, 6.0, 7.0, 8.0],
                      [9.0, 10.0, 11.0, 12.0]])
index = paddle.to_tensor([0, 1, 1], dtype='int32')
out_z1 = paddle.index_select(x=x, index=index)
#[[1. 2. 3. 4.]
# [5. 6. 7. 8.]
# [5. 6. 7. 8.]]
out_z2 = paddle.index_select(x=x, index=index, axis=1)
#[[ 1.  2.  2.]
# [ 5.  6.  6.]
# [ 9. 10. 10.]]