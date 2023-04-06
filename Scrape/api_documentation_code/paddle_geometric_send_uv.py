import paddle

x = paddle.to_tensor([[0, 2, 3], [1, 4, 5], [2, 6, 7]], dtype="float32")
y = paddle.to_tensor([[0, 1, 2], [2, 3, 4], [4, 5, 6]], dtype="float32")
indexes = paddle.to_tensor([[0, 1], [1, 2], [2, 1], [0, 0]], dtype="int32")
src_index = indexes[:, 0]
dst_index = indexes[:, 1]
out = paddle.geometric.send_uv(x, y, src_index, dst_index, message_op="add")
# Outputs: [[2., 5., 7.], [5., 9., 11.], [4., 9., 11.], [0., 3., 5.]]