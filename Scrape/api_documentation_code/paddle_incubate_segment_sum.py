import paddle
data = paddle.to_tensor([[1, 2, 3], [3, 2, 1], [4, 5, 6]], dtype='float32')
segment_ids = paddle.to_tensor([0, 0, 1], dtype='int32')
out = paddle.incubate.segment_sum(data, segment_ids)
#Outputs: [[4., 4., 4.], [4., 5., 6.]]