import paddle
import paddle.nn.functional as F

input = paddle.to_tensor([[1,2,3],[4,5,6],[4,4,4],[1,1,1]], dtype='int64')
label = paddle.to_tensor([[1,3,4,1],[4,5,8,1],[7,7,7,1],[1,1,1,1]], dtype='int64')
input_len = paddle.to_tensor([3,3,3,3], dtype='int64')
label_len = paddle.to_tensor([4,4,4,4], dtype='int64')

distance, sequence_num = F.loss.edit_distance(input=input, label=label, input_length=input_len, label_length=label_len, normalized=False)

# print(distance)
# [[3.]
#  [2.]
#  [4.]
#  [1.]]
# if set normalized to True
# [[0.75]
#  [0.5 ]
#  [1.  ]
#  [0.25]
#
# print(sequence_num)
# [4]