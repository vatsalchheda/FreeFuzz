import paddle
label = paddle.to_tensor([[16], [1]], "int64")
shard_label = paddle.shard_index(input=label,
                                 index_num=20,
                                 nshards=2,
                                 shard_id=0)
print(shard_label)
# [[-1], [1]]