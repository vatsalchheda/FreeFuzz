import paddle

x = paddle.full(shape=[4, 5, 7], fill_value=0, dtype='int32')
numel = paddle.numel(x) # 140