import paddle
gid = paddle.distributed.new_group([2,4,6])
paddle.distributed.get_group(gid.id)