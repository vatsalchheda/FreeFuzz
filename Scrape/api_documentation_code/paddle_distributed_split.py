# required: distributed
import paddle
import paddle.distributed.fleet as fleet

paddle.enable_static()
paddle.set_device('gpu:%d'%paddle.distributed.ParallelEnv().dev_id)
fleet.init(is_collective=True)
data = paddle.randint(0, 8, shape=[10,4])
emb_out = paddle.distributed.split(
    data,
    (8, 8),
    operation="embedding",
    num_partitions=2)