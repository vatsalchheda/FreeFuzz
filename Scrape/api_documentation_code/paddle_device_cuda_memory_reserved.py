# required: gpu
import paddle

memory_reserved_size = paddle.device.cuda.memory_reserved(paddle.CUDAPlace(0))
memory_reserved_size = paddle.device.cuda.memory_reserved(0)
memory_reserved_size = paddle.device.cuda.memory_reserved("gpu:0")