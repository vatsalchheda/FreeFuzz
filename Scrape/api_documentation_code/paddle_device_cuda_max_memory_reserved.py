# required: gpu
import paddle

max_memory_reserved_size = paddle.device.cuda.max_memory_reserved(paddle.CUDAPlace(0))
max_memory_reserved_size = paddle.device.cuda.max_memory_reserved(0)
max_memory_reserved_size = paddle.device.cuda.max_memory_reserved("gpu:0")