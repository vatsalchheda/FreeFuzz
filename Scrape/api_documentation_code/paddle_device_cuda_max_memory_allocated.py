# required: gpu
import paddle

max_memory_allocated_size = paddle.device.cuda.max_memory_allocated(paddle.CUDAPlace(0))
max_memory_allocated_size = paddle.device.cuda.max_memory_allocated(0)
max_memory_allocated_size = paddle.device.cuda.max_memory_allocated("gpu:0")