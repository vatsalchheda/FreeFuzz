# required: gpu
import paddle

memory_allocated_size = paddle.device.cuda.memory_allocated(paddle.CUDAPlace(0))
memory_allocated_size = paddle.device.cuda.memory_allocated(0)
memory_allocated_size = paddle.device.cuda.memory_allocated("gpu:0")