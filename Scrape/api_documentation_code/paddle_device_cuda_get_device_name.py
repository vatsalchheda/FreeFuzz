# required: gpu

import paddle

paddle.device.cuda.get_device_name()

paddle.device.cuda.get_device_name(0)

paddle.device.cuda.get_device_name(paddle.CUDAPlace(0))