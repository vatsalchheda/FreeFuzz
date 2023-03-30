# required: gpu

import paddle

paddle.device.cuda.get_device_capability()

paddle.device.cuda.get_device_capability(0)

paddle.device.cuda.get_device_capability(paddle.CUDAPlace(0))