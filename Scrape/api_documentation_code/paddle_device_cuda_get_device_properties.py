# required: gpu

import paddle
paddle.device.cuda.get_device_properties()
# _gpuDeviceProperties(name='A100-SXM4-40GB', major=8, minor=0, total_memory=40536MB, multi_processor_count=108)

paddle.device.cuda.get_device_properties(0)
# _gpuDeviceProperties(name='A100-SXM4-40GB', major=8, minor=0, total_memory=40536MB, multi_processor_count=108)

paddle.device.cuda.get_device_properties('gpu:0')
# _gpuDeviceProperties(name='A100-SXM4-40GB', major=8, minor=0, total_memory=40536MB, multi_processor_count=108)

paddle.device.cuda.get_device_properties(paddle.CUDAPlace(0))
# _gpuDeviceProperties(name='A100-SXM4-40GB', major=8, minor=0, total_memory=40536MB, multi_processor_count=108)