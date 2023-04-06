# required: gpu
import paddle

paddle.device.cuda.synchronize()
paddle.device.cuda.synchronize(0)
paddle.device.cuda.synchronize(paddle.CUDAPlace(0))