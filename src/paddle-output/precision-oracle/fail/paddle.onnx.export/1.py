results = dict()
import paddle
import time
arg_1 = "__main__LinearNet"
arg_2 = "linear_net"
float_tensor = paddle.rand([-1, 128], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_3_0_tensor = f16_tensor
arg_3_0 = arg_3_0_tensor.clone()
arg_3 = [arg_3_0,]
start = time.time()
results["time_low"] = paddle.onnx.export(arg_1,arg_2,input_spec=arg_3,)
results["time_low"] = time.time() - start
arg_3_0 = arg_3_0_tensor.clone().type(paddle.float32)
arg_3 = [arg_3_0,]
start = time.time()
results["time_high"] = paddle.onnx.export(arg_1,arg_2,input_spec=arg_3,)
results["time_high"] = time.time() - start

print(results)
