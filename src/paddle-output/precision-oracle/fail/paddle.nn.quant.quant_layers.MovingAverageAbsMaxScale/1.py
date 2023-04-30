results = dict()
import paddle
import time
arg_1 = "fc_1.tmp_2"
arg_2 = "int32"
arg_class = paddle.nn.quant.quant_layers.MovingAverageAbsMaxScale(name=arg_1,dtype=arg_2,)
arg_3_0_tensor = paddle.randint(-32768,32768,[-1, 10], dtype=paddle.float16)
arg_3_0 = arg_3_0_tensor.clone()
arg_3 = [arg_3_0,]
start = time.time()
results["time_low"] = arg_class(*arg_3)
results["time_low"] = time.time() - start
arg_3_0 = arg_3_0_tensor.clone().type(paddle.float32)
arg_3 = [arg_3_0,]
start = time.time()
results["time_high"] = arg_class(*arg_3)
results["time_high"] = time.time() - start

print(results)
