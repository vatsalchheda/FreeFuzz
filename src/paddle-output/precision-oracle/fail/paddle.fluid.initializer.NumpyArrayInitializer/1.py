results = dict()
import paddle
import time
arg_1 = "numpyndarray"
arg_class = paddle.fluid.initializer.NumpyArrayInitializer(arg_1,)
arg_2_0_tensor = paddle.randint(-256,32,[128, 100], dtype=paddle.float16)
arg_2_0 = arg_2_0_tensor.clone()
arg_2_1_tensor = paddle.randint(-32768,256,[2, 2], dtype=paddle.float16)
arg_2_1 = arg_2_1_tensor.clone()
arg_2 = [arg_2_0,arg_2_1,]
start = time.time()
results["time_low"] = arg_class(*arg_2)
results["time_low"] = time.time() - start
arg_2_0 = arg_2_0_tensor.clone().type(paddle.float32)
arg_2_1 = arg_2_1_tensor.clone().type(paddle.float32)
arg_2 = [arg_2_0,arg_2_1,]
start = time.time()
results["time_high"] = arg_class(*arg_2)
results["time_high"] = time.time() - start

print(results)
