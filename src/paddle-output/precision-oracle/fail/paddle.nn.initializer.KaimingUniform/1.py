results = dict()
import paddle
import time
arg_1 = None
arg_2 = 31.23606797749979
arg_3 = "leaky_relu"
arg_class = paddle.nn.initializer.KaimingUniform(fan_in=arg_1,negative_slope=arg_2,nonlinearity=arg_3,)
arg_4_0_tensor = paddle.rand([512, 1], dtype=paddle.float32)
arg_4_0 = arg_4_0_tensor.clone()
int_tensor = paddle.randint(low=0, high=255, shape=[47, 45], dtype='int32')
uint8_tensor = int_tensor.astype('uint8')
arg_4_1_tensor = uint8_tensor
arg_4_1 = arg_4_1_tensor.clone()
arg_4 = [arg_4_0,arg_4_1,]
start = time.time()
results["time_low"] = arg_class(*arg_4)
results["time_low"] = time.time() - start
arg_4_0 = arg_4_0_tensor.clone().astype(paddle.float32)
arg_4_1 = arg_4_1_tensor.clone().astype(paddle.uint8)
arg_4 = [arg_4_0,arg_4_1,]
start = time.time()
results["time_high"] = arg_class(*arg_4)
results["time_high"] = time.time() - start

print(results)
