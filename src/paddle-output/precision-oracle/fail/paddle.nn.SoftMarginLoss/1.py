results = dict()
import paddle
import time
arg_1 = "none"
arg_class = paddle.nn.SoftMarginLoss(reduction=arg_1,)
float_tensor = paddle.rand([5, 5], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_2_0_tensor = f16_tensor
arg_2_0 = arg_2_0_tensor.clone()
int_tensor = paddle.randint(low=-128, high=127, shape=[5, 5], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_2_1_tensor = int8_tensor
arg_2_1 = arg_2_1_tensor.clone()
arg_2 = [arg_2_0,arg_2_1,]
start = time.time()
results["time_low"] = arg_class(*arg_2)
results["time_low"] = time.time() - start
arg_2_0 = arg_2_0_tensor.clone().type(paddle.float64)
arg_2_1 = arg_2_1_tensor.clone().type(paddle.int64)
arg_2 = [arg_2_0,arg_2_1,]
start = time.time()
results["time_high"] = arg_class(*arg_2)
results["time_high"] = time.time() - start

print(results)
