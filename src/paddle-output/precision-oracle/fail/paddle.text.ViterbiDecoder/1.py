results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([2, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = "max"
arg_class = paddle.text.ViterbiDecoder(arg_1,include_bos_eos_tag=arg_2,)
arg_3_0_tensor = paddle.rand([2, 4, 3], dtype=paddle.float32)
arg_3_0 = arg_3_0_tensor.clone()
int_tensor = paddle.randint(low=-128, high=127, shape=[2], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_3_1_tensor = int8_tensor
arg_3_1 = arg_3_1_tensor.clone()
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_low"] = arg_class(*arg_3)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.float32)
arg_3_0 = arg_3_0_tensor.clone().astype(paddle.float32)
arg_3_1 = arg_3_1_tensor.clone().astype(paddle.int64)
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_high"] = arg_class(*arg_3)
results["time_high"] = time.time() - start

print(results)
