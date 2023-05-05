results = dict()
import paddle
import time
arg_1_0 = False
arg_1 = [arg_1_0,]
arg_2_0 = 127.5
arg_2 = [arg_2_0,]
arg_class = paddle.vision.transforms.Normalize(arg_1,arg_2,)
int_tensor = paddle.randint(low=0, high=256, shape=[1, 28, 28], dtype='int32')
uint8_tensor = int_tensor.astype('uint8')
arg_3_0_tensor = uint8_tensor
arg_3_0 = arg_3_0_tensor.clone()
arg_3 = [arg_3_0,]
start = time.time()
results["time_low"] = arg_class(*arg_3)
results["time_low"] = time.time() - start
arg_1 = [arg_1_0,]
arg_2 = [arg_2_0,]
arg_3_0 = arg_3_0_tensor.clone().type(paddle.uint8)
arg_3 = [arg_3_0,]
start = time.time()
results["time_high"] = arg_class(*arg_3)
results["time_high"] = time.time() - start

print(results)
