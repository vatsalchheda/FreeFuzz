results = dict()
import paddle
import time
arg_1 = "__main__Logic"
arg_2 = 10.0
int_tensor = paddle.randint(low=-128, high=128, shape=[1], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_3_0_tensor = int8_tensor
arg_3_0 = arg_3_0_tensor.clone()
arg_3 = [arg_3_0,]
int_tensor = paddle.randint(low=-128, high=128, shape=[1], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_4_0_tensor = int8_tensor
arg_4_0 = arg_4_0_tensor.clone()
arg_4 = [arg_4_0,]
start = time.time()
results["time_low"] = paddle.onnx.export(arg_1,arg_2,input_spec=arg_3,output_spec=arg_4,)
results["time_low"] = time.time() - start
arg_3_0 = arg_3_0_tensor.clone().type(paddle.int64)
arg_3 = [arg_3_0,]
arg_4_0 = arg_4_0_tensor.clone().type(paddle.int64)
arg_4 = [arg_4_0,]
start = time.time()
results["time_high"] = paddle.onnx.export(arg_1,arg_2,input_spec=arg_3,output_spec=arg_4,)
results["time_high"] = time.time() - start

print(results)
