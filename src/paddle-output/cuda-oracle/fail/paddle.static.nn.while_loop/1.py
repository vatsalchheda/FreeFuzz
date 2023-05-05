results = dict()
import paddle
arg_1 = "cond"
arg_2 = "body"
int_tensor = paddle.randint(low=0, high=255, shape=[], dtype='int32')
uint8_tensor = int_tensor.astype('uint8')
arg_3_0_tensor = uint8_tensor
arg_3_0 = arg_3_0_tensor.clone()
real = paddle.rand([0], paddle.float32)
imag = paddle.rand([0], paddle.float32)
arg_3_1_tensor = paddle.complex(real, imag)
arg_3_1 = arg_3_1_tensor.clone()
arg_3_2_tensor = paddle.rand([1], dtype=paddle.float32)
arg_3_2 = arg_3_2_tensor.clone()
arg_3_3_tensor = paddle.rand([13], dtype=paddle.float32)
arg_3_3 = arg_3_3_tensor.clone()
int_tensor = paddle.randint(low=-32768, high=32767, shape=[1], dtype='int32')
int16_tensor = int_tensor.astype('int16')
arg_3_4_tensor = int16_tensor
arg_3_4 = arg_3_4_tensor.clone()
real = paddle.rand([], paddle.float64)
imag = paddle.rand([], paddle.float64)
arg_3_5_tensor = paddle.complex(real, imag)
arg_3_5 = arg_3_5_tensor.clone()
arg_3_6_tensor = paddle.randint(0,2,[16])
arg_3_6 = arg_3_6_tensor.clone()
arg_3 = [arg_3_0,arg_3_1,arg_3_2,arg_3_3,arg_3_4,arg_3_5,arg_3_6,]
try:
  results["res_cpu"] = paddle.static.nn.while_loop(cond=arg_1,body=arg_2,loop_vars=arg_3,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_3_0 = arg_3_0_tensor.clone().cuda()
arg_3_1 = arg_3_1_tensor.clone().cuda()
arg_3_2 = arg_3_2_tensor.clone().cuda()
arg_3_3 = arg_3_3_tensor.clone().cuda()
arg_3_4 = arg_3_4_tensor.clone().cuda()
arg_3_5 = arg_3_5_tensor.clone().cuda()
arg_3_6 = arg_3_6_tensor.clone().cuda()
arg_3 = [arg_3_0,arg_3_1,arg_3_2,arg_3_3,arg_3_4,arg_3_5,arg_3_6,]
try:
  results["res_gpu"] = paddle.static.nn.while_loop(cond=arg_1,body=arg_2,loop_vars=arg_3,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
