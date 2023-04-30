results = dict()
import paddle
arg_1_tensor = paddle.randint(-32,256,[8, 257, 376], dtype=paddle.complex128)
arg_1 = arg_1_tensor.clone()
arg_2 = 1
try:
  results["res_cpu"] = paddle.signal.istft(arg_1,n_fft=arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.signal.istft(arg_1,n_fft=arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
