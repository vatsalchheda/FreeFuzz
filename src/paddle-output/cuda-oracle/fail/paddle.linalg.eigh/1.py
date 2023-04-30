results = dict()
import paddle
arg_1_tensor = paddle.randint(-128,256,[257], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = "L"
try:
  results["res_cpu"] = paddle.linalg.eigh(arg_1,UPLO=arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.linalg.eigh(arg_1,UPLO=arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
