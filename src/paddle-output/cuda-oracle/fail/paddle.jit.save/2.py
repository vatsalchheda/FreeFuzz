results = dict()
import paddle
arg_1 = "__main__Mnist"
arg_2 = "inference_model"
arg_3_0_tensor = paddle.rand([-1, 784], dtype=paddle.float32)
arg_3_0 = arg_3_0_tensor.clone()
arg_3 = [arg_3_0,]
try:
  results["res_cpu"] = paddle.jit.save(arg_1,arg_2,input_spec=arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_3_0 = arg_3_0_tensor.clone().cuda()
arg_3 = [arg_3_0,]
try:
  results["res_gpu"] = paddle.jit.save(arg_1,arg_2,input_spec=arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
