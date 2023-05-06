results = dict()
import paddle
arg_1 = "test_export_protobuf.pb"
try:
  results["res_cpu"] = paddle.profiler.load_profiler_result(arg_1,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.profiler.load_profiler_result(arg_1,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
