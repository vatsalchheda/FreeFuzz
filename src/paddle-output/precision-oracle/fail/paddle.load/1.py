results = dict()
import paddle
import time
arg_1 = "C:\Users\phalt\AppData\Local\Temp\tmps5x4q_rr\prompt_dygraph\plm\model_state.pdparams"
arg_2 = False
start = time.time()
results["time_low"] = paddle.load(arg_1,return_numpy=arg_2,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.load(arg_1,return_numpy=arg_2,)
results["time_high"] = time.time() - start

print(results)
