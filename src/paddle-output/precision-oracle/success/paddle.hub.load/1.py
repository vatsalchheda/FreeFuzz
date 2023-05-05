results = dict()
import paddle
import time
arg_1 = "lyuwenyu/paddlehub_demo:main"
arg_2 = "MM"
arg_3 = "github"
start = time.time()
results["time_low"] = paddle.hub.load(arg_1,model=arg_2,source=arg_3,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.hub.load(arg_1,model=arg_2,source=arg_3,)
results["time_high"] = time.time() - start

print(results)
