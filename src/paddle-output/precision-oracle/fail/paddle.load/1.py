results = dict()
import paddle
import time
arg_1 = "C:\Users\phalt\.paddlespeech\models\panns_cnn14-32k\1.0\panns_cnn14\cnn14.pdparams"
start = time.time()
results["time_low"] = paddle.load(arg_1,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.load(arg_1,)
results["time_high"] = time.time() - start

print(results)
