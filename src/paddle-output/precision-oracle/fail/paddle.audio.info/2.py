results = dict()
import paddle
import time
arg_1 = "E:\UIUC\Spring 2023\CS 527\FreeFuzz\Scrape\test.wav"
start = time.time()
results["time_low"] = paddle.audio.info(arg_1,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.audio.info(arg_1,)
results["time_high"] = time.time() - start

print(results)
