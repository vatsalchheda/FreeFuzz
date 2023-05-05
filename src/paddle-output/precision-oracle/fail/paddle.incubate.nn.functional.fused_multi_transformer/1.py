results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([2, 4, 128], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0_tensor = paddle.rand([128], dtype=paddle.float32)
arg_2_0 = arg_2_0_tensor.clone()
arg_2 = [arg_2_0,]
arg_3_0_tensor = paddle.rand([128], dtype=paddle.float32)
arg_3_0 = arg_3_0_tensor.clone()
arg_3 = [arg_3_0,]
arg_4_0_tensor = paddle.rand([3, 2, 64, 128], dtype=paddle.float32)
arg_4_0 = arg_4_0_tensor.clone()
arg_4 = [arg_4_0,]
arg_5_0_tensor = paddle.rand([3, 2, 64], dtype=paddle.float32)
arg_5_0 = arg_5_0_tensor.clone()
arg_5 = [arg_5_0,]
arg_6_0_tensor = paddle.rand([128, 128], dtype=paddle.float32)
arg_6_0 = arg_6_0_tensor.clone()
arg_6 = [arg_6_0,]
arg_7_0_tensor = paddle.rand([128], dtype=paddle.float32)
arg_7_0 = arg_7_0_tensor.clone()
arg_7 = [arg_7_0,]
arg_8_0_tensor = paddle.rand([128], dtype=paddle.float32)
arg_8_0 = arg_8_0_tensor.clone()
arg_8 = [arg_8_0,]
arg_9_0_tensor = paddle.rand([128], dtype=paddle.float32)
arg_9_0 = arg_9_0_tensor.clone()
arg_9 = [arg_9_0,]
arg_10_0_tensor = paddle.rand([128, 512], dtype=paddle.float32)
arg_10_0 = arg_10_0_tensor.clone()
arg_10 = [arg_10_0,]
arg_11_0_tensor = paddle.rand([512], dtype=paddle.float32)
arg_11_0 = arg_11_0_tensor.clone()
arg_11 = [arg_11_0,]
arg_12_0_tensor = paddle.rand([512, 128], dtype=paddle.float32)
arg_12_0 = arg_12_0_tensor.clone()
arg_12 = [arg_12_0,]
arg_13_0_tensor = paddle.rand([128], dtype=paddle.float32)
arg_13_0 = arg_13_0_tensor.clone()
arg_13 = [arg_13_0,]
arg_14 = True
arg_15 = -3.999999
arg_16 = None
arg_17 = None
arg_18_tensor = paddle.rand([2, 1, 4, 4], dtype=paddle.float32)
arg_18 = arg_18_tensor.clone()
arg_19 = 0.0
arg_20 = "gelu"
arg_21 = -58.0
arg_22 = "upscale_in_train"
arg_23 = True
arg_24 = -26
arg_25 = None
start = time.time()
results["time_low"] = paddle.incubate.nn.functional.fused_multi_transformer(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,arg_8,arg_9,arg_10,arg_11,arg_12,arg_13,pre_layer_norm=arg_14,epsilon=arg_15,cache_kvs=arg_16,time_step=arg_17,attn_mask=arg_18,dropout_rate=arg_19,activation=arg_20,training=arg_21,mode=arg_22,trans_qkvw=arg_23,ring_id=arg_24,name=arg_25,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.float32)
arg_2_0 = arg_2_0_tensor.clone().astype(paddle.float32)
arg_2 = [arg_2_0,]
arg_3_0 = arg_3_0_tensor.clone().astype(paddle.float32)
arg_3 = [arg_3_0,]
arg_4_0 = arg_4_0_tensor.clone().astype(paddle.float32)
arg_4 = [arg_4_0,]
arg_5_0 = arg_5_0_tensor.clone().astype(paddle.float32)
arg_5 = [arg_5_0,]
arg_6_0 = arg_6_0_tensor.clone().astype(paddle.float32)
arg_6 = [arg_6_0,]
arg_7_0 = arg_7_0_tensor.clone().astype(paddle.float32)
arg_7 = [arg_7_0,]
arg_8_0 = arg_8_0_tensor.clone().astype(paddle.float32)
arg_8 = [arg_8_0,]
arg_9_0 = arg_9_0_tensor.clone().astype(paddle.float32)
arg_9 = [arg_9_0,]
arg_10_0 = arg_10_0_tensor.clone().astype(paddle.float32)
arg_10 = [arg_10_0,]
arg_11_0 = arg_11_0_tensor.clone().astype(paddle.float32)
arg_11 = [arg_11_0,]
arg_12_0 = arg_12_0_tensor.clone().astype(paddle.float32)
arg_12 = [arg_12_0,]
arg_13_0 = arg_13_0_tensor.clone().astype(paddle.float32)
arg_13 = [arg_13_0,]
arg_18 = arg_18_tensor.clone().astype(paddle.float32)
start = time.time()
results["time_high"] = paddle.incubate.nn.functional.fused_multi_transformer(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,arg_8,arg_9,arg_10,arg_11,arg_12,arg_13,pre_layer_norm=arg_14,epsilon=arg_15,cache_kvs=arg_16,time_step=arg_17,attn_mask=arg_18,dropout_rate=arg_19,activation=arg_20,training=arg_21,mode=arg_22,trans_qkvw=arg_23,ring_id=arg_24,name=arg_25,)
results["time_high"] = time.time() - start

print(results)
