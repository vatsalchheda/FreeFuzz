import paddle

x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
out = paddle.stanh(x, scale_a=0.67, scale_b=1.72) # [1.00616539, 1.49927628, 1.65933108, 1.70390463]