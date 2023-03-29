import paddle

paddle.set_device("cpu")  # and use cpu device

x = paddle.randn([2, 3], 'float32')
output = paddle.empty_like(x)
#[[1.8491974e+20 1.8037303e+28 1.7443726e+28]     # uninitialized
# [4.9640171e+28 3.0186127e+32 5.6715899e-11]]    # uninitialized