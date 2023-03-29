import paddle

# required: gpu
paddle.set_device("gpu")
tensor = paddle.randn([512, 512, 512], "float")
del tensor
paddle.device.cuda.empty_cache()