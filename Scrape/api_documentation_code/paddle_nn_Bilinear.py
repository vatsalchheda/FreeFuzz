import paddle

layer1 = paddle.rand((5, 5)).astype('float32')
layer2 = paddle.rand((5, 4)).astype('float32')
bilinear = paddle.nn.Bilinear(
    in1_features=5, in2_features=4, out_features=1000)
result = bilinear(layer1,layer2)    # result shape [5, 1000]