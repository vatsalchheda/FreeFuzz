import paddle
import numpy

layer1 = numpy.random.random((5, 5)).astype('float32')
layer2 = numpy.random.random((5, 4)).astype('float32')
bilinearTensorProduct = paddle.nn.BilinearTensorProduct(
    input1_dim=5, input2_dim=4, output_dim=1000)
ret = bilinearTensorProduct(paddle.to_tensor(layer1),
                            paddle.to_tensor(layer2))