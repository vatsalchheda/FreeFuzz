# required: gpu
# Demo1: single model and optimizer:
import paddle

model = paddle.nn.Conv2D(3, 2, 3, bias_attr=False)
optimizer = paddle.optimizer.SGD(parameters=model.parameters())

model, optimizer = paddle.fluid.dygraph.amp_decorate(models=model, optimizers=optimizer, level='O2')

data = paddle.rand([10, 3, 32, 32])

with paddle.fluid.dygraph.amp_guard(enable=True, custom_white_list=None, custom_black_list=None, level='O2'):
    output = model(data)
    print(output.dtype) # FP16

# required: gpu
# Demo2: multi models and optimizers:
model2 = paddle.nn.Conv2D(3, 2, 3, bias_attr=False)
optimizer2 = paddle.optimizer.Adam(parameters=model2.parameters())

models, optimizers = paddle.fluid.dygraph.amp_decorate(models=[model, model2], optimizers=[optimizer, optimizer2], level='O2')

data = paddle.rand([10, 3, 32, 32])

with paddle.fluid.dygraph.amp_guard(enable=True, custom_white_list=None, custom_black_list=None, level='O2'):
    output = models[0](data)
    output2 = models[1](data)
    print(output.dtype) # FP16
    print(output2.dtype) # FP16

# required: gpu
# Demo3: optimizers is None:
model3 = paddle.nn.Conv2D(3, 2, 3, bias_attr=False)
optimizer3 = paddle.optimizer.Adam(parameters=model2.parameters())

model = paddle.fluid.dygraph.amp_decorate(models=model3, level='O2')

data = paddle.rand([10, 3, 32, 32])

with paddle.fluid.dygraph.amp_guard(enable=True, custom_white_list=None, custom_black_list=None, level='O2'):
    output = model(data)
    print(output.dtype) # FP16