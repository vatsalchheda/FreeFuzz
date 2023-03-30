import numpy as np
import paddle.fluid as fluid

# use as generator

data = np.array([[2, 3], [4, 5]]).astype('float32')
with fluid.dygraph.guard():
    l0 = fluid.Linear(2, 2)  # l0.weight.gradient() is None
    l1 = fluid.Linear(2, 2)
    with fluid.dygraph.no_grad():
        # l1.weight.stop_gradient is False
        tmp = l1.weight * 2  # tmp.stop_gradient is True
    x = fluid.dygraph.to_variable(data)
    y = l0(x) + tmp
    o = l1(y)
    o.backward()
    print(tmp.gradient() is None)  # True
    print(l0.weight.gradient() is None)  # False

# use as decorator

@fluid.dygraph.no_grad
def test_layer():
    with fluid.dygraph.guard():
        inp = np.ones([3, 1024], dtype='float32')
        t = fluid.dygraph.base.to_variable(inp)
        linear1 = fluid.Linear(1024, 4, bias_attr=False)
        linear2 = fluid.Linear(4, 4)
        ret = linear1(t)
        dy_ret = linear2(ret)

test_layer()