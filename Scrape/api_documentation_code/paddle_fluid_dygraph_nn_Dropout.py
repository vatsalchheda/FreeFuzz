import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable
import numpy as np

x = np.random.random(size=(3, 10, 3, 7)).astype('float32')
with fluid.dygraph.guard():
    x = to_variable(x)
    m = fluid.dygraph.Dropout(p=0.5)
    droped_train = m(x)
    # switch to eval mode
    m.eval()
    droped_eval = m(x)