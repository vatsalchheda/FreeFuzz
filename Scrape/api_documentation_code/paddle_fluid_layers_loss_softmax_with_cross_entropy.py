import paddle
import numpy as np

data = np.random.rand(128).astype("float32")
label = np.random.rand(1).astype("int64")
data = paddle.to_tensor(data)
label = paddle.to_tensor(label)
linear = paddle.nn.Linear(128, 100)
x = linear(data)
out = paddle.nn.functional.softmax_with_cross_entropy(logits=x, label=label)
print(out)