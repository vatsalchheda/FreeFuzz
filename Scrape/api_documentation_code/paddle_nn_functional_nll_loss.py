import paddle
from paddle.nn.functional import nll_loss
log_softmax = paddle.nn.LogSoftmax(axis=1)

input = paddle.to_tensor([[0.88103855, 0.9908683 , 0.6226845 ],
          [0.53331435, 0.07999352, 0.8549948 ],
          [0.25879037, 0.39530203, 0.698465  ],
          [0.73427284, 0.63575995, 0.18827209],
          [0.05689114, 0.0862954 , 0.6325046 ]], "float32")
log_out = log_softmax(input)
label = paddle.to_tensor([0, 2, 1, 1, 0], "int64")
result = nll_loss(log_out, label)
print(result) # Tensor(shape=[1], dtype=float32, place=CPUPlace, stop_gradient=True, [1.07202101])