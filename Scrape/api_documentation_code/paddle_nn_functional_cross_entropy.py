# soft labels
import paddle
paddle.seed(99999)
axis = -1
ignore_index = -100
N = 4
C = 3
shape = [N, C]
reduction='mean'
weight = None
logits = paddle.uniform(shape, dtype='float64', min=0.1, max=1.0)
labels = paddle.uniform(shape, dtype='float64', min=0.1, max=1.0)
labels /= paddle.sum(labels, axis=axis, keepdim=True)
paddle_loss_mean = paddle.nn.functional.cross_entropy(
                                                      logits,
                                                      labels,
                                                      soft_label=True,
                                                      axis=axis,
                                                      weight=weight,
                                                      reduction=reduction)
print(paddle_loss_mean.numpy()) #[1.12908343]