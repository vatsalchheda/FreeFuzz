import paddle
# x is a 2-D Tensor:
x = paddle.to_tensor([[float('nan'), 0.3, 0.5, 0.9],
                      [0.1, 0.2, float('-nan'), 0.7]])
out1 = paddle.nanmean(x)
# [0.44999996]
out2 = paddle.nanmean(x, axis=0)
# [0.1, 0.25, 0.5, 0.79999995]
out3 = paddle.nanmean(x, axis=0, keepdim=True)
# [[0.1, 0.25, 0.5, 0.79999995]]
out4 = paddle.nanmean(x, axis=1)
# [0.56666666 0.33333334]
out5 = paddle.nanmean(x, axis=1, keepdim=True)
# [[0.56666666]
#  [0.33333334]]

# y is a 3-D Tensor:
y = paddle.to_tensor([[[1, float('nan')], [3, 4]],
                       [[5, 6], [float('-nan'), 8]]])
out6 = paddle.nanmean(y, axis=[1, 2])
# [2.66666675, 6.33333349]
out7 = paddle.nanmean(y, axis=[0, 1])
# [3., 6.]