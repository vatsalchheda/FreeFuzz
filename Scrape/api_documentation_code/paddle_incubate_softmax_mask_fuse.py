# required: gpu
import paddle
import paddle.incubate as incubate

x = paddle.rand([2, 8, 8, 32])
mask = paddle.rand([2, 1, 8, 32])

rst = incubate.softmax_mask_fuse(x, mask)
# [[[[0.02404429, 0.04658398, 0.02746007, ..., 0.01489375, 0.02397441, 0.02851614] ... ]]]