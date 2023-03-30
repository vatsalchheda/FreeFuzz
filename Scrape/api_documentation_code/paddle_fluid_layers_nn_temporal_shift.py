import paddle
import paddle.nn.functional as F

input = paddle.randn([6, 4, 2, 2])
out = F.temporal_shift(x=input, seg_num=2, shift_ratio=0.2)