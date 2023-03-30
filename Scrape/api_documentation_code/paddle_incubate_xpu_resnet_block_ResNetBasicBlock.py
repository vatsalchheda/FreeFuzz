# required: xpu
import paddle
from paddle.incubate.xpu.resnet_block import ResNetBasicBlock

ch_in = 4
ch_out = 8
x = paddle.uniform((2, ch_in, 16, 16), dtype='float32', min=-1., max=1.)
resnet_basic_block = ResNetBasicBlock(num_channels1=ch_in,
                                    num_filter1=ch_out,
                                    filter1_size=3,
                                    num_channels2=ch_out,
                                    num_filter2=ch_out,
                                    filter2_size=3,
                                    num_channels3=ch_in,
                                    num_filter3=ch_out,
                                    filter3_size=1,
                                    stride1=1,
                                    stride2=1,
                                    stride3=1,
                                    act='relu',
                                    padding1=1,
                                    padding2=1,
                                    padding3=0,
                                    has_shortcut=True)
out = resnet_basic_block.forward(x)

print(out.shape) # [2, 8, 16, 16]