import paddle
paddle.enable_static()
# for lod mode
import paddle.fluid as fluid
x = fluid.data(name='x', shape=[None, 8], dtype='float32', lod_level=1)
cost = fluid.layers.ctc_greedy_decoder(input=x, blank=0)

# for padding mode
x_pad = fluid.data(name='x_pad', shape=[10, 4, 8], dtype='float32')
x_pad_len = fluid.data(name='x_pad_len', shape=[10, 1], dtype='int64')
out, out_len = fluid.layers.ctc_greedy_decoder(input=x_pad, blank=0,
                input_length=x_pad_len)