import paddle.fluid as fluid
import paddle
paddle.enable_static()
pb = fluid.data(
    name='prior_box', shape=[None, 4], dtype='float32')
pbv = fluid.data(
    name='prior_box_var', shape=[4], dtype='float32')
loc = fluid.data(
    name='target_box', shape=[None, 4*81], dtype='float32')
scores = fluid.data(
    name='scores', shape=[None, 81], dtype='float32')
decoded_box, output_assign_box = fluid.layers.box_decoder_and_assign(
    pb, pbv, loc, scores, 4.135)