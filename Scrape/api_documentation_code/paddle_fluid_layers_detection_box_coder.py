import paddle.fluid as fluid
import paddle
paddle.enable_static()
# For encode
prior_box_encode = fluid.data(name='prior_box_encode',
                      shape=[512, 4],
                      dtype='float32')
target_box_encode = fluid.data(name='target_box_encode',
                       shape=[81, 4],
                       dtype='float32')
output_encode = fluid.layers.box_coder(prior_box=prior_box_encode,
                        prior_box_var=[0.1,0.1,0.2,0.2],
                        target_box=target_box_encode,
                        code_type="encode_center_size")
# For decode
prior_box_decode = fluid.data(name='prior_box_decode',
                      shape=[512, 4],
                      dtype='float32')
target_box_decode = fluid.data(name='target_box_decode',
                       shape=[512, 81, 4],
                       dtype='float32')
output_decode = fluid.layers.box_coder(prior_box=prior_box_decode,
                        prior_box_var=[0.1,0.1,0.2,0.2],
                        target_box=target_box_decode,
                        code_type="decode_center_size",
                        box_normalized=False,
                        axis=1)