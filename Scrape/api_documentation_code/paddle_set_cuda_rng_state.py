import paddle
sts = paddle.get_cuda_rng_state()
paddle.set_cuda_rng_state(sts)