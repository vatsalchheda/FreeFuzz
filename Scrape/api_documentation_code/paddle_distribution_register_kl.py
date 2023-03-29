import paddle

@paddle.distribution.register_kl(paddle.distribution.Beta, paddle.distribution.Beta)
def kl_beta_beta():
    pass # insert implementation here