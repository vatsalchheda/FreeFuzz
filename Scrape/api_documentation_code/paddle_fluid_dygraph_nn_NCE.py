import numpy as np
import paddle.fluid as fluid

window_size = 5
dict_size = 20
label_word = int(window_size // 2) + 1
inp_word = np.array([[1], [2], [3], [4], [5]]).astype('int64')
nid_freq_arr = np.random.dirichlet(np.ones(20) * 1000).astype('float32')

with fluid.dygraph.guard():
    words = []
    for i in range(window_size):
        words.append(fluid.dygraph.base.to_variable(inp_word[i]))

    emb = fluid.Embedding(
        size=[dict_size, 32],
        param_attr='emb.w',
        is_sparse=False)

    embs3 = []
    for i in range(window_size):
        if i == label_word:
            continue

        emb_rlt = emb(words[i])
        embs3.append(emb_rlt)

    embs3 = fluid.layers.concat(input=embs3, axis=1)
    nce = fluid.NCE(
                 num_total_classes=dict_size,
                 dim=embs3.shape[1],
                 num_neg_samples=2,
                 sampler="custom_dist",
                 custom_dist=nid_freq_arr.tolist(),
                 seed=1,
                 param_attr='nce.w',
                 bias_attr='nce.b')

    wl = fluid.layers.unsqueeze(words[label_word], axes=[0])
    nce_loss3 = nce(embs3, wl)