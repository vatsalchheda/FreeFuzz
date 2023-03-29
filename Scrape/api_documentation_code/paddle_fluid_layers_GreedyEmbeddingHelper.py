import paddle.fluid as fluid
import paddle.fluid.layers as layers
trg_emb = fluid.data(name="trg_emb",
                     shape=[None, None, 128],
                     dtype="float32")

trg_embeder = lambda x: fluid.embedding(
    x, size=[10000, 128], param_attr=fluid.ParamAttr(name="trg_embedding"))
output_layer = lambda x: layers.fc(x,
                                size=10000,
                                num_flatten_dims=len(x.shape) - 1,
                                param_attr=fluid.ParamAttr(name=
                                                        "output_w"),
                                bias_attr=False)
helper = layers.GreedyEmbeddingHelper(trg_embeder, start_tokens=0, end_token=1)
decoder_cell = layers.GRUCell(hidden_size=128)
decoder = layers.BasicDecoder(decoder_cell, helper, output_fn=output_layer)
outputs = layers.dynamic_decode(
    decoder=decoder, inits=decoder_cell.get_initial_states(encoder_output))