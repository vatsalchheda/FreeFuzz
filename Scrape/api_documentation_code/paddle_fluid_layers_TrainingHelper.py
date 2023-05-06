import paddle
paddle.enable_static()
import paddle.fluid as fluid
import paddle.fluid.layers as layers
trg_emb = fluid.data(name="trg_emb",
                     shape=[None, None, 128],
                     dtype="float32")
trg_seq_length = fluid.data(name="trg_seq_length",
                            shape=[None],
                            dtype="int64")
helper = layers.TrainingHelper(trg_emb, trg_seq_length)
decoder_cell = layers.GRUCell(hidden_size=128)
decoder = layers.BasicDecoder(decoder_cell, helper)
outputs = layers.dynamic_decode(
    decoder,
    inits=decoder_cell.get_initial_states(trg_emb),
    is_test=False)