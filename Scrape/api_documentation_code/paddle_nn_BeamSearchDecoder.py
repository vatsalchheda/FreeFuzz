import numpy as np
import paddle
from paddle.nn import BeamSearchDecoder, dynamic_decode
from paddle.nn import GRUCell, Linear, Embedding
trg_embeder = Embedding(100, 32)
output_layer = Linear(32, 32)
decoder_cell = GRUCell(input_size=32, hidden_size=32)
decoder = BeamSearchDecoder(decoder_cell,
                            start_token=0,
                            end_token=1,
                            beam_size=4,
                            embedding_fn=trg_embeder,
                            output_fn=output_layer)