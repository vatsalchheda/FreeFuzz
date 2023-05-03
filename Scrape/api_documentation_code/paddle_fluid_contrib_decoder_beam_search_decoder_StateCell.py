import paddle
hidden_state = paddle.fluid.contrib.decoder.beam_search_decoder.InitState(init=None, need_reorder=True)
state_cell = paddle.fluid.contrib.decoder.beam_search_decoder.StateCell(
    inputs={'current_word': None},
    states={'h': hidden_state},
    out_state='h')

decoder = paddle.fluid.contrib.decoder.beam_search_decoder.TrainingDecoder(state_cell)
with decoder.block():
    current_word = decoder.step_input(1)
    decoder.state_cell.compute_state(inputs={'x': current_word})
    current_score = paddle.layers.fc(input=decoder.state_cell.get_state('h'),
                              size=32,
                              act='softmax')
    decoder.state_cell.update_states()
    decoder.output(current_score)