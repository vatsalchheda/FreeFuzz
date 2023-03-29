import paddle.fluid as fluid

sentence = fluid.data(name='sentence', shape=[None, 32], dtype='float32', lod_level=1)
encoder_proj = fluid.data(name='encoder_proj', shape=[None, 32], dtype='float32', lod_level=1)
decoder_boot = fluid.data(name='boot', shape=[None, 10], dtype='float32')

drnn = fluid.layers.DynamicRNN()
with drnn.block():
    # Set sentence as RNN's input, each time step processes a word from the sentence
    current_word = drnn.step_input(sentence)
    # Set encode_proj as RNN's static input
    encoder_word = drnn.static_input(encoder_proj)
    # Initialize memory with boot_memory, which need reorder according to RNN's input sequences
    memory = drnn.memory(init=decoder_boot, need_reorder=True)
    fc_1 = fluid.layers.fc(input=encoder_word, size=30)
    fc_2 = fluid.layers.fc(input=current_word, size=30)
    decoder_inputs = fc_1 + fc_2
    hidden, _, _ = fluid.layers.gru_unit(input=decoder_inputs, hidden=memory, size=30)
    # Update memory with hidden
    drnn.update_memory(ex_mem=memory, new_mem=hidden)
    out = fluid.layers.fc(input=hidden, size=10, bias_attr=True, act='softmax')
    # Set hidden and out as RNN's outputs
    drnn.output(hidden, out)

# Get RNN's result
hidden, out = drnn()
# Get RNN's result of the last time step
last = fluid.layers.sequence_last_step(out)