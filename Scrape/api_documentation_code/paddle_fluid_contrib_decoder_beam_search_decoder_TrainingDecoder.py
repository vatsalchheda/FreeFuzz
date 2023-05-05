System Message: ERROR/3 (/usr/local/lib/python3.8/site-packages/paddle/fluid/contrib/decoder/beam_search_decoder.py:docstring of paddle.fluid.contrib.decoder.beam_search_decoder.TrainingDecoder, line 16)
Error in “code-block” directive: maximum 1 argument(s) allowed, 18 supplied.
.. code-block:: python
  decoder = TrainingDecoder(state_cell)
  with decoder.block():
      current_word = decoder.step_input(trg_embedding)
      decoder.state_cell.compute_state(inputs={'x': current_word})
      current_score = layers.fc(input=decoder.state_cell.get_state('h'),
                                size=32,
                                act='softmax')
      decoder.state_cell.update_states()
      decoder.output(current_score)