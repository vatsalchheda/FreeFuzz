System Message: ERROR/3 (/usr/local/lib/python3.8/site-packages/paddle/fluid/contrib/decoder/beam_search_decoder.py:docstring of paddle.fluid.contrib.decoder.beam_search_decoder.StateCell, line 27)
Error in “code-block” directive: maximum 1 argument(s) allowed, 13 supplied.
.. code-block:: python
  hidden_state = InitState(init=encoder_out, need_reorder=True)
  state_cell = StateCell(
      inputs={'current_word': None},
      states={'h': hidden_state},
      out_state='h')