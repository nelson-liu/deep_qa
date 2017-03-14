from keras import backend as K
from keras.layers import Layer
import time

class PrintTensor(Layer):
    """
    This Layer takes a tensor and input, and returns the same tensor.
    However, it has the side effect of printing to stdout various configurable
    attributes. This is largely a layer because ``Lambda`` layers do not
    support masking (and it's slightly nicer to use in a model definition).

    Note that this will issue a print statement every time data goes through
    it (each batch). This quickly gets verbose, so logging to a file is encouraged
    when using this.

    Inputs:
        - input_tensor: a tensor of arbitrary shape. It can optionally
          take a mask, and we will just return what was given to us.

    Output:
        - input_tensor, the exact same as the input.

    Parameters
    ----------
    message: Callable or None, optional (default=None)
        If not None, prints the evaluation of the callable (converted to
        a string, if possible) as a message with the value.
    """
    def __init__(self, message=lambda: "", **kwargs):
        self.message = message
        self.supports_masking = True
        super(PrintTensor, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes

    def compute_mask(self, inputs, input_mask=None):  # pylint: disable=unused-argument
        return input_mask

    def call(self, input, mask=None):
        if self.message:
            # return K.print_tensor(input, "\n" + str(self.message()))
            return K.print_tensor(input, "\n" + str(time.clock()))
        else:
            return K.print_tensor(input)

    def get_config(self):
        config = {'message': self.message}
        base_config = super(PrintTensor, self).get_config()
        config.update(base_config)
        return config
