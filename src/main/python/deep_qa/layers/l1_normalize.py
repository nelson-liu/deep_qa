from keras import backend as K
from keras.layers import Layer
from overrides import overrides
from ..common.tensors import l1_normalize


class L1Normalize(Layer):
    """
    This Layer normalizes a tensor by its L1 norm. This could just be a
    ``Lambda`` layer that calls our ``tensors.l1_normalize`` function,
    except that ``Lambda`` layers do not properly handle masked input.

    The expected input to this layer is a tensor of shape
    ``(batch_size, x)``, with an optional mask of the same shape.
    We also accept as input a tensor of shape ``(batch_size, x, 1)``,
    which will be squeezed to shape ``(batch_size, x)``
    (though the mask must still be of shape ``(batch_size, x)``).

    We give no output mask, as we expect this to only be used at the end of
    the model, to get a final probability distribution over class labels. If
    you need this to propagate the mask for your model, it would be pretty
    easy to change it to optionally do so - submit a PR.
    """

    def __init__(self, **kwargs):
        self.supports_masking = True
        super(L1Normalize, self).__init__(**kwargs)

    @overrides
    def compute_mask(self, inputs, mask=None):
        # pylint: disable=unused-argument
        # We do not need a mask beyond this layer.
        return None

    @overrides
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1])

    @overrides
    def call(self, tensor_to_normalize, mask=None):
        if K.ndim(tensor_to_normalize) == 3:
            tensor_to_normalize = K.squeeze(tensor_to_normalize, axis=2)
        if K.ndim(tensor_to_normalize) != 2:
            raise ValueError("L1Normalize layer only supports inputs of shape "
                             "(batch_size, x) or (batch_size, x, 1)")
        return l1_normalize(tensor_to_normalize, mask)
