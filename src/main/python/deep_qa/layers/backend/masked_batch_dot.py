from keras.layers import Layer
from overrides import overrides

from ...tensors.masked_operations import masked_batch_dot


class MaskedBatchDot(Layer):
    """
    This ``Layer`` calls ``masked_batch_dot()`` on two inputs ``tensor_a`` and ``tensor_b``.
    This function will work for tensors of arbitrary size as long as
    ``abs(K.ndim(tensor_a) - K.ndim(tensor_b)) < 1``, due to limitations in ``K.batch_dot()``.

    We always assume the dimension to perform the dot is the last one, and that the
    masks have one fewer dimension that the tensors.

    Inputs:
        - tensor_a: tensor with ``ndim >= 2``.
        - tensor_b: tensor with ``ndim >= 2``.

    Output:
        - a_dot_b
    """
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(MaskedBatchDot, self).__init__(**kwargs)

    @overrides
    def compute_mask(self, inputs, mask=None):
        # pylint: disable=unused-argument
        # We do not need a mask beyond this layer.
        return None

    @overrides
    def get_output_shape_for(self, input_shape):
        # This assumes that we do the dot product over the last dimension, so this
        # will have to change if that assumption changes in masked_batch_dot.
        a_out_shape = tuple([input_shape[0][i] for i in range(0, len(input_shape[0]) - 1)])
        b_out_shape = tuple([input_shape[1][i] for i in range(1, len(input_shape[1]) - 1)])
        return a_out_shape + b_out_shape

    @overrides
    def call(self, x, mask=None):
        tensor_a, tensor_b = x
        mask_a, mask_b = mask
        return masked_batch_dot(tensor_a, tensor_b, mask_a, mask_b)
