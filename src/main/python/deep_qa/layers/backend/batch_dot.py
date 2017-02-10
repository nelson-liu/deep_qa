import keras.backend as K
from keras.layers import Layer
from overrides import overrides


class BatchDot(Layer):
    """
    This ``Layer`` calls ``K.batch_dot()`` on two inputs ``tensor_a`` and ``tensor_b``.
    This function will work for tensors of arbitrary size as long as
    ``abs(K.ndim(tensor_a) - K.ndim(tensor_b)) < 1``, due to limitations in ``K.batch_dot()``.

    We always assume the dimension to perform the dot is the last one, and that
    the masks have one fewer dimension that the tensors. Note that this layer
    does not return zeroes in places that are masked, but does pass a correct
    mask forward. If this then gets fed into ``masked_softmax``, for instance,
    your tensor will be correctly normalized. We always assume the dimension to
    perform the dot is the last one, and that the masks have one fewer
    dimension than the tensors.

    Inputs:
        - tensor_a: tensor with ``ndim >= 2``.
        - tensor_b: tensor with ``ndim >= 2``.

    Output:
        - a_dot_b

    """
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(BatchDot, self).__init__(**kwargs)

    @overrides
    def compute_mask(self, inputs, mask=None):
        # pylint: disable=unused-argument
        tensor_a, tensor_b = inputs
        mask_a, mask_b = mask
        a_dot_axis = K.ndim(tensor_a) - 1
        b_dot_axis = K.ndim(tensor_b) - 1

        if K.ndim(tensor_a) < K.ndim(tensor_b):
            # To simplify the logic below, we'll make sure that tensor_a is always the bigger one.
            tensor_a, tensor_b = tensor_b, tensor_a
            mask_a, mask_b = mask_b, mask_a

        if mask_a is None and mask_b is None:
            return None
        elif mask_a is None:
            # (batch_size, a_length)
            mask_a = K.sum(K.ones_like(tensor_a), axis=-1)
        elif mask_b is None:
            # (batch_size, b_length)
            sum_axis = -1
            if b_dot_axis < a_dot_axis:
                sum_axis -= 1
            mask_b = K.sum(K.ones_like(tensor_b), axis=sum_axis)
        # Casting masks to float since we TF would complain if we multiplied bools.
        float_mask_a = K.cast(mask_a, 'float32')
        float_mask_b = K.cast(mask_b, 'float32')

        if b_dot_axis < a_dot_axis:
            float_mask_b = K.expand_dims(float_mask_b, dim=-1)
        else:
            float_mask_a = K.expand_dims(float_mask_a, dim=-1)
            float_mask_b = K.expand_dims(float_mask_b, dim=-2)
        # (batch_size, a_length, b_length)
        a2b_mask = float_mask_a * float_mask_b
        return a2b_mask


    @overrides
    def get_output_shape_for(self, input_shape):
        # This assumes that we do the dot product over the last dimension, so this
        # will have to change if that assumption changes in masked_batch_dot.
        a_out_shape = tuple([input_shape[0][i] for i in range(0, len(input_shape[0]) - 1)])
        b_out_shape = tuple([input_shape[1][i] for i in range(1, len(input_shape[1]) - 1)])
        final_out_shape = a_out_shape + b_out_shape
        if len(final_out_shape) == 1:
            final_out_shape += (1,)
        return final_out_shape

    @overrides
    def call(self, x, mask=None):
        tensor_a, tensor_b = x
        a_dot_axis = K.ndim(tensor_a) - 1
        b_dot_axis = K.ndim(tensor_b) - 1
        return K.batch_dot(tensor_a, tensor_b, axes=(a_dot_axis, b_dot_axis))
