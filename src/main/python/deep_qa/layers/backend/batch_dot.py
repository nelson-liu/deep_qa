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

        if mask_a is None and mask_b is None:
            return None
        elif mask_a is None:
            mask_a = K.sum(K.ones_like(tensor_a), axis=-1)
        elif mask_b is None:
            # (batch_size, b_length)
            mask_b = K.sum(K.ones_like(tensor_b), axis=-1)
        float_mask_a = K.cast(mask_a, "float32")
        float_mask_b = K.cast(mask_b, "float32")
        mask_a_dot_axis = K.ndim(mask_a) - 1
        mask_b_dot_axis = K.ndim(mask_b) - 1
        if b_dot_axis == a_dot_axis:
            # tensor_a and tensor_b have the same length.
            float_mask_a = K.expand_dims(float_mask_a, dim=-1)
            float_mask_b = K.expand_dims(float_mask_b, dim=-1)
            final_mask = K.batch_dot(float_mask_a, float_mask_b,
                                     axes=(a_dot_axis, b_dot_axis))
        elif a_dot_axis < b_dot_axis:
            # tensor_a has less dimensions than tensor_b.
            # We tile tensor_a to have the same shape as tensor_b
            float_mask_a = K.expand_dims(float_mask_a, dim=-1)
            # We take the shape of mask_b as a hacky way of getting
            # around K.int_shape() and the Theano backend.
            tiled_float_mask_a = K.repeat_elements(float_mask_a,
                                                   K.int_shape(mask_b)[-1],
                                                   mask_b_dot_axis)
            final_mask = tiled_float_mask_a * float_mask_b
        else:
            # tensor_a has more dimensions than tensor_b.
            # We tile tensor_b to have the same shape as tensor_a
            float_mask_b = K.expand_dims(float_mask_b, dim=-1)
            # We take the shape of mask_a as a hacky way of getting
            # around K.int_shape() and the Theano backend.
            tiled_float_mask_b = K.repeat_elements(float_mask_b,
                                                   K.int_shape(mask_a)[-1],
                                                   mask_a_dot_axis)
            final_mask = float_mask_a * tiled_float_mask_b
        return final_mask

    @overrides
    def get_output_shape_for(self, input_shape):
        tensor_a_shape, tensor_b_shape = input_shape
        a_dot_axis = len(tensor_a_shape) - 1
        b_dot_axis = len(tensor_b_shape) - 1
        if b_dot_axis < a_dot_axis:
            tensor_b_shape += (1,)
        if b_dot_axis > a_dot_axis:
            tensor_a_shape += (1,)

        # This assumes that we do the dot product over the last dimension
        final_out_shape = []
        for i in range(0, len(tensor_a_shape)):
            if i != a_dot_axis:
                final_out_shape.append(tensor_a_shape[i])
        for i in range(len(tensor_b_shape)-2, len(tensor_b_shape)):
            if i != b_dot_axis and i != 0:
                final_out_shape.append(tensor_b_shape[i])
        if b_dot_axis != a_dot_axis:
            # remove the 1 we inserted
            final_out_shape.pop(a_dot_axis)
        if len(final_out_shape) == 1:
            final_out_shape.append(1)
        return tuple(final_out_shape)

    @overrides
    def call(self, x, mask=None):
        tensor_a, tensor_b = x
        if (K.ndim(tensor_a) > 3 or K.ndim(tensor_b) > 3) and K.backend() == 'theano':
            raise RuntimeError("K.batch_dot() in theano is broken for tensors with more than"
                               " three dimensions.  Use tensorflow instead.")

        a_dot_axis = K.ndim(tensor_a) - 1
        b_dot_axis = K.ndim(tensor_b) - 1
        if a_dot_axis > b_dot_axis:
            tensor_b = K.expand_dims(tensor_b, dim=-1)
        if a_dot_axis < b_dot_axis:
            tensor_a = K.expand_dims(tensor_a, dim=-1)
        a_dot_b = K.batch_dot(tensor_a, tensor_b, axes=(a_dot_axis, b_dot_axis))
        if a_dot_axis != b_dot_axis:
            a_dot_b = K.squeeze(a_dot_b, axis=a_dot_axis)
        return a_dot_b
