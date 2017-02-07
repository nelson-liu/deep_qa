from keras import backend as K
from keras.layers import Layer


class Overlap(Layer):
    """
    This Layer takes 2 inputs: a tensor_a (e.g. a document) and a tensor_b
    (e.g. a question). It returns a one-hot vector with the same shape as tensor_a,
    indicating at each index whether the element in tensor_a appears in tensor_b.

    Inputs:
        - tensor_a: shape ``(batch_size, length_a)``
        - tensor_b shape ``(batch_size, length_b)``

    Output:
        - One-hot indicating overlap: shape ``(batch_size, length_a, 2)``
    """
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(Overlap, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shapes):
        return (input_shapes[0][0], input_shapes[0][1], 2)


    # This is a hack made necessary by
    # https://github.com/fchollet/keras/issues/5311
    def compute_mask(self, inputs, input_mask=None):  # pylint: disable=unused-argument
        if K.backend() == "tensorflow":
            tensor_a = inputs[0]
            length_a = K.int_shape(tensor_a)[1]
            return K.cast(K.ones_like(K.repeat_elements(K.expand_dims(tensor_a, 2),
                                                        length_a,
                                                        axis=2)), "bool")
        else:
            return None

    def call(self, inputs, mask=None):
        # tensor_a is of shape (batch size, length_a)
        # tensor_b is of shape (batch size, length_b)
        tensor_a, tensor_b = inputs
        length_a = K.int_shape(tensor_a)[1]
        length_b = K.int_shape(tensor_b)[1]
        # reshape tensor_a to shape (batch_size, length_a, length_b)
        tensor_a_tiled = K.repeat_elements(K.expand_dims(tensor_a, 2),
                                           length_b,
                                           axis=2)
        # reshape tensor_b to shape (batch_size, length_a, length_b)
        tensor_b_tiled = K.repeat_elements(K.expand_dims(tensor_b, 1),
                                           length_a,
                                           axis=1)
        overlap_mask = K.cast(K.equal(tensor_a_tiled, tensor_b_tiled), "float32")
        indices_overlap = K.sum(overlap_mask, axis=-1)
        binary_indices_overlap = K.cast(K.not_equal(indices_overlap,
                                                    K.zeros_like(indices_overlap)),
                                        "int32")
        one_hot_overlap = K.cast(K.one_hot(binary_indices_overlap, 2), "float32")
        return one_hot_overlap
