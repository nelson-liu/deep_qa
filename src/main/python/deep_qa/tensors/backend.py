"""
These are utility functions that are similar to calls to Keras' backend.  Some of these are here
because a current function in keras.backend is broken, some are things that just haven't been
implemented.
"""
import keras.backend as K


def switch(cond, then_tensor, else_tensor):
    """
    Keras' implementation of K.switch currently uses tensorflow's switch function, which only
    accepts scalar value conditions, rather than boolean tensors which are treated in an
    elementwise function.  This doesn't match with Theano's implementation of switch, but using
    tensorflow's select, we can exactly retrieve this functionality.
    """

    if K.backend() == 'tensorflow':
        import tensorflow as tf
        cond_shape = cond.get_shape()
        input_shape = then_tensor.get_shape()
        if cond_shape[-1] != input_shape[-1] and cond_shape[-1] == 1:
            # This happens when the last dim in the input is an embedding dimension. Keras usually does not
            # mask the values along that dimension. Theano broadcasts the value passed along this dimension,
            # but TF does not. Using K.dot() since cond can be a tensor.
            cond = K.dot(tf.cast(cond, tf.float32), tf.ones((1, input_shape[-1])))
        return tf.select(tf.cast(cond, dtype=tf.bool), then_tensor, else_tensor)
    else:
        import theano.tensor as T
        return T.switch(cond, then_tensor, else_tensor)


def last_dim_flatten(input_tensor):
    '''
    Takes a tensor and returns a matrix while preserving only the last dimension from the input.
    '''
    input_ndim = K.ndim(input_tensor)
    shuffle_pattern = (input_ndim - 1,) + tuple(range(input_ndim - 1))
    dim_shuffled_input = K.permute_dimensions(input_tensor, shuffle_pattern)
    return K.transpose(K.batch_flatten(dim_shuffled_input))


def tile_vector(vector, matrix):
    """
    DEPRECATED: I'm pretty sure K.repeat_elements does exactly this, and should be used instead.

    This method takes a (collection of) vector(s) (shape: (batch_size, vector_dim)), and tiles that
    vector a number of times, giving a matrix of shape (batch_size, tile_length, vector_dim).  (I
    say "vector" and "matrix" here because I'm ignoring the batch_size).  We need the matrix as
    input so we know what the tile_length is - the matrix is otherwise ignored.

    This is necessary in a number of places in the code.  For instance, if you want to do a dot
    product of a vector with all of the vectors in a matrix, the most efficient way to do that is
    to tile the vector first, then do an element-wise product with the matrix, then sum out the
    last mode.  So, we capture this functionality here.

    This is not done as a Keras Layer, however; if you want to use this function, you'll need to do
    it _inside_ of a Layer somehow, either in a Lambda or in the call() method of a Layer you're
    writing.
    """
    # Tensorflow can't use unknown sizes at runtime, so we have to make use of the broadcasting
    # ability of TF and Theano instead to create the tiled sentence encoding.

    # Shape: (tile_length, batch_size, vector_dim)
    k_ones = K.permute_dimensions(K.ones_like(matrix), [1, 0, 2])

    # Now we have a (tile_length, batch_size, vector_dim)*(batch_size, vector_dim)
    # elementwise multiplication which is broadcast. We then reshape back.
    tiled_vector = K.permute_dimensions(k_ones * vector, [1, 0, 2])
    return tiled_vector


def tile_scalar(scalar, vector):
    """
    DEPRECATED: I'm pretty sure K.repeat_elements does exactly this, and should be used instead.

    This method takes a (collection of) scalar(s) (shape: (batch_size, 1)), and tiles that
    scala a number of times, giving a vector of shape (batch_size, tile_length).  (I say "scalar"
    and "vector" here because I'm ignoring the batch_size).  We need the vector as input so we know
    what the tile_length is - the vector is otherwise ignored.

    This is not done as a Keras Layer, however; if you want to use this function, you'll need to do
    it _inside_ of a Layer somehow, either in a Lambda or in the call() method of a Layer you're
    writing.

    TODO(matt): we could probably make a more general `tile_tensor` method, which can do this for
    any dimenionsality.  There is another place in the code where we do this with a matrix and a
    tensor; all three of these can probably be one function.
    """
    # Tensorflow can't use unknown sizes at runtime, so we have to make use of the broadcasting
    # ability of TF and Theano instead to create the tiled sentence encoding.

    # Shape: (tile_length, batch_size)
    k_ones = K.permute_dimensions(K.ones_like(vector), [1, 0])

    # Now we have a (tile_length, batch_size) * (batch_size, 1) elementwise multiplication which is
    # broadcast. We then reshape back.
    tiled_scalar = K.permute_dimensions(k_ones * K.squeeze(scalar, axis=1), [1, 0])
    return tiled_scalar


def hardmax(unnormalized_attention, knowledge_length):
    """
    A similar operation to softmax, except all of the weight is placed on the mode of the
    distribution.  So, e.g., this function transforms [.34, .2, -1.4] -> [1, 0, 0].

    TODO(matt): we really should have this take an optional mask...
    """
    # (batch_size, knowledge_length)
    tiled_max_values = K.tile(K.expand_dims(K.max(unnormalized_attention, axis=1)), (1, knowledge_length))
    # We now have a matrix where every column in each row has the max knowledge score value from
    # the corresponding row in the unnormalized attention matrix.  Next, we will compare that
    # all-max matrix with the original input, resulting in ones where the column equals max and
    # zero everywhere else.
    # Shape: (batch_size, knowledge_length)
    bool_max_attention = K.equal(unnormalized_attention, tiled_max_values)
    # Needs to be cast to be compatible with TensorFlow
    max_attention = K.cast(bool_max_attention, 'float32')
    return max_attention


def apply_feed_forward(input_tensor, weights, activation):
    '''
    Takes an input tensor, sequence of weights and an activation and builds an MLP.
    This can also be achieved by defining a sequence of Dense layers in Keras, but doing this
    might be desirable if the operation needs to be done within the call method of a more complex
    layer. Moreover, we are not applying biases here. The input tensor can have any number of
    dimensions. But the last dimension, and the sequence of weights are expected to be compatible.
    '''
    current_tensor = input_tensor
    for weight in weights:
        current_tensor = activation(K.dot(current_tensor, weight))
    return current_tensor


def l1_normalize(tensor_to_normalize, mask=None):
    """
    Normalize a tensor by its L1 norm. Takes an optional mask.

    Parameters
    ----------
    tensor_to_normalize : Tensor
        Tensor of shape (batch size, x) to be normalized, where
        x is arbitrary.

    mask: Tensor, optional
        Tensor of shape (batch size, x) indicating which elements
        of ``tensor_to_normalize`` are padding and should
        not be considered when normalizing.

    Returns
    -------
    normalized_tensor : Tensor
        Normalized tensor with shape (batch size, x).
    """
    if mask is not None:
        tensor_to_normalize = switch(mask, tensor_to_normalize,
                                     K.zeros_like(tensor_to_normalize))
    norm = K.sum(tensor_to_normalize, keepdims=True)
    normalized_tensor = tensor_to_normalize / norm
    float32_normalized_tensor = K.cast(normalized_tensor, "float32")
    return float32_normalized_tensor
