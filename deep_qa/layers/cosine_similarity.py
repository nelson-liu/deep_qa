from keras import backend as K
from keras.layers import Layer
from ..tensors.backend import cosine_similarity


class CosineSimilarity(Layer):
    """
    This Layer takes 2 inputs, and computes the cosine similarity between
    them. The first input must be a vector. The second input is a
    3-dimensional tensor containing multiple vectors. This function
    will return the cosine similarity for each vector in the second tensor.

    Inputs:
        - vector_a: shape ``(batch_size, embedding size)``
        - tensor_b: shape ``(batch_size, number of documents, embedding size)``

    Output:
        - cosine_similarities: shape ``(batch_size, number of documents)``
          containing the cosine similarity of each document with vector_a.

    Parameters
    ----------
    target_index : int
        The word index to extract the forward and backward GRU output from.
    """
    def __init__(self, target_index, **kwargs):
        self.supports_masking = True
        super(CosineSimilarity, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shapes):
        return (input_shapes[1][0], input_shapes[1][1])

    def compute_mask(self, inputs, input_mask=None):  # pylint: disable=unused-argument
        return None

    def call(self, inputs, mask=None):
        vector_a, tensor_b = inputs
        similarities = cosine_similarity(vector_a, tensor_b)
        return similarities
