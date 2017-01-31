from keras import backend as K
from keras.layers import Layer
from ...common.checks import ConfigurationError
from ...layers.attention.matrix_attention import MatrixAttention
from ...tensors.masked_operations import masked_batch_dot
from ...tensors.similarity_functions import similarity_functions


class GatedAttention(Layer):
    '''
    This Layer takes two inputs: a vector and a matrix.  We compute the similarity between the
    vector and each row in the matrix, and then perform a softmax over rows using those computed
    similarities.  We handle masking properly for masked rows in the matrix, though we ignore any
    masking on the vector.

    By default similarity is computed with a dot product, but you can alternatively use a
    parameterized similarity function if you wish.

    Input shapes:
        vector: (batch_size, embedding_dim), mask is ignored if provided
        matrix: (batch_size, num_rows, embedding_dim), with mask (batch_size, num_rows)
    Output shape: (batch_size, num_rows), no mask (masked input rows have value 0 in the output)
    '''
    def __init__(self, **kwargs):
        self.supports_masking = True
        # We need to wait until below to actually handle this, because self.name gets set in
        # super.__init__.
        # allowed gating functions are "*" (multiply), "+" (sum), and "||" (concatenate)
        self.gating_function = kwargs.pop('gating_function', "*")
        super(GatedAttention, self).__init__(**kwargs)

    def compute_mask(self, inputs, mask=None):
        # pylint: disable=unused-argument
        # We do not need a mask beyond this layer.
        return None

    def get_output_shape_for(self, input_shapes):
        return (input_shapes[1][0], input_shapes[1][1], input_shapes[1][2])

    def call(self, inputs, mask=None):
        # question matrix is of shape (batch, question length, biGRU hidden length)
        # document matrix is of shape (batch, document length, biGRU hidden length)
        question_matrix, document_matrix = inputs

        matrix_attention = MatrixAttention()
        matrix_attention.build([K.int_shape(question_matrix), K.int_shape(document_matrix)])

        # question document attention is of shape (batch, document length, question length)
        question_document_attention = matrix_attention.call([document_matrix, question_matrix])

        # permuted question matrix is of shape (batch, biGRU hidden length, question length)
        permuted_question_matrix = K.permute_dimensions(question_matrix, (0, 2, 1))

        # question update is of shape (batch, document length, biGRU hidden length)
        question_update = masked_batch_dot(question_document_attention,
                                           permuted_question_matrix,
                                           None, None)

        # use the gating function to calculate the new document representation
        if self.gating_function == "*":
            # shape (batch, document length, biGRU hidden length)
            return question_update * document_matrix
        if self.gating_function == "+":
            # shape (batch, document length, biGRU hidden length)
            return question_update + document_matrix
        if self.gating_function == "||":
            # shape (batch, document length, biGRU hidden length*2)
            return K.concatenate(question_update, document_matrix)
        raise ConfigurationError("Invalid gating function "
                                 "found {}".format(self.gating_function))
