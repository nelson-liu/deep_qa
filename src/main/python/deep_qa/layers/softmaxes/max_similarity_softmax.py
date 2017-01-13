from keras import backend as K
from keras.engine import Layer

from ...common.tensors import switch


class MaxSimilaritySoftmax(Layer):
    '''
    This layer takes encoded questions and knowledge in a multiple choice setting and computes the similarity
    between each of the question embeddings and the background knowledge, and returns a softmax over the options.
    Inputs:
        encoded_questions (batch_size, num_options, encoding_dim)
        encoded_knowledge (batch_size, num_options, knowledge_length, encoding_dim)
    Output: option_probabilities (batch_size, num_options)
    We made this a concrete layer instead of a lambda layer to properly handle input masks.
    '''
    def __init__(self, knowledge_axis, max_knowledge_length, **kwargs):
        self.knowledge_axis = knowledge_axis
        self.max_knowledge_length = max_knowledge_length
        super(MaxSimilaritySoftmax, self).__init__(**kwargs)

    def compute_mask(self, inputs, mask=None):
        # pylint: disable=unused-argument
        return None

    def get_output_shape_for(self, input_shapes):
        # (batch_size, num_options)
        return (input_shapes[0][0], input_shapes[0][1])

    def call(self, inputs, mask=None):
        questions, knowledge = inputs
        if mask is not None:
            question_mask, knowledge_mask = mask
            if K.ndim(question_mask) < K.ndim(questions):
                # To use switch, we need the question_mask to be the same size as the encoded_questions.
                # Therefore we expand it and multiply by ones in the shape that we need.
                question_mask = K.expand_dims(question_mask) * K.cast(K.ones_like(questions), dtype="uint8")
            questions = switch(question_mask, questions, K.zeros_like(questions))
            if K.ndim(knowledge_mask) < K.ndim(knowledge):
                # To use switch, we need the knowledge_mask to be the same size as the encoded_knowledge.
                # Therefore we expand it and multiply by ones in the shape that we need.
                knowledge_mask = K.expand_dims(knowledge_mask) * K.cast(K.ones_like(knowledge), dtype="uint8")
            knowledge = switch(knowledge_mask, knowledge, K.zeros_like(knowledge))
        expanded_questions = K.expand_dims(questions, dim=self.knowledge_axis)
        # (samples, num_options, encoding_dim, knowledge_length)
        tiled_questions = K.tile(K.permute_dimensions(expanded_questions, (0, 1, 3, 2)),
                                 (1, 1, 1, self.max_knowledge_length))
        # (samples, num_options, knowledge_length, encoding_dim)
        tiled_questions = K.permute_dimensions(tiled_questions, (0, 1, 3, 2))
        # (samples, num_options, knowledge_length)
        question_knowledge_product = K.sum(tiled_questions * knowledge, axis=-1)
        max_knowledge_similarity = K.max(question_knowledge_product, axis=-1)  # (samples, num_options)

        # TODO(matt): this softmax needs to be masked_softmax instead of what it currently is.  But
        # what mask is that?  Is it question_mask?  The mask should be (batch_size, num_options).
        return K.softmax(max_knowledge_similarity)
