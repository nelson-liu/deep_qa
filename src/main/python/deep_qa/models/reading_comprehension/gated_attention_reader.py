from copy import deepcopy
from typing import Any, Dict
from overrides import overrides
from keras.layers import Input, Dropout

from ...data.instances.whodidwhat_instance import WhoDidWhatInstance
from ...common.checks import ConfigurationError
from ...layers.backend.masked_batch_dot import MaskedBatchDot
from ...layers.attention.attention import Attention
from ...layers.attention.masked_softmax import MaskedSoftmax
from ...layers.option_attention_sum import OptionAttentionSum
from ...layers.attention.gated_attention import GatedAttention
from ...layers.l1_normalize import L1Normalize
from ...layers.vector_matrix_split import VectorMatrixSplit
from ...layers.bigru_index_selector import BiGRUIndexSelector
from ...training.text_trainer import TextTrainer
from ...training.models import DeepQaModel


class GatedAttentionReader(TextTrainer):
    """
    This TextTrainer implements the Gated Attention Reader model described in
    "Gated-Attention Readers for Text Comprehension" by Dhingra et. al 2016.
    """
    def __init__(self, params: Dict[str, Any]):
        self.max_question_length = params.pop('max_question_length', None)
        self.max_passage_length = params.pop('max_passage_length', None)
        self.max_option_length = params.pop('max_option_length', None)
        self.num_options = params.pop('num_options', None)
        # either "mean" or "sum"
        self.multiword_option_mode = params.pop('multiword_option_mode', "mean")
        # number of gated attention layers to use
        self.num_gated_attention_layers = params.pop('num_gated_attention_layers', 4)
        # dropout proportion after each gated attention layer.
        self.gated_attention_dropout = params.pop('gated_attention_dropout', 0.3)
        # If you are using the model on a cloze (fill in the blank) dataset,
        # indicate what token indicates the blank.
        self.cloze_token = params.pop('cloze_token', None)
        self.cloze_token_index = None
        # use the question document common word feature
        self.use_qd_common_feature = params.pop('qd_common_feature', True)
        super(GatedAttentionReader, self).__init__(params)

    @overrides
    def _build_model(self):
        """
        The basic outline here is that we'll pass the questions and the
        document / passage (think of this as a collection of possible answer
        choices) into a word embedding layer.
        """
        # get the index of the cloze token, if applicable
        if self.cloze_token is not None:
            self.cloze_token_index = self.data_indexer.get_word_index(self.cloze_token)

        # First we create input layers and pass the question and document
        # through embedding layers.

        # shape: (batch size, question_length)
        question_input_shape = self._get_sentence_shape(self.max_question_length)
        question_input = Input(shape=question_input_shape,
                               dtype='int32', name="question_input")
        if len(question_input_shape) > 1:
            question_indices = VectorMatrixSplit(split_axis=-1)(question_input)[0]

        # shape: (batch size, document_length)
        document_input_shape = self._get_sentence_shape(self.max_passage_length)
        document_input = Input(shape=self._get_sentence_shape(self.max_passage_length),
                               dtype='int32',
                               name="document_input")
        if len(document_input_shape) > 1:
            document_indices = VectorMatrixSplit(split_axis=-1)(document_input)[0]

        # shape: (batch size, number of options, num words in option)
        options_input_shape = ((self.num_options,) +
                               self._get_sentence_shape(self.max_option_length))
        options_input = Input(shape=options_input_shape,
                              dtype='int32', name="options_input")
        if len(options_input_shape) > 1:
            options_indices = VectorMatrixSplit(split_axis=-1)(options_input)[0]

        # shape: (batch size, question_length, embedding size)
        question_embedding = self._embed_input(question_input)

        # shape: (batch size, document_length, embedding size)
        document_embedding = self._embed_input(document_input)

        # We pass the question and document embedding through a variable
        # number of gated-attention layers.
        if self.num_gated_attention_layers < 1:
            raise ConfigurationError("Need at least one gated attention layer.")
        for i in range(self.num_gated_attention_layers-1):
            # Note that the size of the last dimension of the input
            # is not necessarily the embedding size in the second gated
            # attention layer and beyond.

            # We encode the question embeddings with a seq2seq encoder.
            question_encoder = self._get_seq2seq_encoder(name="question_{}".format(i))
            # shape: (batch size, question_length, 2*seq2seq hidden size)
            encoded_question = question_encoder(question_embedding)

            # We encode the document embeddings with a seq2seq encoder.
            # Note that this is not the same encoder as used for the question
            document_encoder = self._get_seq2seq_encoder(name="document_{}".format(i))
            # shape: (batch size, document_length, 2*seq2seq hidden size)
            encoded_document = document_encoder(document_embedding)

            # (batch size, document length, question length)
            qd_attention = MaskedBatchDot()([encoded_document, encoded_question])
            # (batch size, document length, question length)
            normalized_qd_attention = MaskedSoftmax()([qd_attention])

            gated_attention_layer = GatedAttention()
            # shape: (batch size, document_length, 2*seq2seq hidden size)
            document_embedding = gated_attention_layer([encoded_document,
                                                        encoded_question,
                                                        normalized_qd_attention])
            gated_attention_dropout = Dropout(self.gated_attention_dropout)
            # shape: (batch size, document_length, 2*seq2seq hidden size)
            document_embedding = gated_attention_dropout([document_embedding])

        # Last Layer
        if self.use_qd_common_feature:
            pass

        # We encode the document embeddings with a final seq2seq encoder.
        document_encoder = self._get_seq2seq_encoder(name="document_final")
        # shape: (batch size, document_length, 2*seq2seq hidden size)
        final_encoded_document = document_encoder(document_embedding)

        if self.cloze_token is None:
            # Get a final encoding of the question from a biGRU that does not return
            # the sequence, and use it to calculate attention over the document.
            final_question_encoder = self._get_encoder(name="question_final")
            # shape: (batch size, 2*seq2seq hidden size)
            final_encoded_question = final_question_encoder(question_embedding)
        else:
            # get a final encoding of the question by concatenating the forward
            # and backward GRU at the index of the cloze token.
            final_question_encoder = self._get_seq2seq_encoder(name="question_final")
            # each are shape (batch size, question_length, seq2seq hidden size)
            encoded_question_f, encoded_question_b = final_question_encoder(question_embedding)
            # extract the gru outputs at the cloze token from the forward and
            # backwards passes
            index_selector = BiGRUIndexSelector(self.cloze_token_index)
            final_encoded_question = index_selector([question_indices,
                                                     encoded_question_f,
                                                     encoded_question_b])

        # take the softmax of the document_embedding after it has been passed
        # through gated attention layers to get document probabilities
        # shape: (batch size, document_length)
        document_probabilities = Attention(name='question_document_softmax')([final_encoded_question,
                                                                              final_encoded_document])
        # We sum together the weights of words that match each option
        # and use the multiword_option_mode to determine how to calculate
        # the total probability of the option.
        options_sum_layer = OptionAttentionSum(self.multiword_option_mode,
                                               name="options_probability_sum")
        # shape: (batch size, num_options)
        options_probabilities = options_sum_layer([document_indices,
                                                   document_probabilities,
                                                   options_indices])

        # We normalize the option_probabilities by dividing each
        # element by L1 norm (sum) of the whole tensor.
        l1_norm_layer = L1Normalize()

        # shape: (batch size, num_options)
        option_normalized_probabilities = l1_norm_layer(options_probabilities)
        return DeepQaModel(input=[question_input, document_input, options_input],
                           output=option_normalized_probabilities)

    @overrides
    def _instance_type(self):
        """
        Return the instance type that the model trains on.
        """
        return WhoDidWhatInstance

    @overrides
    def _get_max_lengths(self) -> Dict[str, int]:
        """
        Return a dictionary with the appropriate padding lengths.
        """
        max_lengths = super(GatedAttentionReader, self)._get_max_lengths()
        max_lengths['num_question_words'] = self.max_question_length
        max_lengths['num_passage_words'] = self.max_passage_length
        max_lengths['num_option_words'] = self.max_option_length
        max_lengths['num_options'] = self.num_options
        return max_lengths

    @overrides
    def _set_max_lengths(self, max_lengths: Dict[str, int]):
        """
        Set the padding lengths of the model.
        """
        # TODO(nelson): superclass complains that there is no
        # word_sequence_length key, so we set it to None here.
        # We should probably patch up / organize the API.
        max_lengths["word_sequence_length"] = None
        super(GatedAttentionReader, self)._set_max_lengths(max_lengths)
        self.max_question_length = max_lengths['num_question_words']
        self.max_passage_length = max_lengths['num_passage_words']
        self.max_option_length = max_lengths['num_option_words']
        self.num_options = max_lengths['num_options']

    @overrides
    def _set_max_lengths_from_model(self):
        self.max_sentence_length = self.model.get_input_shape_at(0)[1]
        # TODO(matt): implement this correctly
