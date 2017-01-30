from copy import deepcopy
from typing import Any, Dict
from overrides import overrides
from keras.layers import Input, Dropout

from ...data.instances.whodidwhat_instance import WhoDidWhatInstance
from ...common.checks import ConfigurationError
from ...layers.attention.masked_softmax import MaskedSoftmax
from ...layers.attention.attention import Attention
from ...layers.option_attention_sum import OptionAttentionSum
from ...layers.attention.gated_attention import GatedAttention
from ...layers.l1_normalize import L1Normalize
from ...training.text_trainer import TextTrainer
from ...training.models import DeepQaModel


class GatedAttentionReader(TextTrainer):
    """
    This TextTrainer implements the Gated Attention Reader model described by
    Dhingra et. al 2016.
    """
    def __init__(self, params: Dict[str, Any]):
        self.max_question_length = params.pop('max_question_length', None)
        self.max_passage_length = params.pop('max_passage_length', None)
        self.max_option_length = params.pop('max_option_length', None)
        self.num_options = params.pop('num_options', None)
        # either "mean" or "sum"
        self.multiword_option_mode = params.pop('multiword_option_mode', "mean")
        self.num_gated_attention_layers = params.pop('num_gated_attention_layers', 4)
        self.gated_attention_dropout = params.pop('gated_attention_dropout', 0.3)
        super(GatedAttentionReader, self).__init__(params)

    @overrides
    def _build_model(self):
        """
        The basic outline here is that we'll pass the questions and the
        document / passage (think of this as a collection of possible answer
        choices) into a word embedding layer.

        Then, we run the word embeddings from the document (a sequence) through
        a bidirectional GRU and output a sequence that is the same length as
        the input sequence size. For each time step, the output item
        ("contextual embedding") is the concatenation of the forward and
        backward hidden states in the bidirectional GRU encoder at that time
        step.

        To get the encoded question, we pass the words of the question into
        another bidirectional GRU. This time, the output encoding is a vector
        containing the concatenation of the last hidden state in the forward
        network with the last hidden state of the backward network.

        We then take the dot product of the question embedding with each of the
        contextual embeddings for the words in the documents. We sum up all the
        occurences of a word ("total attention"), and pick the word with the
        highest total attention in the document as the answer.
        """
        # First we create input layers and pass the question and document
        # through embedding layers.

        # shape: (batch size, question_length)
        question_input = Input(shape=self._get_sentence_shape(self.max_question_length),
                               dtype='int32', name="question_input")
        # shape: (batch size, document_length)
        document_input = Input(shape=self._get_sentence_shape(self.max_passage_length),
                               dtype='int32',
                               name="document_input")
        # shape: (batch size, max number of options, num_options)
        options_input = Input(shape=(self.num_options,) + self._get_sentence_shape(self.max_option_length),
                              dtype='int32', name="options_input")

        # shape: (batch size, question_length, embedding size)
        question_embedding = self._embed_input(question_input)

        # shape: (batch size, document_length, embedding size)
        document_embedding = self._embed_input(document_input)

        # We pass the question and document embedding through a variable
        # number of gated-attention layers.
        if self.num_gated_attention_layers < 1:
            raise ConfigurationError("Need at least one gated attention layer.")
        for i in range(self.num_gated_attention_layers):
            # We encode the question embeddings with a seq2seq encoder.
            question_encoder = self._get_new_seq2seq_encoder(self._get_seq2seq_params(),
                                                             name="question_seq2seq_{}".format(i))
            # shape: (batch size, question_length, 2*seq2seq_hidden_size)
            encoded_question = question_encoder(question_embedding)

            # We encode the document embeddings with a seq2seq encoder.
            # Note that this is not the same encoder as used for the question
            document_encoder = self._get_new_seq2seq_encoder(self._get_seq2seq_params(),
                                                             name="document_seq2seq_{}".format(i))
            # shape: (batch size, document_length, 2*seq2seq_hidden_size)
            encoded_document = document_encoder(document_embedding)

            gated_attention_layer = GatedAttention()
            # shape: (batch size, document_length, 2*seq2seq_hidden_state)
            # Note that the size of the last dimension of the input to the next layer
            # is not necessarily the embedding size.
            document_embedding = gated_attention_layer([question_embedding, document_embedding])
            gated_attention_dropout = Dropout(self.gated_attention_dropout)
            document_embedding = gated_attention_dropout([document_embedding])

        # take the softmax of the document_embedding after it has been passed
        # through gated attention layers to get document probabilities
        # shape: (batch size, max document length in words)
        document_probabilities = MaskedSoftmax()[document_embedding]

        # We sum together the weights of words that match each option
        # and use the multiword_option_mode to determine how to calculate
        # the total probability of the option.
        options_sum_layer = OptionAttentionSum(self.multiword_option_mode,
                                               name="options_probability_sum")
        # shape: (batch size, num_options)
        options_probabilities = options_sum_layer([document_input,
                                                   document_probabilities, options_input])
        # We normalize the option_probabilities by dividing each
        # element by L1 norm (sum) of the whole tensor.
        l1_norm_layer = L1Normalize()

        # shape: (batch size, num_options)
        option_normalized_probabilities = l1_norm_layer(options_probabilities)
        return DeepQaModel(input=[question_input, document_input, options_input],
                           output=option_normalized_probabilities)

    def _get_seq2seq_params(self):
        params = deepcopy(self.seq2seq_encoder_params)
        return params

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
