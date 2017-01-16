from typing import Any, Dict
from overrides import overrides
from keras.layers import Input

from ...data.instances.whodidwhat_instance import WhoDidWhatInstance
from ...layers.softmaxes.similarity_softmax import SimilaritySoftmax
from ...layers.option_attention_sum import OptionAttentionSum
from ...layers.l1_normalize import L1Normalize
from ...training.text_trainer import TextTrainer
from ...training.models import DeepQaModel


class AttentionSumReader(TextTrainer):
    """
    This TextTrainer implements the Attention Sum Reader model described by
    Kadlec et. al 2016. It takes a question and document as input, encodes the
    document and question words with two separate Bidirectional GRUs, and then
    takes the dotproduct of the question embedding with the document embedding
    of each word in the document. This create an attention over words in the
    document, and it then selects the word with the highest summed weight as
    the answer.
    """
    def __init__(self, params: Dict[str, Any]):
        self.max_question_length = params.pop('max_question_length', None)
        self.max_passage_length = params.pop('max_passage_length', None)
        self.max_option_length = params.pop('max_option_length', None)
        self.num_options = params.pop('num_options', None)
        # either "mean" or "sum"
        self.multiword_option_mode = params.pop('multiword_option_mode', "mean")
        super(AttentionSumReader, self).__init__(params)

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
        # First we create input layers and pass the inputs through embedding layers.

        # ? = batch size
        # shape: (?, max question length in words), say (?, 8)
        question_input = Input(shape=self._get_sentence_shape(self.max_question_length),
                               dtype='int32', name="question_input")
        # shape: (?, max document length in words), say (?, 7)
        document_input = Input(shape=self._get_sentence_shape(self.max_passage_length),
                               dtype='int32',
                               name="document_input")
        # shape: (?, max number of options, max number of words in option), say (?, 3, 2)
        options_input = Input(shape=(self.num_options,) + self._get_sentence_shape(self.max_option_length),
                              dtype='int32', name="options_input")
        # shape: (?, max question length in words, embedding size), (?, 8, 5)
        question_embedding = self._embed_input(question_input)

        # shape: (?, max document length in words, embedding size), (?, 7, 5)
        document_embedding = self._embed_input(document_input)

        # Then we encode the question embeddings with some encoder.
        question_encoder = self._get_sentence_encoder()
        # shape: (?, 2xembed size), (?, 10)
        encoded_question = question_encoder(question_embedding)

        # encode the document with some seq2seq encoder
        seq2seq_input_shape = (self._get_sentence_shape(self.max_passage_length) + (self.embedding_size,))
        document_encoder = self._get_seq2seq_encoder(input_shape=seq2seq_input_shape)
        # shape: (?, ?, 2xembed size), (?, ?, 10) should be (?, max_document_length, 10)
        encoded_document = document_encoder(document_embedding)

        # take the dotproduct of `encoded_question` and each word in `encoded_document`
        # shape: (?, max_document_length) or (?, 7)
        document_probabilities = SimilaritySoftmax(name='question_document_softmax')([encoded_question,
                                                                                      encoded_document])
        # sum together the weights of words that match each option
        options_sum_layer = OptionAttentionSum(self.multiword_option_mode,
                                               name="options_probability_sum")
        # shape should be (?, max_number_of_options), or (?, 3) in this case
        options_probabilities = options_sum_layer([document_input,
                                                   document_probabilities, options_input])
        # normalize the option_probabilities by dividing each
        # element by L1 norm (sum) of vector.
        l1_norm_layer = L1Normalize()

        # shape should be (?, max_number_of_options), or (?, 3) in this case
        option_normalized_probabilities = l1_norm_layer(options_probabilities)
        return DeepQaModel(input=[question_input, document_input, options_input],
                           output=option_normalized_probabilities)

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
        # set the ones that i add myself, and do the same for _set_max_lengths
        max_lengths = super(AttentionSumReader, self)._get_max_lengths()
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
        # superclass complains that there is no word_sequence_length
        # key, should probably patch up API
        max_lengths["word_sequence_length"] = None
        super(AttentionSumReader, self)._set_max_lengths(max_lengths)
        self.max_question_length = max_lengths['num_question_words']
        self.max_passage_length = max_lengths['num_passage_words']
        self.max_option_length = max_lengths['num_option_words']
        self.num_options = max_lengths['num_options']

    @overrides
    def _set_max_lengths_from_model(self):
        self.max_sentence_length = self.model.get_input_shape_at(0)[1]
        # TODO(matt): implement this correctly
