from typing import Any, Dict
from overrides import overrides
from keras.layers import Input

from ...data.instances.sentence_selection_instance import SentenceSelectionInstance
from ...layers.attention.attention import Attention
from ...layers.wrappers.encoder_wrapper import EncoderWrapper
from ...training.text_trainer import TextTrainer
from ...training.models import DeepQaModel


class SiameseSentenceSelector(TextTrainer):
    def __init__(self, params: Dict[str, Any]):
        self.max_question_length = params.pop('max_question_length', None)
        self.max_sentence_length = params.pop('max_sentence_length', None)
        self.num_sentences = params.pop('num_sentences', None)
        super(SiameseSentenceSelector, self).__init__(params)

    @overrides
    def _build_model(self):
        """
        The basic outline here is that we'll pass the questions and each
        sentence in the passage through some sort of encoder. At the simplest,
        this could be a biLSTM or a biGRU.

        Then, we take the encoded representation of the question and calculate
        a cosine similarity with the encoded representation of each sentence in
        the passage, to get a tensor of cosine similarities with shape
        (batch_size, num_sentences_per_passage). We then normalize for each
        batch to get a probability distribution over sentences in the passage.
        """
        # First we create input layers and pass the inputs through embedding layers.
        # shape: (batch size, question_length)
        question_input = Input(shape=self._get_sentence_shape(self.max_question_length),
                               dtype='int32', name="question_input")

        # shape: (batch size, num_sentences, sentences_length)
        sentences_input_shape = ((self.num_sentences,) +
                                 self._get_sentence_shape(self.max_sentence_length))
        sentences_input = Input(shape=sentences_input_shape,
                                dtype='int32', name="sentences_input")

        # shape: (batch size, question_length, embedding size)
        question_embedding = self._embed_input(question_input)

        # shape: (batch size, num_sentences, sentences_length, embedding size)
        sentences_embedding = self._embed_input(sentences_input)

        # We encode the question embeddings with some encoder.
        question_encoder = self._get_encoder(name="question_encoder")
        # shape: (batch size, encoder_output_dimension)
        encoded_question = question_encoder(question_embedding)

        # We encode the document embeddings with some encoder.
        sentences_encoder = EncoderWrapper(self._get_encoder(name="sentence_encoder"),
                                           name="TimeDistributed_sentences_encoder")
        # shape: (batch size, num_sentences, encoder_output_dimension)
        encoded_sentences = sentences_encoder(sentences_embedding)

        # Here we use the Attention layer with the cosine similarity function
        # to get the cosine similarities of each sesntence with the question.
        # shape: (batch size, num_sentences)
        attention_name = 'question_sentences_similarity'
        similarity_params = {"type": "cosine_similarity"}
        sentence_probabilities = Attention(name=attention_name,
                                           similarity_function=similarity_params)([encoded_question,
                                                                                   encoded_sentences])

        return DeepQaModel(input=[question_input, sentences_input],
                           output=sentence_probabilities)

    @overrides
    def _instance_type(self):
        """
        Return the instance type that the model trains on.
        """
        return SentenceSelectionInstance

    @overrides
    def _get_max_lengths(self) -> Dict[str, int]:
        """
        Return a dictionary with the appropriate padding lengths.
        """
        max_lengths = super(SiameseSentenceSelector, self)._get_max_lengths()
        max_lengths['num_question_words'] = self.max_question_length
        max_lengths['num_sentences'] = self.num_sentences
        max_lengths['num_sentence_words'] = self.max_sentence_length
        return max_lengths

    @overrides
    def _set_max_lengths(self, max_lengths: Dict[str, int]):
        """
        Set the padding lengths of the model.
        """
        # TODO(nelson): superclass complains that there is no
        # num_sentence_words key, so we set it to None here.
        # We should probably patch up / organize the API.
        super(SiameseSentenceSelector, self)._set_max_lengths(max_lengths)
        self.max_question_length = max_lengths['num_question_words']
        self.num_sentences = max_lengths['num_sentences']
        self.max_sentence_length = max_lengths['num_sentence_words']

    @overrides
    def _set_max_lengths_from_model(self):
        self.set_text_lengths_from_model_input(self.model.get_input_shape_at(0)[1][1:])
        self.max_question_length = self.model.get_input_shape_at(0)[0][1]
        self.num_sentences = self.model.get_input_shape_at(0)[1][1]
        self.max_sentence_length = self.model.get_input_shape_at(0)[1][2]

    @classmethod
    def _get_custom_objects(cls):
        custom_objects = super(SiameseSentenceSelector, cls)._get_custom_objects()
        custom_objects["Attention"] = Attention
        custom_objects["EncoderWrapper"] = EncoderWrapper
        return custom_objects
