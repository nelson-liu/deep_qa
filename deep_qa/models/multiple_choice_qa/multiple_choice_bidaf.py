from typing import Any, Dict, List

from keras.layers import Input
from overrides import overrides

from ..reading_comprehension.bidirectional_attention import BidirectionalAttentionFlow
from ...layers.wrappers.time_distributed_with_mask import TimeDistributedWithMask
from ...training.text_trainer import TextTrainer
from ...training.models import DeepQaModel


class MultipleChoiceBidaf(TextTrainer):
    """
    This class extends Minjoon Seo's `Bidirectional Attention Flow model
    <https://www.semanticscholar.org/paper/Bidirectional-Attention-Flow-for-Machine-Seo-Kembhavi/7586b7cca1deba124af80609327395e613a20e9d>`_,
    which was originally applied to predicting spans from a passage, to answering multiple choice
    questions.

    The approach we're going to take here is to load a BiDAF model directly (literally taking all
    of the parameters we need to construct the ``BidirectionalAttentionFlow`` model class),
    applying it to a question and passage, and then adding a few layers on top to try to match the
    predicted span to the answer options we have.

    To match the predicted span to the answer options, we'll first constructed a weighted
    representation of the passage, weighted by the likelihood of each word in the passage being a
    part of the span.  Then we'll compare that representation to a representation for each answer
    option.

    Input:
        - a passage of shape ``(batch_size, num_passage_words)``
        - a question of shape ``(batch_size, num_question_words)``
        - a set of answer options of shape ``(batch_size, num_options, num_option_words)``

    Output:
        - a probability distribution over the answer options, of shape ``(batch_size, num_options)``

    Parameters
    ----------
    bidaf_params : Dict[str, Any]
        These parameters get passed to a
        :class:`~deep_qa.models.reading_comprehension.bidirectional_attention.BidirectionalAttentionFlow`
        object, which we load.  They should be exactly the same as the parameters used to train the
        saved model.
    num_options : int, optional (default=``None``)
        For padding.  How many options should we pad the data to?  If ``None``, this is set from
        the data.
    num_option_words : int, optional (default=``None``)
        For padding.  How many words are in each answer option?  If ``None``, this is set from
        the data.
    """
    # pylint: disable=protected-access
    def __init__(self, params: Dict[str, Any]):
        bidaf_params = params.pop('bidaf_params')
        self._bidaf_model = BidirectionalAttentionFlow(bidaf_params)
        self._bidaf_model.load_model()
        self.num_options = params.pop('num_options', None)
        self.num_option_words = params.pop('num_option_words', None)
        # TODO(matt): copy most of the TextTrainer params over from the bidaf params, so you don't
        # have to duplicate them (and can't mess up the duplication).
        super(MultipleChoiceBidaf, self).__init__(params)

    @overrides
    def _build_model(self):
        """
        Our basic outline here will be to run the BiDAF model on the question and the passage, the
        compute an envelope over the passage for what words BiDAF thought were in the answer span.
        Then we'll weight the BiDAF passage, and use the BiDAF encoder to encode the answer
        options.  Then we'll have a simple similarity function on top to score the similarity
        between each answer option and the predicted answer span.

        Getting the right stuff out of the BiDAF model is a little tricky.  We're going to use the
        same approach as done in :meth:`TextTrainer._build_debug_model
        <deep_qa.training.trainer.TextTrainer._build_debug_model>`: we won't modify the model at
        all, but we'll construct a new model that just changes the outputs to be various layers of
        the original model.
        """
        question_shape = self._get_sentence_shape(self._bidaf_model.num_question_words)
        question_input = Input(shape=question_shape, dtype='int32', name="question_input")
        passage_shape = self._get_sentence_shape(self._bidaf_model.num_passage_words)
        passage_input = Input(shape=passage_shape, dtype='int32', name="passage_input")
        options_shape = (self.num_options,) + self._get_sentence_shape(self.num_option_words)
        options_input = Input(shape=options_shape, dtype='int32', name='options_input')

        bidaf_passage_model = self._get_model_from_bidaf(['question_input', 'passage_input'],
                                                         ['final_merged_passage',
                                                          'span_begin_softmax',
                                                          'span_end_softmax'])
        modeled_passage, span_begin, span_end = bidaf_passage_model([question_input, passage_input])
        # TODO(matt): compute the envelope over the passage, get a weighted passage representation,
        # compare it with the encoded answer options.

        bidaf_question_model = self._get_model_from_bidaf(['question_input'], ['phrase_encoder'])
        encoded_options = TimeDistributedWithMask(bidaf_question_model)(options_input)
        return self._bidaf_model.model

    def _get_model_from_bidaf(self, input_layer_names: List[str], output_layer_names: List[str]):
        """
        Returns a new model constructed from ``self._bidaf_model``.  This model will be a subset of
        BiDAF, with the inputs specified by ``input_layer_names`` and the outputs specified by
        ``output_layer_names``.  For example, you get use this to get a model that outputs the
        passage embedding, just before the span prediction layers, by calling
        ``self._get_model_from_bidaf(['question_input', 'passage_input'], ['final_merged_passage'])``.
        """
        layer_input_dict = {}
        layer_output_dict = {}
        for layer in self._bidaf_model.model.layers:
            layer_input_dict[layer.name] = layer.get_input_at(0)
            layer_output_dict[layer.name] = layer.get_output_at(0)
        input_layers = [layer_input_dict[name] for name in input_layer_names]
        output_layers = [layer_output_dict[name] for name in output_layer_names]
        return DeepQaModel(input=input_layers, output=output_layers)

    @overrides
    def _instance_type(self):  # pylint: disable=no-self-use
        return self._bidaf_model._instance_type()

    @overrides
    def _get_max_lengths(self) -> Dict[str, int]:
        max_lengths = self._bidaf_model._get_max_lengths()
        max_lengths['num_options'] = self.num_options
        max_lengths['num_option_words'] = self.num_option_words
        return max_lengths

    @overrides
    def _set_max_lengths(self, max_lengths: Dict[str, int]):
        self._bidaf_model._set_max_lengths(max_lengths)
        self.num_options = max_lengths['num_options']
        self.num_option_words = max_lengths['num_option_words']

    @overrides
    def _set_max_lengths_from_model(self):
        self._bidaf_model._set_max_lengths_from_model()
        # TODO(matt): finish this

    @classmethod
    def _get_custom_objects(cls):
        custom_objects = BidirectionalAttentionFlow._get_custom_objects()
        return custom_objects
