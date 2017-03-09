from typing import Any, Dict

from overrides import overrides

from ..reading_comprehension.bidirectional_attention import BidirectionalAttentionFlow
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
        - a set of answer options of shape ``(batch_size, num_options, num_answer_words)``

    Output:
        - a probability distribution over the answer options, of shape ``(batch_size, num_options)``

    Parameters
    ----------
    bidaf_params : Dict[str, Any]
        These parameters get passed to a
        :class:`~deep_qa.models.reading_comprehension.bidirectional_attention.BidirectionalAttentionFlow`
        object, which we load.  They should be exactly the same as the parameters used to train the
        saved model.
    """
    # pylint: disable=protected-access
    def __init__(self, params: Dict[str, Any]):
        bidaf_params = params.pop('bidaf_params')
        self.bidaf_model = BidirectionalAttentionFlow(bidaf_params)
        self.bidaf_model.load_model()
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
        return self.bidaf_model.model

    @overrides
    def _instance_type(self):  # pylint: disable=no-self-use
        return self.bidaf_model._instance_type()

    @overrides
    def _get_max_lengths(self) -> Dict[str, int]:
        return self.bidaf_model._get_max_lengths()

    @overrides
    def _set_max_lengths(self, max_lengths: Dict[str, int]):
        self.bidaf_model._set_max_lengths(max_lengths)

    @overrides
    def _set_max_lengths_from_model(self):
        self.bidaf_model._set_max_lengths_from_model()

    @classmethod
    def _get_custom_objects(cls):
        custom_objects = BidirectionalAttentionFlow._get_custom_objects()
        return custom_objects
