# pylint: disable=no-self-use,invalid-name
from keras import backend as K

from deep_qa.models.multiple_choice_qa.multiple_choice_bidaf import MultipleChoiceBidaf
from deep_qa.models.reading_comprehension.bidirectional_attention import BidirectionalAttentionFlow
from ...common.test_case import DeepQaTestCase


class TestMultipleChoiceBidaf(DeepQaTestCase):
    def test_trains_and_loads_correctly(self):
        self.write_span_prediction_files()
        args = {
                'model_serialization_prefix': self.TEST_DIR + "_bidaf",
                'embedding_size': 4,
                'save_models': True,
                'tokenizer': {'type': 'words and characters'},
                'show_summary_with_masking_info': True,
                }
        bidaf_model = self.get_model(BidirectionalAttentionFlow, args)
        bidaf_model.train()

        bidaf_model_params = self.get_model_params(BidirectionalAttentionFlow, args)
        args = {
                'bidaf_params': bidaf_model_params,
                'train_bidaf': False,
                'similarity_function': {'type': 'linear', 'combination': 'x,y'},
                }
        self.write_who_did_what_files()
        model, _ = self.ensure_model_trains_and_loads(MultipleChoiceBidaf, args)
        # All of the params come from the linear similarity function in the attention layer,
        # because we set `train_bidaf` to `False`.
        assert sum([K.count_params(p) for p in model.model.trainable_weights]) == 41
