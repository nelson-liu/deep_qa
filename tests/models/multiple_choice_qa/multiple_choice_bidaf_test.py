# pylint: disable=no-self-use,invalid-name
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
                }
        self.write_who_did_what_files()
        self.ensure_model_trains_and_loads(MultipleChoiceBidaf, args)
