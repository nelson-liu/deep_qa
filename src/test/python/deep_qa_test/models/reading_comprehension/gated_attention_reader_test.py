# pylint: disable=no-self-use,invalid-name
from unittest import TestCase
import os
import shutil
from numpy.testing import assert_allclose

from deep_qa.models.reading_comprehension.gated_attention_reader import GatedAttentionReader
from ...common.constants import TEST_DIR
from ...common.models import get_model, write_who_did_what_files


class TestGatedAttention(TestCase):
    def setUp(self):
        os.makedirs(TEST_DIR, exist_ok=True)
        write_who_did_what_files()

    def tearDown(self):
        shutil.rmtree(TEST_DIR)

    def test_cloze_train_does_not_crash(self):
        args = {
                'save_models': True,
                "qd_common_feature": True,
                "cloze_token": "xxxxx",
                "num_gated_attention_layers": 1,
                "tokenizer": {
                        "type": "words and characters"
                },
                "encoder": {
                        "word": {
                                "type": "bi_gru",
                                "output_dim": 2,
                        }
                },
                "seq2seq_encoder": {
                        "document_final": {
                                "type": "bi_gru",
                                "encoder_params": {
                                        "output_dim": 3
                                },
                                "wrapper_params": {}
                        },
                        "question_final": {
                                "type": "bi_gru",
                                "encoder_params": {
                                        "output_dim": 3
                                },
                                "wrapper_params": {
                                        "merge_mode": None
                                }
                        }
                },
                "embedding_size": 4,
        }
        model = get_model(GatedAttentionReader, args)
        model.train()

        # load the model that we serialized
        loaded_model = get_model(GatedAttentionReader, args)
        loaded_model.load_model()

        # verify that original model and the loaded model predict the same outputs
        assert_allclose(model.model.predict(model.__dict__["validation_input"][0]),
                        loaded_model.model.predict(model.__dict__["validation_input"][0]))

    def test_non_cloze_train_does_not_crash(self):
        args = {
                'save_models': True,
                "qd_common_feature": True,
                "num_gated_attention_layers": 1,
                "tokenizer": {
                        "type": "words and characters"
                },
                "encoder": {
                        "word": {
                                "type": "bi_gru",
                                "output_dim": 2,
                        },
                        "question_final": {
                                "type": "bi_gru",
                                "output_dim": 3
                        }

                },
                "seq2seq_encoder": {
                        "document_final": {
                                "type": "bi_gru",
                                "encoder_params": {
                                        "output_dim": 3
                                },
                                "wrapper_params": {}
                        }
                },
                "embedding_size": 4,
        }
        model = get_model(GatedAttentionReader, args)
        model.train()
        # load the model that we serialized
        loaded_model = get_model(GatedAttentionReader, args)
        loaded_model.load_model()

        # verify that original model and the loaded model predict the same outputs
        assert_allclose(model.model.predict(model.__dict__["validation_input"][0]),
                        loaded_model.model.predict(model.__dict__["validation_input"][0]))
