# pylint: disable=no-self-use,invalid-name
from unittest import TestCase
import os
import shutil

from deep_qa.models.reading_comprehension.gated_attention_reader import GatedAttentionReader
from ...common.constants import TEST_DIR
from ...common.models import get_model, write_who_did_what_files


class TestGatedAttention(TestCase):
    def setUp(self):
        os.makedirs(TEST_DIR, exist_ok=True)
        write_who_did_what_files()

    def tearDown(self):
        shutil.rmtree(TEST_DIR)

    def test_train_does_not_crash(self):
        args = {
                "num_gated_attention_layers": 1,
                "encoder": {
                        "type": "bi_gru",
                        "output_dim": 6
                },
                "seq2seq_encoder": {
                        "type": "bi_gru",
                        "encoder_params": {
                                "output_dim": 6
                        },
                        "wrapper_params": {}
                },
                "embedding_size": 5,
        }
        solver = get_model(GatedAttentionReader, args)
        solver.train()
