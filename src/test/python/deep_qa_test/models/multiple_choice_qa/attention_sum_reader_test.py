# pylint: disable=no-self-use,invalid-name
from unittest import TestCase
import os
import shutil

from deep_qa.models.multiple_choice_qa.attention_sum_reader import AttentionSumReader
from ...common.constants import TEST_DIR
from ...common.models import get_model, write_who_did_what_files


class TestAttentionSumReader(TestCase):
    def setUp(self):
        os.makedirs(TEST_DIR, exist_ok=True)
        write_who_did_what_files()

    def tearDown(self):
        shutil.rmtree(TEST_DIR)

    def test_train_does_not_crash(self):
        args = {
                'encoder': {"type": "bi_gru"},
        }
        solver = get_model(AttentionSumReader, args)
        solver.train()
