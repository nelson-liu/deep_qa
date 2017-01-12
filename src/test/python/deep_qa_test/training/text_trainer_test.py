# pylint: disable=no-self-use,invalid-name

from unittest import TestCase, mock
import os
import shutil

from deep_qa.solvers.no_memory.true_false_solver import TrueFalseSolver
from ..common.constants import TEST_DIR
from ..common.solvers import get_solver
from ..common.solvers import write_true_false_solver_files
from ..common.test_markers import requires_tensorflow


class TestTextTrainer(TestCase):
    # pylint: disable=protected-access

    def setUp(self):
        os.mkdir(TEST_DIR)
        write_true_false_solver_files()

    def tearDown(self):
        shutil.rmtree(TEST_DIR)

    @mock.patch.object(TrueFalseSolver, '_output_debug_info')
    def test_padding_works_correctly(self, _output_debug_info):
        args = {
                'embedding_size': 2,
                'tokenizer': {'type': 'words and characters'},
                'show_summary_with_masking_info': True,
                'debug': {
                        'data': 'training',
                        'layer_names': [
                                'combined_word_embedding',
                                ],
                        'masks': [
                                'combined_word_embedding',
                                ],
                        }
                }
        solver = get_solver(TrueFalseSolver, args)

        def new_debug(output_dict, epoch):  # pylint: disable=unused-argument
            # We're going to check two things in here: that the shape of combined word embedding is
            # as expected, and that the mask is computed correctly.
            # TODO(matt): actually, from this test, it looks like the mask is returned as
            # output_dict['combined_word_embedding'][1].  Maybe this means we can simplify the
            # logic in Trainer._debug()?  I need to look into this more to be sure that's
            # consistently happening, though.
            word_embeddings = output_dict['combined_word_embedding'][0]
            assert len(word_embeddings) == 6
            assert word_embeddings[0].shape == (3, 4)
            word_masks = output_dict['masks']['combined_word_embedding']
            # Zeros are added to sentences _from the left_.
            assert word_masks[0][0] == 0
            assert word_masks[0][1] == 0
            assert word_masks[0][2] == 1
            assert word_masks[1][0] == 1
            assert word_masks[1][1] == 1
            assert word_masks[1][2] == 1
            assert word_masks[2][0] == 0
            assert word_masks[2][1] == 1
            assert word_masks[2][2] == 1
            assert word_masks[3][0] == 0
            assert word_masks[3][1] == 0
            assert word_masks[3][2] == 1
        _output_debug_info.side_effect = new_debug
        solver.train()

    @requires_tensorflow
    def test_tensorboard_logs_does_not_crash(self):
        solver = get_solver(TrueFalseSolver, {'tensorboard_log': TEST_DIR})
        solver.train()
