from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import keras.backend as K
from keras.layers import Input
from keras.models import Model
from deep_qa.layers.option_attention_sum import OptionAttentionSum


class TestOptionAttentionSum(TestCase):
    # pylint: disable=no-self-use
    def test_mean_mode(self):
        document_probabilities_length = 6
        document_indicies_length = document_probabilities_length
        max_num_options = 3
        max_num_words_per_option = 2

        document_indicies_input = Input(shape=(document_indicies_length,),
                                        dtype='int32',
                                        name="document_indicies_input")
        document_probabilities_input = Input(shape=(document_probabilities_length,),
                                             dtype='float32',
                                             name="document_probabilities_input")
        options_input = Input(shape=(max_num_options, max_num_words_per_option),
                              dtype='int32', name="options_input")
        option_attention_sum_mean = OptionAttentionSum()([document_indicies_input,
                                                          document_probabilities_input,
                                                          options_input])
        model = Model([document_indicies_input,
                       document_probabilities_input,
                       options_input],
                      option_attention_sum_mean)

        document_indicies = np.array([[1, 2, 3, 4, 1, 2]])
        document_probabilities = np.array([[.1, .2, .3, .4, 0.01, 0.03]])

        options = np.array([[[1, 2], [3, 4], [1, 2]]])
        result = model.predict([document_indicies, document_probabilities, options])
        assert_array_almost_equal(result, np.array([[0.17, 0.35, 0.17]]))

        options = np.array([[[1, 1], [3, 1], [4, 2]]])
        result = model.predict([document_indicies, document_probabilities, options])
        assert_array_almost_equal(result, np.array([[0.11, 0.205, 0.315]]))

    def test_sum_mode(self):
        document_probabilities_length = 6
        document_indicies_length = document_probabilities_length
        max_num_options = 3
        max_num_words_per_option = 2

        document_indicies_input = Input(shape=(document_indicies_length,),
                                        dtype='int32',
                                        name="document_indicies_input")
        document_probabilities_input = Input(shape=(document_probabilities_length,),
                                             dtype='float32',
                                             name="document_probabilities_input")
        options_input = Input(shape=(max_num_options, max_num_words_per_option),
                              dtype='int32', name="options_input")
        option_attention_sum_mean = OptionAttentionSum("sum")([document_indicies_input,
                                                               document_probabilities_input,
                                                               options_input])
        model = Model([document_indicies_input,
                       document_probabilities_input,
                       options_input],
                      option_attention_sum_mean)

        document_indicies = np.array([[1, 2, 3, 4, 1, 2]])
        document_probabilities = np.array([[.1, .2, .3, .4, 0.01, 0.03]])

        options = np.array([[[1, 2], [3, 4], [1, 2]]])
        result = model.predict([document_indicies, document_probabilities, options])
        assert_array_almost_equal(result, np.array([[0.34, 0.70, 0.34]]))

        options = np.array([[[1, 1], [3, 1], [4, 2]]])
        result = model.predict([document_indicies, document_probabilities, options])
        assert_array_almost_equal(result, np.array([[0.22, 0.41, 0.63]]))

    def test_multiword_option_mode_validation(self):
        self.assertRaises(ValueError, OptionAttentionSum, "summean")

    def test_compute_mask(self):
        option_attention_sum = OptionAttentionSum()
        result = option_attention_sum.compute_mask(["_", "_",
                                                    K.variable(np.array([[[1, 2, 0], [2, 3, 3],
                                                                          [0, 0, 0], [0, 0, 0]]],
                                                                        dtype="int32"))])
        assert_array_equal(K.eval(result), np.array([[1, 1, 0, 0]]))
        result = option_attention_sum.compute_mask(["_", "_",
                                                    K.variable(np.array([[[1, 2, 0], [1, 0, 0],
                                                                          [0, 0, 0], [0, 0, 0]]],
                                                                        dtype="int32"))])
        assert_array_equal(K.eval(result), np.array([[1, 1, 0, 0]]))
        result = option_attention_sum.compute_mask(["_", "_",
                                                    K.variable(np.array([[[1, 2, 0], [0, 0, 0],
                                                                          [0, 0, 0], [0, 0, 0]]],
                                                                        dtype="int32"))])
        assert_array_equal(K.eval(result), np.array([[1, 0, 0, 0]]))
