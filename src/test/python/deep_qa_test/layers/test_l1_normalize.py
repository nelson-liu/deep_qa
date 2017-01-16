from unittest import TestCase
import numpy as np
from numpy.testing import assert_array_almost_equal

from keras.layers import Input
from keras.models import Model
from deep_qa.layers.l1_normalize import L1Normalize


class TestL1Normalize(TestCase):
    # pylint: disable=no-self-use
    def test_general_case(self):
        # TODO write tests for mask case
        input_length = 6

        input_layer = Input(shape=(input_length,), dtype='float32', name="input")
        l1_normalize_layer = L1Normalize()
        normalized_input = l1_normalize_layer(input_layer)

        model = Model([input_layer], normalized_input)
        unnormalized_vector = np.array([[.1, .2, .3, .4, 0.01, 0.03]])
        result = model.predict([unnormalized_vector])
        assert_array_almost_equal(result, np.array([[0.09615385, 0.1923077,
                                                     0.28846157, 0.38461539,
                                                     0.00961538, 0.02884615]]))
        assert_array_almost_equal(np.sum(result, axis=1), np.ones(1))

    def test_squeeze_case(self):
        # TODO write tests for mask case
        input_length = 6

        input_layer = Input(shape=(input_length, 1), dtype='float32', name="input")
        l1_normalize_layer = L1Normalize()
        normalized_input = l1_normalize_layer(input_layer)

        model = Model([input_layer], normalized_input)
        unnormalized_vector = np.array([[[.1], [.2], [.3], [.4], [0.01], [0.03]]])
        result = model.predict([unnormalized_vector])
        assert_array_almost_equal(result, np.array([[0.09615385, 0.1923077,
                                                     0.28846157, 0.38461539,
                                                     0.00961538, 0.02884615]]))
        assert_array_almost_equal(np.sum(result, axis=1), np.ones(1))
