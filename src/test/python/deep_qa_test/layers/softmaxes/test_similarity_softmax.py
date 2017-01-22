from unittest import TestCase

import keras.backend as K
import numpy as np
from numpy.testing import assert_array_almost_equal
from keras.layers import Input
from keras.models import Model
from deep_qa.layers.softmaxes.similarity_softmax import SimilaritySoftmax


class TestSimilaritySoftmax(TestCase):
    # pylint: disable=no-self-use
    def test_no_mask(self):
        vector_length = 3
        matrix_num_rows = 2

        vector_input = Input(shape=(vector_length,),
                             dtype='float32',
                             name="vector_input")
        matrix_input = Input(shape=(matrix_num_rows, vector_length),
                             dtype='float32',
                             name="matrix_input")
        similarity_softmax = SimilaritySoftmax()([vector_input, matrix_input])
        model = Model([vector_input, matrix_input],
                      similarity_softmax)

        # Testing general non-batched case.
        vector = np.array([[0.3, 0.1, 0.5]])
        matrix = np.array([[[0.6, 0.8, 0.1], [0.15, 0.5, 0.2]]])

        result = model.predict([vector, matrix])
        assert_array_almost_equal(result, np.array([[0.52871835, 0.47128162]]))

        # Testing non-batched case where inputs are all 0s.
        vector = np.array([[0, 0, 0]])
        matrix = np.array([[[0, 0, 0], [0, 0, 0]]])

        result = model.predict([vector, matrix])
        assert_array_almost_equal(result, np.array([[0.5, 0.5]]))

    def test_masked(self):
        # Testing general masked non-batched case.
        vector = K.variable(np.array([[0.3, 0.1, 0.5]]))
        matrix = K.variable(np.array([[[0.6, 0.8, 0.1], [0.15, 0.5, 0.2], [0.1, 0.4, 0.3]]]))
        mask = K.variable(np.array([[1.0, 0.0, 1.0]]))
        result = K.eval(SimilaritySoftmax().call([vector, matrix], mask=["_", mask]))
        assert_array_almost_equal(result, np.array([[0.52248482, 0.0, 0.47751518]]))

    def test_batched_no_mask(self):
        vector_length = 3
        matrix_num_rows = 2

        vector_input = Input(shape=(vector_length,),
                             dtype='float32',
                             name="vector_input")
        matrix_input = Input(shape=(matrix_num_rows, vector_length),
                             dtype='float32',
                             name="matrix_input")
        similarity_softmax = SimilaritySoftmax()([vector_input, matrix_input])
        model = Model([vector_input, matrix_input],
                      similarity_softmax)

        # Testing general batched case.
        vector = np.array([[0.3, 0.1, 0.5], [0.3, 0.1, 0.5]])
        matrix = np.array([[[0.6, 0.8, 0.1], [0.15, 0.5, 0.2]],
                           [[0.6, 0.8, 0.1], [0.15, 0.5, 0.2]]])

        result = model.predict([vector, matrix])
        assert_array_almost_equal(result, np.array([[0.52871835, 0.47128162],
                                                    [0.52871835, 0.47128162]]))

    def test_batched_masked(self):
        # Testing general masked non-batched case.
        vector = K.variable(np.array([[0.3, 0.1, 0.5], [0.3, 0.1, 0.5]]))
        matrix = K.variable(np.array([[[0.6, 0.8, 0.1], [0.15, 0.5, 0.2], [0.5, 0.3, 0.2]],
                                      [[0.6, 0.8, 0.1], [0.15, 0.5, 0.2], [0.5, 0.3, 0.2]]]))
        mask = K.variable(np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]]))
        result = K.eval(SimilaritySoftmax().call([vector, matrix], mask=["_", mask]))
        assert_array_almost_equal(result, np.array([[0.52871835, 0.47128162, 0.0],
                                                    [0.50749944, 0.0, 0.49250056]]))

        # Test the case where a mask is all 0s and an input is all 0s.
        vector = K.variable(np.array([[0.0, 0.0, 0.0], [0.3, 0.1, 0.5]]))
        matrix = K.variable(np.array([[[0.6, 0.8, 0.1], [0.15, 0.5, 0.2], [0.5, 0.3, 0.2]],
                                      [[0.6, 0.8, 0.1], [0.15, 0.5, 0.2], [0.5, 0.3, 0.2]]]))
        mask = K.variable(np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]))
        result = K.eval(SimilaritySoftmax().call([vector, matrix], mask=["_", mask]))
        assert_array_almost_equal(result, np.array([[0.5, 0.5, 0.0],
                                                    [0.0, 0.0, 0.0]]))
