# pylint: disable=no-self-use,invalid-name
import numpy
from numpy.testing import assert_allclose
from keras import backend as K

from deep_qa.tensors.backend import cosine_similarity, cumulative_sum
from ..common.test_case import DeepQaTestCase


class TestBackendTensorFunctions(DeepQaTestCase):
    def test_cumulative_sum(self):
        vector = numpy.asarray([1, 2, 3, 4, 5])
        result = K.eval(cumulative_sum(K.variable(vector)))
        assert_allclose(result, [1, 3, 6, 10, 15])

        vector = numpy.asarray([[1, 2, 3, 4, 5],
                                [1, 1, 1, 1, 1],
                                [3, 5, 0, 0, 0]])
        result = K.eval(cumulative_sum(K.variable(vector)))
        assert_allclose(result, [[1, 3, 6, 10, 15],
                                 [1, 2, 3, 4, 5],
                                 [3, 8, 8, 8, 8]])

    def test_cosine_similarity_unbatched(self):
        vector1 = numpy.array([[2, 1, 0, 2, 0, 1, 1, 1]])
        vector2 = numpy.array([[[2, 1, 1, 1, 1, 0, 1, 1],
                                [3, 0, 2, 0, 1, 1, 0, 1]]])
        similarities = K.eval(cosine_similarity(K.variable(vector1),
                                                K.variable(vector2)))
        assert_allclose(similarities,
                        numpy.array([[0.82158381, 0.57735026]]),
                        rtol=1e-6)

    def test_cosine_similarity_batched(self):
        vector1 = numpy.array([[2, 1, 0, 2, 0, 1, 1, 1],
                               [2, 1, 0, 2, 0, 1, 1, 1]])
        vector2 = numpy.array([[[2, 1, 1, 1, 1, 0, 1, 1],
                                [3, 0, 2, 0, 1, 1, 0, 1]],
                               [[2, 1, 1, 1, 1, 0, 1, 1],
                                [3, 0, 2, 0, 1, 1, 0, 1]]])
        similarities = K.eval(cosine_similarity(K.variable(vector1),
                                                K.variable(vector2)))
        assert_allclose(similarities,
                        numpy.array([[0.82158381, 0.57735026],
                                     [0.82158381, 0.57735026]]),
                        rtol=1e-6)
