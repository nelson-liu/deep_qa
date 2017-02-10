# pylint: disable=no-self-use,invalid-name
import numpy
from numpy.testing import assert_almost_equal
import keras.backend as K

from deep_qa.layers.backend.batch_dot import BatchDot


class TestMaskedBatchDotLayer:
    def test_compute_mask(self):
        tensor_a = K.variable(numpy.array([[[0.3, 0.1], [0.4, 0.2], [0.8, 0.1]],
                                           [[0.3, 0.1], [0.4, 0.2], [0.8, 0.1]]]))
        mask_a = K.variable(numpy.array([[1, 0, 1], [1, 1, 0]]))
        tensor_b = K.variable(numpy.array([[[0.2, 0.6], [0.4, 0.3], [0.5, 0.7], [0.1, .6]],
                                           [[0.2, 0.6], [0.4, 0.3], [0.5, 0.7], [0.1, .6]]]))
        mask_b = K.variable(numpy.array([[0, 1, 1, 1], [1, 0, 1, 1]]))
        calculated_mask = K.eval(BatchDot().compute_mask([tensor_a, tensor_b],
                                                         [mask_a, mask_b]))
        assert_almost_equal(calculated_mask, numpy.array([[[0.0, 1.0, 1.0, 1.0],
                                                           [0.0, 0.0, 0.0, 0.0],
                                                           [0.0, 1.0, 1.0, 1.0]],
                                                          [[1.0, 0.0, 1.0, 1.0],
                                                           [1.0, 0.0, 1.0, 1.0],
                                                           [0.0, 0.0, 0.0, 0.0]]]))

    def test_get_output_shape_for(self):
        mbd = BatchDot()
        assert mbd.get_output_shape_for([(5, 10), (5, 10)]) == (5, 1)
        assert mbd.get_output_shape_for([(1, 1, 1), (1, 1, 1)]) == (1, 1, 1)
        assert mbd.get_output_shape_for([(1, 5, 3), (1, 2, 3)]) == (1, 5, 2)
        assert mbd.get_output_shape_for([(1, 5, 4, 3), (1, 2, 3)]) == (1, 5, 4, 2)
        assert mbd.get_output_shape_for([(1, 5, 4), (1, 2, 3, 4)]) == (1, 5, 2, 3)
