# pylint: disable=no-self-use,invalid-name
import numpy
from numpy.testing import assert_almost_equal
import keras.backend as K
from keras.layers import Input
from keras.models import Model

from deep_qa.layers.backend.masked_batch_dot import MaskedBatchDot

class TestMaskedBatchDotLayer:
    def test_call_works_on_simple_input(self):
        batch_size = 2
        tensor_a = Input(shape=(3, 2), dtype='float32')
        tensor_b = Input(shape=(4, 2), dtype='float32')
        masked_batch_dot = MaskedBatchDot()([tensor_a, tensor_b])
        model = Model(input=[tensor_a, tensor_b], output=[masked_batch_dot])

        tensor_a = numpy.array([[[0.3, 0.1], [0.4, 0.2], [0.8, 0.1]],
                                [[0.3, 0.1], [0.4, 0.2], [0.8, 0.1]]])
        tensor_b = numpy.array([[[0.2, 0.6], [0.4, 0.3], [0.5, 0.7], [0.1, .6]],
                                [[0.2, 0.6], [0.4, 0.3], [0.5, 0.7], [0.1, .6]]])
        a_dot_b = model.predict([tensor_a, tensor_b])
        assert a_dot_b.shape == (batch_size, 3, 4)
        assert_almost_equal(a_dot_b, numpy.array([[[0.12, 0.15, 0.22, 0.09],
                                                   [0.2, 0.22, 0.34, 0.16],
                                                   [0.22, 0.35, 0.47, 0.14]],
                                                  [[0.12, 0.15, 0.22, 0.09],
                                                   [0.2, 0.22, 0.34, 0.16],
                                                   [0.22, 0.35, 0.47, 0.14]]]))

    def test_call_works_on_masked_input(self):
        tensor_a = K.variable(numpy.array([[[0.3, 0.1], [0.4, 0.2], [0.8, 0.1]],
                                           [[0.3, 0.1], [0.4, 0.2], [0.8, 0.1]]]))
        print(tensor_a)
        mask_a = K.variable(numpy.array([[1, 0, 1], [1, 1, 0]]))
        print(mask_a)
        tensor_b = K.variable(numpy.array([[[0.2, 0.6], [0.4, 0.3], [0.5, 0.7], [0.1, .6]],
                                           [[0.2, 0.6], [0.4, 0.3], [0.5, 0.7], [0.1, .6]]]))
        mask_b = K.variable(numpy.array([[0, 1, 1, 1], [1, 0, 1, 1]]))
        masked_a_dot_b = K.eval(MaskedBatchDot()([tensor_a, tensor_b],
                                                 [mask_a, mask_b]))
        assert_almost_equal(masked_a_dot_b, numpy.array([[[0.0, 0.15, 0.22, 0.09],
                                                          [0.0, 0.0, 0.0, 0.0],
                                                          [0.0, 0.35, 0.47, 0.14]],
                                                         [[0.12, 0.0, 0.22, 0.09],
                                                          [0.2, 0.0, 0.34, 0.16],
                                                          [0.0, 0.0, 0.0, 0.0]]]))

    def test_get_output_shape_for(self):
        # TODO(nelson): verify the correctness of this, esp. in rank 1 edge case.
        mbd = MaskedBatchDot()
        assert mbd.get_output_shape_for([(5, 10), (5, 10)]) == (5, 1)
        assert mbd.get_output_shape_for([(1, 1, 1), (1, 1, 1)]) == (1, 1, 1)
        assert mbd.get_output_shape_for([(1, 5, 3), (1, 2, 3)]) == (1, 5, 2)
        assert mbd.get_output_shape_for([(1, 5, 4, 3), (1, 2, 3)]) == (1, 5, 4, 2)
        assert mbd.get_output_shape_for([(1, 5, 4), (1, 2, 3, 4)]) == (1, 5, 2, 3)
