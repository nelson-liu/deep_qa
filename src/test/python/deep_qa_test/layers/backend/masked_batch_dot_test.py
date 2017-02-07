# pylint: disable=no-self-use,invalid-name
import numpy
from numpy.testing import assert_almost_equal
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
        adotb = model.predict([tensor_a, tensor_b])
        assert adotb.shape == (batch_size, 3, 4)
        assert_almost_equal(adotb, numpy.array([[[ 0.12, 0.15000001, 0.22, 0.09],
                                                 [ 0.20000002, 0.22000001, 0.34, 0.16000001],
                                                 [ 0.22000001, 0.35000002, 0.47, 0.14000002]],
                                                [[ 0.12, 0.15000001, 0.22, 0.09],
                                                 [ 0.20000002, 0.22000001, 0.34, 0.16000001],
                                                 [ 0.22000001, 0.35000002, 0.47, 0.14000002]]]))

    def test_get_output_shape_for(self):
        # TODO(nelson): verify the correctness of this, esp. in rank 1 edge case.
        mbd = MaskedBatchDot()
        assert mbd.get_output_shape_for([(1, 1, 1), (1, 1, 1)]) == (1, 1, 1)
        assert mbd.get_output_shape_for([(1, 5, 3), (1, 2, 3)]) == (1, 5, 2)
        assert mbd.get_output_shape_for([(1, 5, 4, 3), (1, 2, 3)]) == (1, 5, 4, 2)
        assert mbd.get_output_shape_for([(1, 5, 4), (1, 2, 3, 4)]) == (1, 5, 2, 3)
