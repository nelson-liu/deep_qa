# pylint: disable=no-self-use,invalid-name

import numpy
import keras.backend as K

from deep_qa.common import tensors

class TestTensors:
    def test_masked_batch_dot_masks_properly(self):
        embedding_dim = 3
        a_length = 4
        b_length = 5
        batch_size = 2

        keras_tensor_a = K.placeholder(shape=(batch_size, a_length, embedding_dim))
        keras_tensor_b = K.placeholder(shape=(batch_size, b_length, embedding_dim))
        keras_mask_a = K.placeholder(shape=(batch_size, a_length))
        keras_mask_b = K.placeholder(shape=(batch_size, b_length))
        function = K.function([keras_tensor_a, keras_tensor_b, keras_mask_a, keras_mask_b],
                              [tensors.masked_batch_dot(keras_tensor_a, keras_tensor_b,
                                                        keras_mask_a, keras_mask_b)])

        tensor_a = numpy.random.rand(batch_size, a_length, embedding_dim)
        tensor_b = numpy.random.rand(batch_size, b_length, embedding_dim)
        mask_a = numpy.ones((batch_size, a_length))
        mask_a[1, 3] = 0
        mask_b = numpy.ones((batch_size, b_length))
        mask_b[1, 2] = 0
        result = function([tensor_a, tensor_b, mask_a, mask_b])[0]
        assert numpy.all(result[0, :, :] != numpy.zeros((a_length, b_length)))
        assert numpy.any(result[1, 0, :] != numpy.zeros((b_length)))
        assert numpy.any(result[1, 1, :] != numpy.zeros((b_length)))
        assert numpy.any(result[1, 2, :] != numpy.zeros((b_length)))
        assert numpy.all(result[1, 3, :] == numpy.zeros((b_length)))
        assert numpy.any(result[1, :, 0] != numpy.zeros((a_length)))
        assert numpy.any(result[1, :, 1] != numpy.zeros((a_length)))
        assert numpy.all(result[1, :, 2] == numpy.zeros((a_length)))
        assert numpy.any(result[1, :, 3] != numpy.zeros((a_length)))
        assert numpy.any(result[1, :, 4] != numpy.zeros((a_length)))

        function_no_mask = K.function([keras_tensor_a, keras_tensor_b, keras_mask_a, keras_mask_b],
                                      [tensors.masked_batch_dot(keras_tensor_a, keras_tensor_b, None, None)])
        result = function_no_mask([tensor_a, tensor_b, mask_a, mask_b])[0]
        assert numpy.all(result[0, :, :] != numpy.zeros((a_length, b_length)))
        assert numpy.all(result[1, :, :] != numpy.zeros((a_length, b_length)))

        function_no_mask = K.function([keras_tensor_a, keras_tensor_b, keras_mask_a, keras_mask_b],
                                      [tensors.masked_batch_dot(keras_tensor_a, keras_tensor_b, mask_a, None)])
        result = function_no_mask([tensor_a, tensor_b, mask_a, mask_b])[0]
        assert numpy.all(result[0, :, :] != numpy.zeros((a_length, b_length)))
        assert numpy.any(result[1, 0, :] != numpy.zeros((b_length)))
        assert numpy.any(result[1, 1, :] != numpy.zeros((b_length)))
        assert numpy.any(result[1, 2, :] != numpy.zeros((b_length)))
        assert numpy.all(result[1, 3, :] == numpy.zeros((b_length)))
        assert numpy.any(result[1, :, 0] != numpy.zeros((a_length)))
        assert numpy.any(result[1, :, 1] != numpy.zeros((a_length)))
        assert numpy.any(result[1, :, 2] != numpy.zeros((a_length)))
        assert numpy.any(result[1, :, 3] != numpy.zeros((a_length)))
        assert numpy.any(result[1, :, 4] != numpy.zeros((a_length)))

        function_no_mask = K.function([keras_tensor_a, keras_tensor_b, keras_mask_a, keras_mask_b],
                                      [tensors.masked_batch_dot(keras_tensor_a, keras_tensor_b, None, mask_b)])
        result = function_no_mask([tensor_a, tensor_b, mask_a, mask_b])[0]
        assert numpy.all(result[0, :, :] != numpy.zeros((a_length, b_length)))
        assert numpy.any(result[1, 0, :] != numpy.zeros((b_length)))
        assert numpy.any(result[1, 1, :] != numpy.zeros((b_length)))
        assert numpy.any(result[1, 2, :] != numpy.zeros((b_length)))
        assert numpy.any(result[1, 3, :] != numpy.zeros((b_length)))
        assert numpy.any(result[1, :, 0] != numpy.zeros((a_length)))
        assert numpy.any(result[1, :, 1] != numpy.zeros((a_length)))
        assert numpy.all(result[1, :, 2] == numpy.zeros((a_length)))
        assert numpy.any(result[1, :, 3] != numpy.zeros((a_length)))
        assert numpy.any(result[1, :, 4] != numpy.zeros((a_length)))

    def test_masked_batch_dot_handles_uneven_tensors(self):
        # We're going to test masked_batch_dot with tensors of shape (batch_size, common,
        # a_length, embedding_dim) and (batch_size, common, embedding_dim).  The result should have
        # shape (batch_size, common, a_length).
        embedding_dim = 3
        common_length = 4
        a_length = 5
        batch_size = 2

        keras_tensor_a = K.placeholder(shape=(batch_size, common_length, a_length, embedding_dim))
        keras_tensor_b = K.placeholder(shape=(batch_size, common_length, embedding_dim))
        keras_mask_a = K.placeholder(shape=(batch_size, common_length, a_length))
        keras_mask_b = K.placeholder(shape=(batch_size, common_length))
        function = K.function([keras_tensor_a, keras_tensor_b, keras_mask_a, keras_mask_b],
                              [tensors.masked_batch_dot(keras_tensor_a, keras_tensor_b,
                                                        keras_mask_a, keras_mask_b)])

        tensor_a = numpy.random.rand(batch_size, common_length, a_length, embedding_dim)
        tensor_b = numpy.random.rand(batch_size, common_length, embedding_dim)
        mask_a = numpy.ones((batch_size, common_length, a_length))
        mask_a[1, 1, 3] = 0
        mask_b = numpy.ones((batch_size, common_length))
        mask_b[1, 2] = 0
        result = function([tensor_a, tensor_b, mask_a, mask_b])[0]
        assert numpy.all(result[0, :, :] != numpy.zeros((common_length, a_length)))
        assert numpy.all(result[1, 0, :] != numpy.zeros((a_length)))
        assert result[1, 1, 0] != 0
        assert result[1, 1, 1] != 0
        assert result[1, 1, 2] != 0
        assert result[1, 1, 3] == 0
        assert result[1, 1, 4] != 0
        assert numpy.all(result[1, 2, :] == numpy.zeros((a_length)))
        assert numpy.all(result[1, 3, :] != numpy.zeros((a_length)))

        # We should get the same result if we pass the smaller tensor in first.  Note the subtle
        # difference here - we're only changing the order the tensors gets passed in to
        # masked_batch_dot.
        function = K.function([keras_tensor_a, keras_tensor_b, keras_mask_a, keras_mask_b],
                              [tensors.masked_batch_dot(keras_tensor_b, keras_tensor_a,
                                                        keras_mask_b, keras_mask_a)])
        result = function([tensor_a, tensor_b, mask_a, mask_b])[0]
        assert numpy.all(result[0, :, :] != numpy.zeros((common_length, a_length)))
        assert numpy.all(result[1, 0, :] != numpy.zeros((a_length)))
        assert result[1, 1, 0] != 0
        assert result[1, 1, 1] != 0
        assert result[1, 1, 2] != 0
        assert result[1, 1, 3] == 0
        assert result[1, 1, 4] != 0
        assert numpy.all(result[1, 2, :] == numpy.zeros((a_length)))
        assert numpy.all(result[1, 3, :] != numpy.zeros((a_length)))
