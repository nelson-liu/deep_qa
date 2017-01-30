# pylint: disable=no-self-use,invalid-name

import numpy
from numpy.testing import assert_almost_equal
from keras.layers import Input
from keras.models import Model
from deep_qa.layers.attention.gated_attention import GatedAttention


class TestGatedAttentionLayer:
    def test_multiplication(self):
        document_len = 3
        question_len = 4
        bigru_hidden_dim = 2

        document_input = Input(shape=(document_len, bigru_hidden_dim,),
                               dtype='float32',
                               name="document_input")
        question_input = Input(shape=(question_len, bigru_hidden_dim,),
                               dtype='float32',
                               name="question_input")
        attention_input = Input(shape=(document_len, question_len,),
                                dtype='float32',
                                name="attention_input")

        gated_attention = GatedAttention()([document_input, question_input,
                                            attention_input])
        model = Model([document_input, question_input, attention_input],
                      gated_attention)

        # Testing general non-batched case.
        document = numpy.array([[[0.3, 0.1], [0.4, 0.2], [0.8, 0.1]]])
        question = numpy.array([[[0.2, 0.6], [0.4, 0.3], [0.5, 0.7], [0.1, .6]]])
        attention = numpy.array([[[0.3, 0.1, 0.5, 0.2],
                                  [0.4, 0.2, 0.8, 0.7],
                                  [0.8, 0.1, 0.6, 0.4]]])
        result = model.predict([document, question, attention])
        assert_almost_equal(result, numpy.array([[[0.111, 0.068],
                                                  [0.252, 0.25599999],
                                                  [0.43200002, 0.11700001]]]))

    def test_addition(self):
        document_len = 3
        question_len = 4
        bigru_hidden_dim = 2

        document_input = Input(shape=(document_len, bigru_hidden_dim,),
                               dtype='float32',
                               name="document_input")
        question_input = Input(shape=(question_len, bigru_hidden_dim,),
                               dtype='float32',
                               name="question_input")
        attention_input = Input(shape=(document_len, question_len,),
                                dtype='float32',
                                name="attention_input")

        gated_attention = GatedAttention(gating_function="+")([document_input, question_input,
                                                               attention_input])
        model = Model([document_input, question_input, attention_input],
                      gated_attention)

        # Testing general non-batched case.
        document = numpy.array([[[0.3, 0.1], [0.4, 0.2], [0.8, 0.1]]])
        question = numpy.array([[[0.2, 0.6], [0.4, 0.3], [0.5, 0.7], [0.1, .6]]])
        attention = numpy.array([[[0.3, 0.1, 0.5, 0.2],
                                  [0.4, 0.2, 0.8, 0.7],
                                  [0.8, 0.1, 0.6, 0.4]]])
        result = model.predict([document, question, attention])
        assert_almost_equal(result, numpy.array([[[0.67, 0.78000001],
                                                  [1.03, 1.47999997],
                                                  [1.34000002, 1.27000008]]]))

    def test_concatenation(self):
        document_len = 3
        question_len = 4
        bigru_hidden_dim = 2

        document_input = Input(shape=(document_len, bigru_hidden_dim,),
                               dtype='float32',
                               name="document_input")
        question_input = Input(shape=(question_len, bigru_hidden_dim,),
                               dtype='float32',
                               name="question_input")
        attention_input = Input(shape=(document_len, question_len,),
                                dtype='float32',
                                name="attention_input")

        gated_attention = GatedAttention(gating_function="||")([document_input, question_input,
                                                                attention_input])
        model = Model([document_input, question_input, attention_input],
                      gated_attention)

        # Testing general non-batched case.
        document = numpy.array([[[0.3, 0.1], [0.4, 0.2], [0.8, 0.1]]])
        question = numpy.array([[[0.2, 0.6], [0.4, 0.3], [0.5, 0.7], [0.1, .6]]])
        attention = numpy.array([[[0.3, 0.1, 0.5, 0.2],
                                  [0.4, 0.2, 0.8, 0.7],
                                  [0.8, 0.1, 0.6, 0.4]]])
        result = model.predict([document, question, attention])
        assert_almost_equal(result, numpy.array([[[0.37, 0.68000001, 0.3, 0.1],
                                                  [0.63, 1.27999997, 0.4, 0.2],
                                                  [0.54000002, 1.17000008, 0.8, 0.1]]]))
