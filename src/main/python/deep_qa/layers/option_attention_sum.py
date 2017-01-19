from keras import backend as K
from keras.layers import Layer


class OptionAttentionSum(Layer):
    """
    This Layer takes four inputs: a tensor of document indices, a tensor of
    document probabilities, a tensor of answer options, and a string describing
    how to calculate the probability of options that consist of multiple words.
    We compute the probability of each of the answer options in the fashion
    described in the paper "Text Comprehension with the Attention Sum
    Reader Network" (Kadlec et. al 2016).
    """
    def __init__(self, multiword_option_mode="mean", **kwargs):
        """
        Construct a new OptionAttentionSum layer.

        Parameters
        ----------
        multiword_option_mode: str, optional (default="mean")
            Describes how to calculate the probability of options
            that contain multiple words. If "mean", the probability of
            the option is taken to be the mean of the probabilities of
            its constituent words. If "sum", the probability of the option
            is taken to be the sum of the probabilities of its constituent
            words.
        """

        if multiword_option_mode != "mean" and multiword_option_mode != "sum":
            raise ValueError("multiword_option_mode must be 'mean' or "
                             "'sum', got {}.".format(multiword_option_mode))
        self.multiword_option_mode = multiword_option_mode
        self.supports_masking = True
        super(OptionAttentionSum, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shapes):
        return (input_shapes[2][0], input_shapes[2][1])

    def compute_mask(self, inputs, input_mask=None):  # pylint: disable=unused-argument
        options = inputs[2]
        padding_mask = K.not_equal(options, K.zeros_like(options))
        return K.cast(K.any(padding_mask, axis=2), "float32")

    def call(self, inputs, mask=None):
        """
        Calculate the probability of each answer option.

        Parameters
        ----------
        Inputs: List of Tensors
            The inputs to the layer must be passed in as a list to the
            ``call`` function. The inputs expected are a Tensor of
            document indicies, a Tensor of document probabilities, and
            a Tensor of options (in that order).
            The documents indicies tensor is a 2D tensor of shape
            (batch size, max document length in words), where each row
            represents which words compose the document.
            The document probabilities tensor is a 2D Tensor of shape
            (batch size, max document length in words), where each row
            represents which words compose the document.
            The options tensor is of shape (batch size, max number of options,
            max number of words in option) representing the possible answer
            options.
        mask: Tensor or None, optional (default=None)
            Tensor of shape (batch size, max number of options) representing
            which options are padding and thus have a 0 in the associated
            mask position.

        Returns
        -------
        options_probabilities : Tensor
            Tensor with shape (batch size, max number of options) with floats,
            where each float is the probability of the option as calculated by
            ``self.multiword_option_mode``.
        """
        document_indicies, document_probabilities, options = inputs
        expanded_indicies = K.expand_dims(K.expand_dims(document_indicies, 1), 1)
        tiled_indicies = K.repeat_elements(K.repeat_elements(expanded_indicies,
                                                             K.int_shape(options)[1], axis=1),
                                           K.int_shape(options)[2], axis=2)

        expanded_probabilities = K.expand_dims(K.expand_dims(document_probabilities, 1), 1)
        tiled_probabilities = K.repeat_elements(K.repeat_elements(expanded_probabilities,
                                                                  K.int_shape(options)[1], axis=1),
                                                K.int_shape(options)[2], axis=2)

        expanded_options = K.expand_dims(options, 3)
        tiled_options = K.repeat_elements(expanded_options,
                                          K.int_shape(document_indicies)[-1], axis=3)

        # generate a binary tensor of the same shape as tiled_options /
        # tiled_indicies indicating if index is option or padding
        options_words_mask = K.cast(K.equal(tiled_options, tiled_indicies),
                                    "float32")

        # apply mask to the probabilities to select the
        # indices for probabilities that correspond with option words
        selected_probabilities = options_words_mask * tiled_probabilities

        # sum up the probabilities to get aggregate probability for
        # each options constituent words
        options_word_probabilities = K.sum(selected_probabilities, axis=3)

        sum_option_words_probabilities = K.sum(options_word_probabilities,
                                               axis=2)

        if self.multiword_option_mode == "mean":
            # figure out how many words (excluding padding) are in each option
            # generate mask on the input option
            option_mask = K.cast(K.not_equal(options, K.zeros_like(options)),
                                 "float32")
            # num words in each option
            divisor = K.sum(option_mask, axis=2)
        else:
            # since we're taking the sum, just divide all sums by 1
            divisor = K.ones_like(sum_option_words_probabilities)

        # now divide the sums by the divisor we generated aboce
        option_probabilities = sum_option_words_probabilities / divisor
        return option_probabilities
