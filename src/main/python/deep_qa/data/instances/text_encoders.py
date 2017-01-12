from collections import OrderedDict
from typing import Callable, Dict, List, Tuple

from keras import backend as K
from keras.layers import Layer, merge
from overrides import overrides

from ..data_indexer import DataIndexer
from ...layers.vector_matrix_split import VectorMatrixSplit
from ...layers.wrappers import FixedTimeDistributed

class TextEncoder:
    """
    A TextEncoder maps strings to numpy arrays, given a Tokenizer and a DataIndexer.  This is used in
    TextInstance.to_indexed_instance().

    There are several ways you might want to do this, so we abstract away that decision into a class
    that does the work, so we can provide several options.  For instance, you might want a sequence of
    word tokens, or a sequence of characters, or word tokens plus the word's associated characters, or
    something else.
    """
    def get_words_for_indexer(self, text: str, tokenize: Callable[[str], List[str]]) -> Dict[str, List[str]]:
        """
        The DataIndexer needs to assign indices to whatever strings we see in the training data
        (possibly doing some frequency filtering and using an OOV token).  This method takes some
        text and returns whatever the DataIndexer would be asked to index from that text.  Note
        that this returns a dictionary of token lists keyed by namespace.  Typically, the key would
        be either 'words' or 'characters'.  An example for indexing the string 'the' might be
        {'words': ['the'], 'characters': ['t', 'h', 'e']}, if you are indexing both words and
        characters.
        """
        raise NotImplementedError

    def index_text(self,
                   text: str,
                   tokenize: Callable[[str], List[str]],
                   data_indexer: DataIndexer) -> List:
        """
        This method actually converts some text into an indexed list.  This could be a list of
        integers (for either word tokens or characters), or it could be a list of arrays (for word
        tokens combined with characters), or something else.
        """
        raise NotImplementedError

    def embed_input(self,
                    input_layer: Layer,
                    text_trainer: 'TextTrainer',
                    embedding_name: str="embedding"):
        """
        Applies embedding layers to the input_layer.  See TextTrainer._embed_input for a more
        detailed comment on what this method does.

        - `input_layer` should be a Keras Input() layer.
        - `text_trainer` is a TextTrainer instance, so we can access methods on it like
          `text_trainer._get_embedded_input`, which actually applies an embedding layer, projection
          layers, and dropout to the input layer.  Simple TextEncoders will basically just call
          this function and be done.  More complicated TextEncoders might need additional logic on
          top of just calling `text_trainer._get_embedded_input`.
        - `embedding_name` allows for different embedding matrices.
        """
        raise NotImplementedError

    def get_sentence_shape(self, sentence_length: int, word_length: int) -> Tuple[int]:
        """
        If we have a text sequence of length `sentence_length`, what shape would that correspond to
        with this encoding?  For words or characters only, this would just be (sentence_length,).
        For an encoding that contains both words and characters, it might be (sentence_length,
        word_length).
        """
        raise NotImplementedError

    def get_max_lengths(self, sentence_length: int, word_length: int) -> Dict[str, int]:
        """
        When dealing with padding in TextTrainer, TextInstances need to know what to pad and how
        much.  This function takes a potential max sentence length and word length, and returns a
        `lengths` dictionary containing keys for the padding that is applicable to this encoding.
        """
        raise NotImplementedError


class WordTokenEncoder(TextEncoder):
    @overrides
    def get_words_for_indexer(self, text: str, tokenize: Callable[[str], List[str]]) -> List[str]:
        return {'words': tokenize(text)}

    @overrides
    def index_text(self,
                   text: str,
                   tokenize: Callable[[str], List[str]],
                   data_indexer: DataIndexer) -> List:
        return [data_indexer.get_word_index(token, namespace='words') for token in tokenize(text)]

    @overrides
    def embed_input(self,
                    input_layer: Layer,
                    text_trainer: 'TextTrainer',
                    embedding_name: str="embedding"):
        # pylint: disable=protected-access
        return text_trainer._get_embedded_input(input_layer, 'word_' + embedding_name, 'words')

    @overrides
    def get_sentence_shape(self, sentence_length: int, word_length: int) -> Tuple[int]:
        return (sentence_length,)

    @overrides
    def get_max_lengths(self, sentence_length: int, word_length: int) -> Dict[str, int]:
        return {'word_sequence_length': sentence_length}


class CharacterEncoder(TextEncoder):
    @overrides
    def get_words_for_indexer(self, text: str, tokenize: Callable[[str], List[str]]) -> List[str]:
        return {'characters': [char for char in text]}

    @overrides
    def index_text(self,
                   text: str,
                   tokenize: Callable[[str], List[str]],
                   data_indexer: DataIndexer) -> List:
        return [data_indexer.get_word_index(char, namespace='characters') for char in text]

    @overrides
    def embed_input(self,
                    input_layer: Layer,
                    text_trainer: 'TextTrainer',
                    embedding_name: str="embedding"):
        # pylint: disable=protected-access
        return text_trainer._get_embedded_input(input_layer, 'character_' + embedding_name, 'characters')

    @overrides
    def get_sentence_shape(self, sentence_length: int, word_length: int) -> Tuple[int]:
        return (sentence_length,)

    @overrides
    def get_max_lengths(self, sentence_length: int, word_length: int) -> Dict[str, int]:
        # Note that `sentence_length` here is the number of _characters_ in the sentence, because
        # of how `self.index_text` works.  And even though the name isn't great, we'll use
        # `word_sequence_length` for the key to this, so that the rest of the code is simpler.
        return {'word_sequence_length': sentence_length}


class WordAndCharacterEncoder(TextEncoder):
    @overrides
    def get_words_for_indexer(self, text: str, tokenize: Callable[[str], List[str]]) -> List[str]:
        words = tokenize(text)
        characters = [char for char in text]
        return {'words': words, 'characters': characters}

    @overrides
    def index_text(self,
                   text: str,
                   tokenize: Callable[[str], List[str]],
                   data_indexer: DataIndexer) -> List:
        words = tokenize(text)
        arrays = []
        for word in words:
            word_index = data_indexer.get_word_index(word, namespace='words')
            # TODO(matt): I'd be nice to keep the capitalization of the word in the character
            # representation.  Doing that would require pretty fancy logic here, though.
            char_indices = [data_indexer.get_word_index(char, namespace='characters') for char in word]
            arrays.append([word_index] + char_indices)
        return arrays

    @overrides
    def embed_input(self,
                    input_layer: Layer,
                    text_trainer: 'TextTrainer',
                    embedding_name: str="embedding"):
        """
        A combined word-and-characters representation requires some fancy footwork to do the
        embedding properly.

        This method assumes the input shape is (..., sentence_length, word_length + 1), where the
        first integer for each word in the tensor is the word index, and the remaining word_length
        entries is the character sequence.  We'll first split this into two tensors, one of shape
        (..., sentence_length), and one of shape (..., sentence_length, word_length), where the
        first is the word sequence, and the second is the character sequence for each word.  We'll
        pass the word sequence through an embedding layer, as normal, and pass the character
        sequence through a _separate_ embedding layer, then an encoder, to get a word vector out.
        We'll then concatenate the two word vectors, returning a tensor of shape
        (..., sentence_length, embedding_dim * 2).
        """
        # pylint: disable=protected-access
        # This is happening before any masking is done, so we don't need to worry about the
        # mask_split_axis argument to VectorMatrixSplit.
        words, characters = VectorMatrixSplit(split_axis=-1)(input_layer)
        word_embedding = text_trainer._get_embedded_input(words,
                                                          embedding_name='word_' + embedding_name,
                                                          vocab_name='words')
        character_embedding = text_trainer._get_embedded_input(characters,
                                                               embedding_name='character_' + embedding_name,
                                                               vocab_name='characters')

        # A note about masking here: we care about the character masks when encoding a character
        # sequence, so we need the mask to be passed to the character encoder correctly.  However,
        # we _don't_ care here about whether the whole word will be masked, as the word_embedding
        # will carry that information, so the output mask returned by the TimeDistributed layer
        # here will be ignored.
        word_encoder = FixedTimeDistributed(text_trainer._get_word_encoder())
        word_encoding = word_encoder(character_embedding)

        merge_mode = lambda inputs: K.concatenate(inputs, axis=-1)
        def merge_shape(input_shapes):
            output_shape = list(input_shapes[0])
            output_shape[-1] *= 2
            return tuple(output_shape)
        merge_mask = lambda masks: masks[0]
        final_embedded_input = merge([word_embedding, word_encoding],
                                     mode=merge_mode,
                                     output_shape=merge_shape,
                                     output_mask=merge_mask,
                                     name='combined_word_embedding')
        return final_embedded_input

    @overrides
    def get_sentence_shape(self, sentence_length: int, word_length: int=None) -> Tuple[int]:
        return (sentence_length, word_length)

    @overrides
    def get_max_lengths(self, sentence_length: int, word_length: int) -> Dict[str, int]:
        return {'word_sequence_length': sentence_length, 'word_character_length': word_length}


# The first item added here will be used as the default in some cases.
text_encoders = OrderedDict()  # pylint: disable=invalid-name
text_encoders['word tokens'] = WordTokenEncoder()
text_encoders['characters'] = CharacterEncoder()
text_encoders['words and characters'] = WordAndCharacterEncoder()
