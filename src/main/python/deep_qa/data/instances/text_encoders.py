from collections import OrderedDict
from typing import Callable, Dict, List

from overrides import overrides

from ..data_indexer import DataIndexer

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


# The first item added here will be used as the default in some cases.
text_encoders = OrderedDict()  # pylint: disable=invalid-name
text_encoders['word tokens'] = WordTokenEncoder()
text_encoders['characters'] = CharacterEncoder()
text_encoders['words and characters'] = WordAndCharacterEncoder()
