from typing import Dict, List, Any

import numpy as np
from overrides import overrides

from .instance import TextInstance, IndexedInstance
from ..data_indexer import DataIndexer
from ..tokenizer import tokenizers, Tokenizer


class QuestionPassageInstance(TextInstance):
    """
    A QuestionPassageInstance is a base class for datasets that contain
    consist primarily of a question text and a passage, where the passage contains
    the answer to the question. This class should not be used directly due to
    the missing ``_index_label`` function, use a subclass instead.
    """
    def __init__(self,
                 question_text: str,
                 passage_text: str,
                 label: Any,
                 index: int=None,
                 tokenizer: Tokenizer=tokenizers['default']()):
        super(QuestionPassageInstance, self).__init__(label, index, tokenizer)
        self.question_text = question_text
        self.passage_text = passage_text

    def __str__(self):
        return ('QuestionPassageInstance(' + self.question_text +
                ', ' + self.passage_text + ', ' +
                str(self.label) + ')')

    @overrides
    def words(self) -> List[str]:
        return (self._words_from_text(self.question_text) +
                self._words_from_text(self.passage_text))

    def _index_label(self, label: Any) -> List[int]:
        """
        Index the labels. Since we don't know what form the label takes,
        we leave it to subclasses to implement this method.
        """
        raise NotImplementedError

    @overrides
    def to_indexed_instance(self, data_indexer: DataIndexer):
        question_indices = self._index_text(self.question_text, data_indexer)
        passage_indices = self._index_text(self.passage_text, data_indexer)
        label_indices = self._index_label(self.label)
        return IndexedQuestionPassageInstance(question_indices,
                                              passage_indices, label_indices,
                                              self.index)

class IndexedQuestionPassageInstance(IndexedInstance):
    """
    This is an indexed instance that is used for (question, passage) pairs.
    """
    def __init__(self,
                 question_indices: List[int],
                 passage_indices: List[int],
                 label: List[int],
                 index: int=None):
        super(IndexedQuestionPassageInstance, self).__init__(label, index)
        self.question_indices = question_indices
        self.passage_indices = passage_indices

    @classmethod
    @overrides
    def empty_instance(cls):
        return IndexedQuestionPassageInstance([], [], label=None, index=None)

    @overrides
    def get_lengths(self) -> Dict[str, int]:
        """
        We need to pad at least the question length, the passage length, and the
        word length across all the questions and passages. Subclasses that
        add more arguments should also override this method to enable padding on said
        arguments.
        """
        question_lengths = self._get_word_sequence_lengths(self.question_indices)
        passage_lengths = self._get_word_sequence_lengths(self.passage_indices)
        lengths = {}

        # the number of words in the longest question
        lengths['num_question_words'] = question_lengths['word_sequence_length']

        # the number of words in the longest passage
        lengths['num_passage_words'] = passage_lengths['word_sequence_length']

        if 'word_character_length' in question_lengths and 'word_character_length' in passage_lengths:
            # the length of the longest word across the passage and question
            lengths['word_character_length'] = max(question_lengths['word_character_length'],
                                                   passage_lengths['word_character_length'])
        return lengths

    @overrides
    def pad(self, max_lengths: List[int]):
        """
        In this function, we pad the questions and passages (in terms of number of words in each),
        as well as the individual words in the questions and passages themselves.
        """
        max_lengths['word_sequence_length'] = max_lengths['num_question_words']
        self.question_indices = self.pad_word_sequence(self.question_indices,
                                                       max_lengths)
        max_lengths['word_sequence_length'] = max_lengths['num_question_words']
        self.passage_indices = self.pad_word_sequence(self.passage_indices,
                                                      max_lengths)

    @overrides
    def as_training_data(self):
        question_array = np.asarray(self.question_indices, dtype='int32')
        passage_array = np.asarray(self.passage_indices, dtype='int32')
        return (question_array, passage_array), np.asarray(self.label)
