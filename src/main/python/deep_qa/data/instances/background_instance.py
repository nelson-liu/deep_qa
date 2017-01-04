from typing import Dict, List

import numpy
from overrides import overrides

from .instance import TextInstance, IndexedInstance
from ..data_indexer import DataIndexer


class BackgroundInstance(TextInstance):
    """
    An Instance that has background knowledge associated with it.  That background knowledge can
    currently only be expressed as a list of sentences.  Maybe someday we'll expand that to allow
    other kinds of background knowledge.
    """
    def __init__(self, instance: TextInstance, background: List[str]):
        super(BackgroundInstance, self).__init__(instance.label, instance.index, instance.tokenizer)
        self.instance = instance
        self.background = background

    def __str__(self):
        string = 'BackgroundInstance(\n   '
        string += str(self.instance)
        string += ',\n'
        string += '   [\n'
        for background in self.background:
            string += '      '
            string += background
            string += ',\n'
        string += '   ]\n'
        string += ')'
        return string

    @overrides
    def words(self):
        words = self.instance.words()
        for background_text in self.background:
            background_words = self._words_from_text(background_text)
            for namespace in words:
                words[namespace].extend(background_words[namespace])
        return words

    @overrides
    def to_indexed_instance(self, data_indexer: DataIndexer):
        indexed_instance = self.instance.to_indexed_instance(data_indexer)
        background_indices = []
        for text in self.background:
            background_indices.append(self._index_text(text, data_indexer))
        return IndexedBackgroundInstance(indexed_instance, background_indices)


class IndexedBackgroundInstance(IndexedInstance):
    """
    An IndexedInstance that has background knowledge associated with it, where the background
    knowledge has also been indexed.
    """
    contained_instance_type = None
    def __init__(self,
                 indexed_instance: IndexedInstance,
                 background_indices: List[List[int]]):
        super(IndexedBackgroundInstance, self).__init__(indexed_instance.label, indexed_instance.index)
        self.indexed_instance = indexed_instance
        self.background_indices = background_indices

        # We need to set this here so that we know what kind of contained instance we should create
        # when we're asked for an empty IndexedBackgroundInstance.  Note that this assumes that
        # you'll only ever have one underlying Instance type, which is a reasonable assumption
        # given our current code.
        IndexedBackgroundInstance.contained_instance_type = indexed_instance.__class__

    @classmethod
    @overrides
    def empty_instance(cls):
        contained_instance = IndexedBackgroundInstance.contained_instance_type.empty_instance()
        return IndexedBackgroundInstance(contained_instance, [])

    @overrides
    def get_lengths(self) -> Dict[str, int]:
        """
        Prep for padding; see comment on this method in the super class.  Here we extend the return
        value from our super class with the padding lengths necessary for background_indices.

        Additionally, as we currently use the same encoder for both a sentence and its background
        knowledge, we'll also modify the word_indices length to look at the background sentences
        too.
        """
        lengths = self.indexed_instance.get_lengths()
        lengths['background_sentences'] = len(self.background_indices)
        background_lengths = [self._get_word_sequence_lengths(background)
                              for background in self.background_indices]
        if self.background_indices:
            max_background_length = max(l['word_sequence_length'] for l in background_lengths)
            lengths['word_sequence_length'] = max(lengths['word_sequence_length'], max_background_length)
            if 'word_character_length' in lengths:
                max_background_character_length = max([l['word_character_length'] for l in background_lengths])
                max_character_length = max([lengths['word_character_length'], max_background_character_length])
                lengths['word_character_length'] = max_character_length
        return lengths

    @overrides
    def pad(self, max_lengths: Dict[str, int]):
        """
        We let self.indexed_instance pad itself, and in this method we mostly worry about padding
        background_indices.  We need to pad it in two ways: (1) we need len(background_indices) to
        be the same for all instances, and (2) we need len(background_indices[i]) to be the same
        for all i, for all instances.  We'll use the word_indices length from the super class for
        (2).
        """
        self.indexed_instance.pad(max_lengths)
        background_length = max_lengths['background_sentences']

        # Padding (1): making sure we have the right number of background sentences.  We also need
        # to truncate, if necessary.
        if len(self.background_indices) > background_length:
            self.background_indices = self.background_indices[:background_length]
        for _ in range(background_length - len(self.background_indices)):
            self.background_indices.append([0])

        # Padding (2): making sure all background sentences have the right length.
        padded_background = []
        for background in self.background_indices:
            padded_background.append(self.pad_word_sequence(background, max_lengths))
        self.background_indices = padded_background

    @overrides
    def as_training_data(self):
        """
        This returns a complex output.  In the simplest case, the contained instance is just a
        TrueFalseInstance, with a single sentence input.  In this case, we'll return a tuple of
        (sentence_array, background_array) as the inputs (and, as always, the label from the
        contained instance).

        If the contained instance itself has multiple inputs it returns, we need the
        background_array to be second in the list (because that makes the implementation in the
        memory network solver much easier).  That means we need to change the order of things
        around a bit.
        """
        instance_inputs, label = self.indexed_instance.as_training_data()
        background_array = numpy.asarray(self.background_indices, dtype='int32')
        if isinstance(instance_inputs, tuple):
            final_inputs = (instance_inputs[0],) + (background_array,) + instance_inputs[1:]
        else:
            final_inputs = (instance_inputs, background_array)
        return final_inputs, label
