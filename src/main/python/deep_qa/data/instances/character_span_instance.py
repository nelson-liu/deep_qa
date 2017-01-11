from typing import Tuple, List

from overrides import overrides

from .question_passage_instance import QuestionPassageInstance
from ..tokenizer import tokenizers, Tokenizer


class CharacterSpanInstance(QuestionPassageInstance):
    """
    A CharacterSpanInstance is a QuestionPassageInstance that represents a
    (question, passage) pair with an associated label, which is the data given
    for the span prediction task. The label is a span of characters in the
    passage that indicates where the answer to the question begins and where
    the answer to the question ends. The main thing this class handles over
    QuestionPassageInstance is in specifying the form of and how to index the
    label, which is given as a span of _characters_ in the passage. The label
    we are going to use in the rest of the code is a span of _tokens_ in the
    passage, so the mapping from character labels to token labels depends on
    the tokenization we did, and the logic to handle this is, unfortunately, a
    little complicated. The label conversion happens when converting a
    CharacterSpanInstance to in IndexedInstance (where character indices are
    generally lost, anyway).

    This class should be used to represent training instances for the SQuAD
    (Stanford Question Answering) and NewsQA datasets, to name a few.
    """
    def __init__(self,
                 question: str,
                 passage: str,
                 label: Tuple[int, int],
                 index: int=None,
                 tokenizer: Tokenizer=tokenizers['default']()):
        super(CharacterSpanInstance, self).__init__(question, passage, label,
                                                    index, tokenizer)

    def __str__(self):
        return ('CharacterSpanInstance(' + self.question_text + ', ' +
                self.passage_text + ', ' + str(self.label) + ')')

    @overrides
    def _index_label(self, label: Tuple[int, int]) -> List[int]:
        """
        Specify how to index `self.label`, which is needed to convert the
        CharacterSpanInstance into an IndexedInstance (handled in superclass).
        """
        new_label = None
        if self.label is not None:
            new_label = self.tokenizer.char_span_to_token_span(self.passage_text,
                                                               self.label)
        return new_label

    @classmethod
    def read_from_line(cls,
                       line: str,
                       default_label: bool=None,
                       tokenizer: Tokenizer=tokenizers['default']()):
        """
        Reads a CharacterSpanInstance object from a line. The format
        has one of two options:

        (1) [example index][tab][question][tab][passage][tab][label]
        (2) [question][tab][passage][tab][label]

        [label] is assumed to be a comma-separated pair of integers.

        default_label is ignored, but we keep the argument to match the
        interface.

        """
        fields = line.split("\t")

        if len(fields) == 4:
            index_string, question, passage, label = fields
            index = int(index_string)
        elif len(fields) == 3:
            question, passage, label = fields
            index = None
        else:
            raise RuntimeError("Unrecognized line format: " + line)
        label_fields = label.split(",")
        span_begin = int(label_fields[0])
        span_end = int(label_fields[1])
        return cls(question, passage, (span_begin, span_end), index, tokenizer)
