# pylint: disable=no-self-use,invalid-name
from os.path import dirname, realpath, join
import shutil

from overrides import overrides
from deep_qa.data.dataset_readers.squad_sentence_selection_reader import SquadSentenceSelectionReader
from ...common.test_case import DeepQaTestCase

dir_path = dirname(realpath(__file__))


class TestSquadSentenceSelectionReader(DeepQaTestCase):

    @overrides
    def tearDown(self):
        shutil.rmtree(join(dir_path, "processed"))
        super(TestSquadSentenceSelectionReader, self).tearDown()

    def test_default_squad_sentence_selection_reader(self):
        question0 = "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?"
        context0 = ("Architecturally, the school has a Catholic character.###Atop "
                    "the Main Building's gold dome is a golden statue of the Virgin "
                    "Mary.###Immediately behind the basilica is the Grotto, a "
                    "Marian place of prayer and reflection.###Next to the Main "
                    "Building is the Basilica of the Sacred Heart.###Immediately "
                    "in front of the Main Building and facing it, is a copper "
                    "statue of Christ with arms upraised with the legend \"Venite "
                    "Ad Me Omnes\".###It is a replica of the grotto at Lourdes, "
                    "France where the Virgin Mary reputedly appeared to Saint "
                    "Bernadette Soubirous in 1858.###At the end of the main drive "
                    "(and in a direct line that connects through 3 statues and the "
                    "Gold Dome), is a simple, modern stone statue of Mary.")
        index0 = "5"
        expected_line0 = question0 + "\t" + context0 + "\t" + index0

        question1 = "What is in front of the Notre Dame Main Building?"
        context1 = ("Immediately behind the basilica is the Grotto, a Marian "
                    "place of prayer and reflection.###It is a replica of the grotto "
                    "at Lourdes, France where the Virgin Mary reputedly appeared to "
                    "Saint Bernadette Soubirous in 1858.###Next to the Main Building "
                    "is the Basilica of the Sacred Heart.###Atop the Main Building's "
                    "gold dome is a golden statue of the Virgin Mary.###At the end "
                    "of the main drive (and in a direct line that connects through 3 "
                    "statues and the Gold Dome), is a simple, modern stone statue of "
                    "Mary.###Architecturally, the school has a Catholic "
                    "character.###Immediately in front of the Main Building and "
                    "facing it, is a copper statue of Christ with arms upraised with "
                    "the legend \"Venite Ad Me Omnes\".")
        index1 = "6"
        expected_line1 = question1 + "\t" + context1 + "\t" + index1

        reader = SquadSentenceSelectionReader()
        output_filepath = reader.read_file(join(dir_path, "SQUAD_SAMPLE_FILE.json"))
        with open(output_filepath, "r") as generated_file:
            lines = []
            for line in generated_file:
                lines.append(line.strip())
        assert expected_line0 == lines[0]
        assert expected_line1 == lines[1]
