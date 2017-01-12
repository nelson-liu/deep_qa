from copy import deepcopy
from typing import Any, Dict, List, Tuple
import logging
import pickle

from keras import backend as K
from keras.layers import Dense, Dropout, Input, Layer, TimeDistributed
from overrides import overrides
import numpy

from ..common.params import get_choice_with_default
from ..data.dataset import TextDataset
from ..data.instances.instance import Instance, TextInstance
from ..data.instances.true_false_instance import TrueFalseInstance
from ..data.embeddings import PretrainedEmbeddings
from ..data.tokenizers import tokenizers
from ..data.data_indexer import DataIndexer
from ..layers.encoders import encoders, set_regularization_params
from ..layers.time_distributed_embedding import TimeDistributedEmbedding
from .models import DeepQaModel
from .trainer import Trainer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class TextTrainer(Trainer):
    """
    This is a Trainer that deals with word sequences as its fundamental data type (any TextDataset
    or TextInstance subtype is fine).  That means we have to deal with padding, with converting
    words (or characters) to indices, and encoding word sequences.  This class adds methods on top
    of Trainer to deal with all of that stuff.
    """
    def __init__(self, params: Dict[str, Any]):

        # If specified, we will use the vectors in this file and learn a projection matrix to get
        # word vectors of dimension `embedding_size`, instead of learning the embedding matrix
        # ourselves.
        self.pretrained_embeddings_file = params.pop('pretrained_embeddings_file', None)

        # If we're using pre-trained embeddings, should we fine tune them?
        self.fine_tune_embeddings = params.pop('fine_tune_embeddings', False)

        # Should we have a projection layer on top of our embedding layer? (mostly useful with
        # pre-trained embeddings)
        self.project_embeddings = params.pop('project_embeddings', False)

        # Number of dimensions to use for word embeddings
        self.embedding_size = params.pop('embedding_size', 50)

        # Dropout parameter to apply to the word embedding layer
        self.embedding_dropout = params.pop('embedding_dropout', 0.5)

        # Upper limit on length of word sequences in the training data. Ignored during testing (we
        # use the value set at training time, either from this parameter or from a loaded model).
        # If this is not set, we'll calculate a max length from the data.
        self.max_sentence_length = params.pop('max_sentence_length', None)

        # Upper limit on length of words in the training data. Only applicable for "words and
        # characters" text encoding.
        self.max_word_length = params.pop('max_word_length', None)

        # Which tokenizer to use for TextInstances.
        # Note that the way this works is a little odd - we need each Instance object to do the
        # right thing when we call instance.words() and instance.to_indexed_instance().  So we set
        # a class variable on TextInstance so that _all_ TextInstance objects use the setting that
        # we read here.
        tokenizer_params = params.pop('tokenizer', {})
        tokenizer_choice = get_choice_with_default(tokenizer_params, 'type', list(tokenizers.keys()))
        self.tokenizer = tokenizers[tokenizer_choice](tokenizer_params)
        TextInstance.tokenizer = self.tokenizer

        # These parameters specify the kind of encoder used to encode any word sequence input.
        # If given, this must be a dict.  We will use the "type" key in this dict (which must match
        # one of the keys in `encoders`) to determine the type of the encoder, then pass the
        # remaining args to the encoder constructor.
        # Hint: Use lstm or cnn for sentences, treelstm for logical forms, and bow for either.
        self.encoder_params = params.pop('encoder', {})

        # With some text_encodings, you can have separate sentence encoders and word encoders
        # (where sentence encoders combine word vectors into sentence vectors, and word encoders
        # combine character vectors into sentence vectors).  If you want to have separate encoders,
        # here's your chance to specify the word encoder.
        self.word_encoder_params = params.pop('word_encoder', self.encoder_params)

        super(TextTrainer, self).__init__(params)

        self.name = "TextTrainer"
        self.data_indexer = DataIndexer()

        # Model-specific member variables that will get set and used later.  For many of these, we
        # don't want to set them now, because they use max length information that only gets set
        # after reading the training data.
        self.embedding_layers = {}
        self.sentence_encoder_layer = None
        self.word_encoder_layer = None
        self._sentence_encoder_model = None

    @overrides
    def _prepare_data(self, dataset: TextDataset, for_train: bool):
        """
        Takes dataset, which could be a complex tuple for some classes, and produces as output a
        tuple of (inputs, labels), which can be used directly with Keras to either train or
        evaluate self.model.
        """
        if for_train:
            self.data_indexer.fit_word_dictionary(dataset)
        logger.info("Indexing dataset")
        indexed_dataset = dataset.to_indexed_dataset(self.data_indexer)
        max_lengths = self._get_max_lengths()
        logger.info("Padding dataset to lengths %s", str(max_lengths))
        indexed_dataset.pad_instances(max_lengths)
        if for_train:
            self._set_max_lengths(indexed_dataset.max_lengths())
        inputs, labels = indexed_dataset.as_training_data()
        if isinstance(inputs[0], tuple):
            inputs = [numpy.asarray(x) for x in zip(*inputs)]
        else:
            inputs = numpy.asarray(inputs)
        return inputs, numpy.asarray(labels)

    @overrides
    def _prepare_instance(self, instance: TextInstance, make_batch: bool=True):
        indexed_instance = instance.to_indexed_instance(self.data_indexer)
        indexed_instance.pad(self._get_max_lengths())
        inputs, label = indexed_instance.as_training_data()
        if make_batch:
            if isinstance(inputs, tuple):
                inputs = [numpy.expand_dims(x, axis=0) for x in inputs]
            else:
                inputs = numpy.expand_dims(inputs, axis=0)
        return inputs, label

    @overrides
    def _process_pretraining_data(self):
        """
        Adds words to the vocabulary based on the data used by the pretrainers.  We want this to
        happen before loading the training data so that we can use pretraining to expand our
        applicable vocabulary.
        """
        logger.info("Fitting the data indexer using the pretraining data")
        for pretrainer in self.pretrainers:
            pretrainer.fit_data_indexer()

    def _load_layers(self):
        """
        We have some variables that store individual layers used by the model, so that they can be
        re-used in several places if desired.  When we load a model, we have to set those layers,
        or things might break in really odd ways.  This method is in charge of finding those
        layers and initializing their variables.

        Note that this specifically looks for the layers defined by _get_embedded_sentence_input
        and _get_sentence_encoder.  If you change any of that in a subclass, or add other layers
        that are re-used, you must override this method, or loading models will break.  Similarly,
        if you change code in those two methods (e.g., making the sentence encoder into two
        layers), this method must be changed accordingly.

        Note that we don't need to store any TimeDistributed() layers directly, because they don't
        have any parameters themselves.
        """
        logger.info("Loading individual layers from model for re-use")
        for layer in self.model.layers:
            if 'embedding' in layer.name:
                # Because we store two layers in self.embedding_layers (an embedding and an
                # optional projection), this logic is a little complicated.  We need to check
                # whether this layer is the embedding layer or the projection layer, and handle
                # updating self.embedding_layers accordingly.
                #
                # TODO(matt): I don't think this logic will work with distributed projections, but
                # we'll worry about that later.
                embedding_name = layer.name.replace("_projection", "")
                if embedding_name in self.embedding_layers:
                    current_embedding, current_projection = self.embedding_layers[embedding_name]
                    if '_projection' in layer.name:
                        self.embedding_layers[embedding_name] = (current_embedding, layer)
                    else:
                        self.embedding_layers[embedding_name] = (layer, current_projection)
                else:
                    if '_projection' in layer.name:
                        self.embedding_layers[embedding_name] = (None, layer)
                    else:
                        self.embedding_layers[embedding_name] = (layer, None)
            elif layer.name == "sentence_encoder":
                logger.info("  Found sentence encoder")
                self.sentence_encoder_layer = layer
            elif layer.name == "timedist_sentence_encoder":
                logger.info("  Found sentence encoder")
                if self.sentence_encoder_layer is None:
                    self.sentence_encoder_layer = layer
                else:
                    logger.warning("  FOUND DUPLICATE SENTENCE ENCODER LAYER!  NOT SURE WHAT TO DO!")

    def get_sentence_vector(self, sentence: str):
        """
        Given a sentence (just a string), use the model's sentence encoder to convert it into a
        vector.  This is mostly just useful for debugging.
        """
        if self._sentence_encoder_model is None:
            self._build_sentence_encoder_model()
        instance = TrueFalseInstance(sentence, True)
        indexed_instance = instance.to_indexed_instance(self.data_indexer)
        indexed_instance.pad({'word_sequence_length': self.max_sentence_length})
        instance_input, _ = indexed_instance.as_training_data()
        encoded_instance = self._sentence_encoder_model.predict(numpy.asarray([instance_input]))
        return encoded_instance[0]

    def _get_max_lengths(self) -> Dict[str, int]:
        """
        This is about padding.  Any solver will have some number of things that need padding in
        order to make a compilable model, like the length of a sentence.  This method returns a
        dictionary of all of those things, mapping a length key to an int.

        Here we return the lengths that are applicable to encoding words and sentences.  If you
        have additional padding dimensions, call super()._get_max_lengths() and then update the
        dictionary.
        """
        return self.tokenizer.get_max_lengths(self.max_sentence_length, self.max_word_length)

    def _set_max_lengths(self, max_lengths: Dict[str, int]):
        """
        This is about padding.  Any solver will have some number of things that need padding in
        order to make a compilable model, like the length of a sentence.  This method sets those
        variables given a dictionary of lengths, perhaps computed from training data or loaded from
        a saved model.
        """
        self.max_sentence_length = max_lengths['word_sequence_length']
        self.max_word_length = max_lengths.get('word_character_length', None)

    @overrides
    def _set_params_from_model(self):
        self._set_max_lengths_from_model()

    @overrides
    def _save_auxiliary_files(self):
        super(TextTrainer, self)._save_auxiliary_files()
        data_indexer_file = open("%s_data_indexer.pkl" % self.model_prefix, "wb")
        pickle.dump(self.data_indexer, data_indexer_file)
        data_indexer_file.close()

    @overrides
    def _load_auxiliary_files(self):
        super(TextTrainer, self)._load_auxiliary_files()
        data_indexer_file = open("%s_data_indexer.pkl" % self.model_prefix, "rb")
        self.data_indexer = pickle.load(data_indexer_file)
        data_indexer_file.close()

    def _set_max_lengths_from_model(self):
        """
        Given a loaded model, set the max_lengths needed for padding.  This is necessary so that we
        can pad the test data if we just loaded a saved model.
        """
        raise NotImplementedError

    def _instance_type(self) -> Instance:
        """
        When reading datasets, what instance type should we create?
        """
        raise NotImplementedError

    def _load_dataset_from_files(self, files: List[str]):
        """
        This method assumes you have a TextDataset that can be read from a single file.  If you
        have something more complicated, you'll need to override this method (though, a solver that
        has background information could call this method, then do additional processing on the
        rest of the list, for instance).
        """
        return TextDataset.read_from_file(files[0], self._instance_type())

    def _get_sentence_shape(self, sentence_length: int=None) -> Tuple[int]:
        """
        Returns a tuple specifying the shape of a tensor representing a sentence.  This is not
        necessarily just (self.max_sentence_length,), because different text_encodings lead to
        different tensor shapes.
        """
        if sentence_length is None:
            # This can't be the default value for the function argument, because
            # self.max_sentence_length will not have been set at class creation time.
            sentence_length = self.max_sentence_length
        return self.tokenizer.get_sentence_shape(sentence_length, self.max_word_length)

    def _embed_input(self, input_layer: Layer, embedding_name: str="embedding"):
        """
        This function embeds a word sequence input, using an embedding defined by `embedding_name`.

        We need to take the input Layer here, instead of just returning a Layer that you can use as
        you wish, because we might have to apply several layers to the input, depending on the
        parameters you specified for embedding things.  So we return, essentially,
        `embedding(input_layer)`.

        The input layer can have arbitrary shape, as long as it ends with a word sequence.  For
        example, you could pass in a single sentence, a set of sentences, or a set of sets of
        sentences, and we will handle them correctly.

        Internally, we will create a dictionary mapping embedding names to embedding layers, so if
        you have several things you want to embed with the same embedding layer, be sure you use
        the same name each time (or just don't pass a name, which accomplishes the same thing).  If
        for some reason you want to have different embeddings for different inputs, use a different
        name for the embedding.

        In this function, we pass the work off to self.tokenizer, which might need to do some
        additional processing to actually give you a word embedding (e.g., if your text encoder
        uses both words and characters, we need to run the character encoder and concatenate the
        result with a word embedding).
        """
        return self.tokenizer.embed_input(input_layer, self, embedding_name)

    def _get_embedded_input(self, input_layer: Layer, embedding_name: str="embedding", vocab_name: str='words'):
        """
        This function does most of the work for self._embed_input.

        Additionally, we allow for multiple vocabularies, e.g., if you want to embed both
        characters and words with separate embedding matrices.
        """
        if embedding_name not in self.embedding_layers:
            self.embedding_layers[embedding_name] = self._get_new_embedding(embedding_name, vocab_name)

        embedding_layer, projection_layer = self.embedding_layers[embedding_name]
        embedded_input = embedding_layer(input_layer)
        if projection_layer is not None:
            for _ in range(2, K.ndim(input_layer)):  # 2 here to account for batch_size.
                projection_layer = TimeDistributed(projection_layer, name="timedist_" + projection_layer.name)
            embedded_input = projection_layer(embedded_input)
        if self.embedding_dropout > 0.0:
            embedded_input = Dropout(self.embedding_dropout)(embedded_input)

        return embedded_input

    def _get_new_embedding(self, name, vocab_name='words'):
        """
        Creates an Embedding Layer (and possibly also a Dense projection Layer) based on the
        parameters you've passed to the TextTrainer.  These could be pre-trained embeddings or not,
        could include a projection or not, and so on.
        """
        if self.pretrained_embeddings_file:
            embedding_layer = PretrainedEmbeddings.get_embedding_layer(
                    self.pretrained_embeddings_file,
                    self.data_indexer,
                    self.fine_tune_embeddings)
        else:
            # TimeDistributedEmbedding works with inputs of any shape.
            embedding_layer = TimeDistributedEmbedding(
                    input_dim=self.data_indexer.get_vocab_size(vocab_name),
                    output_dim=self.embedding_size,
                    mask_zero=True,  # this handles padding correctly
                    name=name)
        projection_layer = None
        if self.project_embeddings:
            projection_layer = TimeDistributed(Dense(output_dim=self.embedding_size,),
                                               name=name + '_projection')
        return embedding_layer, projection_layer

    def _get_sentence_encoder(self):
        """
        A sentence encoder takes as input a sequence of word embeddings, and returns as output a
        single vector encoding the sentence.  This is typically either a simple RNN or an LSTM, but
        could be more complex, if the "sentence" is actually a logical form.
        """
        if self.sentence_encoder_layer is None:
            self.sentence_encoder_layer = self._get_new_sentence_encoder()
        return self.sentence_encoder_layer

    def _get_new_sentence_encoder(self, name="sentence_encoder"):
        # The code that follows would be destructive to self.encoder_params (lots of calls to
        # params.pop()), but we may need to create several encoders.  So we'll make a copy and use
        # that instead of self.encoder_params.
        return self._get_new_encoder(deepcopy(self.encoder_params), name)

    def _get_word_encoder(self):
        """
        This is like a sentence encoder, but for sentences; we don't just use
        self._get_sentence_encoder() for this, because we allow different models to be specified
        for this.
        """
        if self.word_encoder_layer is None:
            self.word_encoder_layer = self._get_new_word_encoder()
        return self.word_encoder_layer

    def _get_new_word_encoder(self, name="word_encoder"):
        return self._get_new_encoder(deepcopy(self.word_encoder_params), name)

    def _get_new_encoder(self, params: Dict[str, Any], name: str):
        encoder_type = get_choice_with_default(params, "type", list(encoders.keys()))
        params["name"] = name
        params["output_dim"] = self.embedding_size
        set_regularization_params(encoder_type, params)
        return encoders[encoder_type](**params)

    def _build_sentence_encoder_model(self):
        """
        Here we pull out just a couple of layers from self.model and use them to define a
        stand-alone encoder model.

        Specifically, we need the part of the model that gets us from word index sequences to word
        embedding sequences, and the part of the model that gets us from word embedding sequences
        to sentence vectors.

        This must be called after self.max_sentence_length has been set, which happens when
        self._get_training_data() is called.
        """
        sentence_input = Input(shape=(self.max_sentence_length,), dtype='int32', name="sentence_input")
        embedded_input = self._embed_input(sentence_input)
        encoder_layer = self._get_sentence_encoder()
        encoded_input = encoder_layer(embedded_input)
        self._sentence_encoder_model = DeepQaModel(input=sentence_input, output=encoded_input)

        # Loss and optimizer do not matter here since we're not going to train this model. But it
        # needs to be compiled to use it for prediction.
        self._sentence_encoder_model.compile(loss="mse", optimizer="adam")
        self._sentence_encoder_model.summary()

    @overrides
    def _overall_debug_output(self, output_dict: Dict[str, numpy.array]) -> str:
        """
        We'll do something different here: if "embedding" is in output_dict, we'll output the
        embedding matrix at the top of the debug file.  Note that this could be _huge_ - you should
        only do this for debugging on very simple datasets.
        """
        result = super(TextTrainer, self)._overall_debug_output(output_dict)
        if any('embedding' in layer_name for layer_name in output_dict.keys()):
            embedding_layers = set([n for n in output_dict.keys() if 'embedding' in n])
            for embedding_layer in embedding_layers:
                if '_projection' in embedding_layer:
                    continue
                result += self._render_embedding_matrix(embedding_layer)
        return result

    def _render_embedding_matrix(self, embedding_name: str) -> str:
        result = 'Embedding matrix for %s:\n' % embedding_name
        embedding_weights = self.embedding_layers[embedding_name][0].get_weights()[0]
        for i in range(self.data_indexer.get_vocab_size()):
            word = self.data_indexer.get_word_from_index(i)
            word_vector = '[' + ' '.join('%.4f' % x for x in embedding_weights[i]) + ']'
            result += '%s\t%s\n' % (word, word_vector)
        result += '\n'
        return result

    @classmethod
    def _get_custom_objects(cls):
        custom_objects = super(TextTrainer, cls)._get_custom_objects()
        for value in encoders.values():
            if value.__name__ not in ['LSTM']:
                custom_objects[value.__name__] = value
        return custom_objects
