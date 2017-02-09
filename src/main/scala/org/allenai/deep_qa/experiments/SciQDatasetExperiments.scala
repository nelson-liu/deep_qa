package org.allenai.deep_qa.experiments

import org.allenai.deep_qa.experiments.datasets.SciQDatasets
import org.allenai.deep_qa.experiments.datasets.ScienceDatasets
import org.allenai.deep_qa.pipeline.Evaluator

import com.mattg.util.FileUtil
import org.json4s.JsonDSL._
import org.json4s._

object SciQDatasetExperiments {
  val fileUtil = new FileUtil

  def experiment(
    name: String,
    modelParams: JValue,
    trainingDataset: JValue,
    validationDataset: JValue
  ): JValue = {
    ("name" -> name) ~
      ("model params" -> modelParams) ~
      ("dataset" -> trainingDataset) ~
      ("validation dataset" -> validationDataset)
  }

  /*
   * Turn Omnibus-8 reading comprehension train, test, and dev files
   * with BUSC background to datasets.
   */

  val omnibusEightTrainReadingComprehensionDataset: JValue =
    ("dataset type" -> "from sentence producers") ~
      ("data files" -> Seq(ScienceDatasets.readingComprehensionOmnibusQaGradeEightTrainQuestionsWithBuscBackground))

  val omnibusEightDevReadingComprehensionDataset: JValue =
    ("dataset type" -> "from sentence producers") ~
      ("data files" -> Seq(ScienceDatasets.readingComprehensionOmnibusQaGradeEightDevQuestionsWithBuscBackground))

  val omnibusEightTestReadingComprehensionDataset: JValue =
    ("dataset type" -> "from sentence producers") ~
      ("data files" -> Seq(ScienceDatasets.readingComprehensionOmnibusQaGradeEightTestQuestionsWithBuscBackground))

  /*
   * Turn Omnibus-4 reading comprehension train, test, and dev files
   * with BUSC background to datasets.
   */

  val omnibusFourTrainReadingComprehensionDataset: JValue =
    ("dataset type" -> "from sentence producers") ~
      ("data files" -> Seq(ScienceDatasets.readingComprehensionOmnibusQaGradeFourTrainQuestionsWithBuscBackground))

  val omnibusFourDevReadingComprehensionDataset: JValue =
    ("dataset type" -> "from sentence producers") ~
      ("data files" -> Seq(ScienceDatasets.readingComprehensionOmnibusQaGradeFourDevQuestionsWithBuscBackground))

  val omnibusFourTestReadingComprehensionDataset: JValue =
    ("dataset type" -> "from sentence producers") ~
      ("data files" -> Seq(ScienceDatasets.readingComprehensionOmnibusQaGradeFourTestQuestionsWithBuscBackground))

  /*
   * Create combined datasets with Omnibus4Train + SciQ Train and
   * Omnibus8Train + SciQ Train.
   */

  val combinedSciQTrainOmnibusEightTrainDataset: JValue =
    ("dataset type" -> "combined") ~
      ("datasets" -> Seq(SciQDatasets.sciQTrainDataset, omnibusEightTrainReadingComprehensionDataset))~
      ("output directory" -> s"/efs/data/dlfa/processed/omnibus_8_train_and_sciq_train_combined/")

  val combinedSciQTrainOmnibusFourTrainDataset: JValue =
    ("dataset type" -> "combined") ~
      ("datasets" -> Seq(SciQDatasets.sciQTrainDataset, omnibusFourTrainReadingComprehensionDataset))~
      ("output directory" -> s"/efs/data/dlfa/processed/omnibus_4_train_and_sciq_train_combined/")

  def omnibusGradeFourExperiment(
    name: String,
    modelParams: JValue,
    trainingDataset: JValue=omnibusFourTrainReadingComprehensionDataset
  ): JValue = {
    experiment(
      name,
      modelParams,
      trainingDataset,
      omnibusFourDevReadingComprehensionDataset
    )
  }

  def omnibusGradeEightExperiment(
    name: String,
    modelParams: JValue,
    trainingDataset: JValue=omnibusEightTrainReadingComprehensionDataset
  ): JValue = {
    experiment(
      name,
      modelParams,
      trainingDataset,
      omnibusEightDevReadingComprehensionDataset
    )
  }

  def combinedSciQOmnibusGradeFourExperiment(
    name: String,
    modelParams: JValue,
    trainingDataset: JValue=combinedSciQTrainOmnibusFourTrainDataset
  ): JValue = {
    experiment(
      name,
      modelParams,
      trainingDataset,
      omnibusFourDevReadingComprehensionDataset
    )
  }

  def combinedSciQOmnibusGradeEightExperiment(
    name: String,
    modelParams: JValue,
    trainingDataset: JValue=combinedSciQTrainOmnibusEightTrainDataset
  ): JValue = {
    experiment(
      name,
      modelParams,
      trainingDataset,
      omnibusEightDevReadingComprehensionDataset
    )
  }

  val attentionSumReader: JValue = {
    Models.attentionSumReader merge
    (("patience" -> 1) ~
      ("preferred_backend" -> "theano") ~
      ("encoder" ->
        ("default" ->
          ("type" -> "bi_gru")~
          ("output_dim" -> 384)
        )
      ) ~
      ("seq2seq_encoder" ->
        ("default" ->
          ("type" -> "bi_gru") ~
          ("encoder_params" ->
            ("output_dim" -> 384)
          ) ~
          ("wrapper_params" -> JObject())
        )
      ) ~
      ("optimizer" ->
        ("type" -> "adam") ~
        ("clipnorm" -> 10.0) ~
        ("lr" -> 0.0005)
      ) ~
      ("embedding_dropout" -> 0.0) ~
      ("patience" -> 0) ~
      ("embedding_size" -> 256) ~
      ("num_epochs" -> 5)
    )
  }

  val gatedAttentionReader: JValue = {
    Models.gatedAttentionReader merge
    (("embedding_size" -> 100) ~
      ("pretrained_embeddings_file" -> "/efs/data/dlfa/glove/glove.6B.100d.txt.gz") ~
      ("fine_tune_embeddings" -> false) ~
      ("project_embeddings" -> false) ~
      ("num_gated_attention_layers" -> 3) ~
      ("patience" -> 0) ~
      ("preferred_backend" -> "theano") ~
      ("encoder" ->
        ("question_final" ->
          ("type" -> "bi_gru")~
          ("output_dim" -> 128)
        )
      ) ~
      ("seq2seq_encoder" ->
        ("question_0" ->
          ("type" -> "bi_gru") ~
          ("encoder_params" ->
            ("output_dim" -> 128)
          ) ~
          ("wrapper_params" -> JObject())
        ) ~
        ("document_0" ->
          ("type" -> "bi_gru") ~
          ("encoder_params" ->
            ("output_dim" -> 128)
          ) ~
          ("wrapper_params" -> JObject())
        ) ~
        ("question_1" ->
          ("type" -> "bi_gru") ~
          ("encoder_params" ->
            ("output_dim" -> 128)
          ) ~
          ("wrapper_params" -> JObject())
        ) ~
        ("document_1" ->
          ("type" -> "bi_gru") ~
          ("encoder_params" ->
            ("output_dim" -> 128)
          ) ~
          ("wrapper_params" -> JObject())
        ) ~
        ("document_final" ->
          ("type" -> "bi_gru") ~
          ("encoder_params" ->
            ("output_dim" -> 128)
          ) ~
          ("wrapper_params" -> JObject())
        )
      ) ~
      ("optimizer" ->
        ("type" -> "adam") ~
        ("clipnorm" -> 10.0) ~
        ("lr" -> 0.0005)
      ) ~
      ("embedding_dropout" -> 0.0) ~
      ("patience" -> 0) ~
      ("num_epochs" -> 5)
    )
  }


  def main(args: Array[String]) {
    runASReaderOmnibusEightExperiment
    runASReaderOmnibusFourExperiment
    runGAReaderOmnibusEightExperiment
    runGAReaderOmnibusFourExperiment
  }

  def runASReaderOmnibusFourExperiment() {
    val asReaderOmnibusFourDefault = omnibusGradeFourExperiment(
      "ASReader omnibus four",
      attentionSumReader,
      omnibusFourTrainReadingComprehensionDataset
    )

    val asReaderOmnibusFourWithSciQDataset = omnibusGradeFourExperiment(
      "ASReader omnibus four plus SciQ Dataset",
      attentionSumReader,
      combinedSciQTrainOmnibusFourTrainDataset
    )

    val models = Seq(asReaderOmnibusFourDefault, asReaderOmnibusFourWithSciQDataset)
    new Evaluator(Some("ASReader_omnibus_four_plus_sciq_dataset"), models, fileUtil).runPipeline()
  }

  def runASReaderOmnibusEightExperiment() {
    val asReaderOmnibusEightDefault = omnibusGradeEightExperiment(
      "ASReader omnibus eight",
      attentionSumReader,
      omnibusEightTrainReadingComprehensionDataset
    )

    val asReaderOmnibusEightWithSciQDataset = omnibusGradeEightExperiment(
      "ASReader omnibus eight plus SciQ Dataset",
      attentionSumReader,
      combinedSciQTrainOmnibusEightTrainDataset
    )

    val models = Seq(asReaderOmnibusEightDefault, asReaderOmnibusEightWithSciQDataset)
    new Evaluator(Some("ASReader_omnibus_eight_plus_sciq_dataset"), models, fileUtil).runPipeline()
  }

    def runGAReaderOmnibusFourExperiment() {
    val gaReaderOmnibusFourDefault = omnibusGradeFourExperiment(
      "GAReader omnibus four",
      gatedAttentionReader,
      omnibusFourTrainReadingComprehensionDataset
    )

    val gaReaderOmnibusFourWithSciQDataset = omnibusGradeFourExperiment(
      "GAReader omnibus four plus SciQ Dataset",
      gatedAttentionReader,
      combinedSciQTrainOmnibusFourTrainDataset
    )

    val models = Seq(gaReaderOmnibusFourDefault, gaReaderOmnibusFourWithSciQDataset)
    new Evaluator(Some("GAReader_omnibus_four_plus_sciq_dataset"), models, fileUtil).runPipeline()
  }

  def runGAReaderOmnibusEightExperiment() {
    val gaReaderOmnibusEightDefault = omnibusGradeEightExperiment(
      "GAReader omnibus eight",
      gatedAttentionReader,
      omnibusEightTrainReadingComprehensionDataset
    )

    val gaReaderOmnibusEightWithSciQDataset = omnibusGradeEightExperiment(
      "GAReader omnibus eight plus SciQ Dataset",
      gatedAttentionReader,
      combinedSciQTrainOmnibusEightTrainDataset
    )

    val models = Seq(gaReaderOmnibusEightDefault, gaReaderOmnibusEightWithSciQDataset)
    new Evaluator(Some("GAReader_omnibus_eight_plus_sciq_dataset"), models, fileUtil).runPipeline()
  }

}
