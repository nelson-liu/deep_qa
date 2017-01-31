package org.allenai.deep_qa.experiments.datasets

import org.json4s._
import org.json4s.JsonDSL._

/**
 * This object contains a bunch of JValue specifications for Johannes' datasets' data files.
 */
object JohannesDatasets {

  def johannesFile(johannesDir: String, split: String, version: String="1.0"): JValue = {
    val outputDirectory = johannesDir + "processed/"
    val inputFile = johannesDir + s"v${version}/"+ s"${split}-v${version}.json"
    val outputFiles = Seq(outputDirectory + s"${split}.tsv")
    ("sentence producer type" -> "dataset reader") ~
    ("reader" -> "johannes") ~
    ("create sentence indices" -> true) ~
    ("input file" -> inputFile) ~
    ("output files" -> outputFiles)
  }

  def johannesDataset(johannesDir: String, split: String, version: String="1.0"): JValue = {
    val file = johannesFile(johannesDir, split, version)
    ("data files" -> List(file))
  }

  def makeMCFile(sentence_params: JValue): JValue = {
    ("sentences" -> sentence_params) ~
    ("sentence producer type" -> "drop first column")
  }

  def makePassageBackgroundFile(
    sentences: JValue,
    sentenceFormat: String,
    corpus: JValue
  ) : JValue = {
    ("sentence producer type" -> "background searcher") ~
      ("searcher" -> corpus) ~
      ("sentences" -> sentences) ~
      ("sentence format" -> sentenceFormat)
  }

  val baseDir = "/efs/data/dlfa/turk_johannes_questions/"

  val trainFile = johannesFile(baseDir, "train")
  val train = johannesDataset(baseDir, "train")
  val mcTrainFile = makeMCFile(trainFile)
  val mcTrainBuscBackgroundFile = makePassageBackgroundFile(
    mcTrainFile,
    "question and answer",
    ScienceCorpora.buscElasticSearchIndex(3)
  )
  val readingComprehensionTrainWithBuscBackgroundFile: JValue =
    ("sentence producer type" -> "qa and background to mc") ~
    ("sentences" -> mcTrainFile) ~
    ("background" -> mcTrainBuscBackgroundFile)

  val devFile = johannesFile(baseDir, "dev")
  val dev = johannesDataset(baseDir, "dev")
  val mcDevFile = makeMCFile(devFile)
  val mcDevBuscBackgroundFile = makePassageBackgroundFile(
    mcDevFile,
    "question and answer",
    ScienceCorpora.buscElasticSearchIndex(3)
  )
  val readingComprehensionDevWithBuscBackgroundFile: JValue =
    ("sentence producer type" -> "qa and background to mc") ~
    ("sentences" -> mcDevFile) ~
    ("background" -> mcDevBuscBackgroundFile)

  val testFile = johannesFile(baseDir, "test")
  val test = johannesDataset(baseDir, "test")
  val mcTestFile = makeMCFile(testFile)
  val mcTestBuscBackgroundFile = makePassageBackgroundFile(
    mcTestFile,
    "question and answer",
    ScienceCorpora.buscElasticSearchIndex(3)
  )
  val readingComprehensionTestWithBuscBackgroundFile: JValue =
    ("sentence producer type" -> "qa and background to mc") ~
    ("sentences" -> mcTestFile) ~
    ("background" -> mcTestBuscBackgroundFile)

  val mcTrainWithBuscBackground: JValue = ScienceDatasets.makeBackgroundDataset(
    mcTrainFile,
    "question and answer",
    ScienceCorpora.buscElasticSearchIndex(3)
  )

  val mcDevWithBuscBackground: JValue = ScienceDatasets.makeBackgroundDataset(
    mcDevFile,
    "question and answer",
    ScienceCorpora.buscElasticSearchIndex(3)
  )

  val mcTestWithBuscBackground: JValue = ScienceDatasets.makeBackgroundDataset(
    mcTestFile,
    "question and answer",
    ScienceCorpora.buscElasticSearchIndex(3)
  )
}
