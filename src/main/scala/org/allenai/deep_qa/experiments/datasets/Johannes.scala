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
    ("input file" -> inputFile) ~
    ("output files" -> outputFiles)
  }

  def johannesDataset(johannesDir: String, split: String, version: String="1.0"): JValue = {
    val file = johannesFile(johannesDir, split, version)
    ("data files" -> List(file))
  }

  val baseDir = "/efs/data/dlfa/turk_johannes_questions/"

  val trainFile = johannesFile(baseDir, "train")
  val trainDataset = johannesDataset(baseDir, "train")
  val devFile = johannesFile(baseDir, "dev")
  val devDataset = johannesDataset(baseDir, "dev")
  val testFile = johannesFile(baseDir, "test")
  val test = johannesDataset(baseDir, "test")
}
