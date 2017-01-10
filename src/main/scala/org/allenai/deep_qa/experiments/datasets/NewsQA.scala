package org.allenai.deep_qa.experiments.datasets

import org.json4s._
import org.json4s.JsonDSL._

/**
 * This object contains a bunch of JValue specifications for Newsqa data files.
 */
object NewsQADatasets {

  def newsQAFile(newsQADir: String, split: String): JValue = {
    val outputDirectory = newsQADir + "processed/"
    val inputFile = newsQADir + s"${split}.json"
    val outputFiles = Seq(outputDirectory + s"${split}.tsv")
    ("sentence producer type" -> "dataset reader") ~
    ("reader" -> "newsQA") ~
    ("input file" -> inputFile) ~
    ("output files" -> outputFiles)
  }

  def newsQADataset(newsQADir: String, split: String): JValue = {
    val file = newsQAFile(newsQADir, split)
    ("data files" -> List(file))
  }

  val baseDir = "/efs/data/dlfa/newsQA/split_data/"

  val trainFile = newsQAFile(baseDir, "train")
  val trainDataset = newsQADataset(baseDir, "train")
  val devFile = newsQAFile(baseDir, "dev")
  val devDataset = newsQADataset(baseDir, "dev")
}
