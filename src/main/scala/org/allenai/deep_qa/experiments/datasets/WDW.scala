package org.allenai.deep_qa.experiments.datasets

import org.json4s._
import org.json4s.JsonDSL._

/**
 * This object contains a bunch of JValue specifications for WDW data files.
 */
object WDWDatasets {

  def wdwFile(wdwDir: String, split: String): JValue = {
    val outputDirectory = wdwDir + "processed/"
    val inputFile = wdwDir + s"${split}.xml"
    val outputFiles = Seq(outputDirectory + s"${split}.tsv")
    ("sentence producer type" -> "dataset reader") ~
    ("reader" -> "wdw") ~
    ("input file" -> inputFile) ~
    ("output files" -> outputFiles)
  }

  def wdwDataset(wdwDir: String, split: String): JValue = {
    val file = wdwFile(wdwDir, split)
    ("data files" -> List(file))
  }

  val baseDir = "/efs/data/dlfa/who_did_what/Strict/"

  val trainFile = wdwFile(baseDir, "train")
  val train = wdwDataset(baseDir, "train")
  val devFile = wdwFile(baseDir, "dev")
  val dev = wdwDataset(baseDir, "dev")
}
