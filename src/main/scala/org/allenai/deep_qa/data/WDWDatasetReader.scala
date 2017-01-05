package org.allenai.deep_qa.data

import com.mattg.util.FileUtil

import scala.collection.mutable
import scala.xml


class WDWDatasetReader(fileUtil: FileUtil) extends DatasetReader[WDWInstance] {
  override def readFile(filename: String): Dataset[WDWInstance] = {
    val xml = scala.xml.Utility.trim(scala.xml.XML.loadString(fileUtil.readFileContents(filename)))
    val instanceTuples = for {
      mc <- xml \ "mc"
      question_node = mc \ "question"
      question_text = question_node.text

      leftContext_node = question_node \ "leftcontext"
      leftContext = leftContext_node.text

      rightContext_node = question_node \ "rightcontext"
      rightContext = rightContext_node.text

      passage_node = mc \ "contextart"
      passage = passage_node.text

      answer_nodes = mc \ "choice"
      answers = answer_nodes.map((answer: scala.xml.Node) => answer.text)

      label_nodes = answer_nodes.filter((answer: scala.xml.Node) => (answer \ "@correct").text == "true")
      label = Integer.parseInt((label_nodes \ "@idx").text)
    } yield (passage, leftContext, rightContext, answers, label)

    val instances = instanceTuples.map { case (passage, leftContext, rightContext, answers, label) => {
      WDWInstance(passage, leftContext, rightContext, answers, Some(label))
    }}
    Dataset(instances)
  }
}
