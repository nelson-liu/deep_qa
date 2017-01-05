package org.allenai.deep_qa.data

/**
 * A single training / testing example.
 */
trait Instance {
  val label: Option[Any]

  /**
   * Converts the instance into a sequence of strings.  The return type here is unfortunately very
   * complex, in order to handle a wide variety of possible ways that Instances can be written to
   * disk.  First, some Instance types (like BackgroundInstances) get written to multiple files, so
   * we need a Seq[] to be able to handle that.  Second, some Instance types (like
   * MultipleTrueFalseInstances) get written to multiple lines in a file, and because we need to be
   * able to number the lines, we need to return a Seq[] to handle this correctly.
   *
   * So, the length of the outer Seq[] is the number of files needed to write this Instance to
   * disk, and the length of the inner Seq[] is the number of lines to write for this Instance.
   * The length of all inner Seqs must match, or the numbering will not work correctly.
   */
  def asStrings(): Seq[Seq[String]]
}

/**
 * An Instance that has a single true/false statement.
 */
case class TrueFalseInstance(
  statement: String,
  override val label: Option[Boolean]
) extends Instance {
  def asStrings(): Seq[Seq[String]] = {
    label match {
      case Some(true) => Seq(Seq(s"$statement\t1"))
      case Some(false) => Seq(Seq(s"$statement\t0"))
      case None => Seq(Seq(s"$statement"))
    }
  }
}

/**
 * An Instance that combines multiple true/false instances, where exactly one of them has label
 * true.  The label in this Instance is the index to the one whose label is true.
 */
case class MultipleTrueFalseInstance[T <: Instance](
  instances: Seq[Instance],
  override val label: Option[Int]
) extends Instance {

  /**
   * Each contained instance returns its own Seq[Seq[String]].  We just flatten together all of the
   * inner Seq[Strings], and return a Seq[Seq[String]] with all of the instances combined.
   */
  def asStrings(): Seq[Seq[String]] = {
    instances.map(_.asStrings()).transpose.map(_.flatten)
  }
}

/**
 * An Instance that has question text and several answer options.
 */
case class QuestionAnswerInstance(
  question: String,
  answers: Seq[String],
  override val label: Option[Seq[Int]]
) extends Instance {
  def asStrings(): Seq[Seq[String]] = {
    val answerString = answers.mkString("###")
    label match {
      case Some(l) => Seq(Seq(s"$question\t$answerString\t${l.mkString(",")}"))
      case None => Seq(Seq(s"$question\t$answerString"))
    }
  }
}

/**
 * An Instance that wraps another Instance and adds background information.
 */
case class BackgroundInstance[T <: Instance](
  containedInstance: T,
  background: Seq[String]
) extends Instance {
  def asStrings(): Seq[Seq[String]] = {
    val backgroundString = background.mkString("\t")
    containedInstance.asStrings() ++ Seq(Seq(backgroundString))
  }

  override val label = containedInstance.label
}

/**
 * An Instance created from the Stanford Natural Language Inference corpus.
 */
case class SnliInstance(
  text: String,
  hypothesis: String,
  override val label: Option[String]
) extends Instance {
  def asStrings(): Seq[Seq[String]] = {
    label match {
      case Some(l) => Seq(Seq(s"$text\t$hypothesis\t$l"))
      case None => Seq(Seq(s"$text\t$hypothesis"))
    }
  }
}

/**
  * An Instance created from the Stanford Question Answering Dataset (SQuAD).
  */
case class SquadInstance(
  question: String,
  passage: String,
  override val label: Option[(Int, Int)]
) extends Instance {
  def asStrings(): Seq[Seq[String]] = {
    val labelString = (label.map { case (start, end) => s"\t${start},${end}" }).getOrElse("")
    Seq(Seq(s"${question}\t${passage}${labelString}"))
  }
}

/**
  * An Instance created from the Who Did What (WDW) dataset.
  */
case class WDWInstance(
  passage: String,
  leftContext: String,
  rightContext: String,
  answers: Seq[String],
  override val label: Option[Int]
) extends Instance {
  def asStrings(): Seq[Seq[String]] = {
    val answerString = answers.mkString("###")
    label match {
      case Some(l) => Seq(Seq(s"$passage\t$leftContext\t$rightContext\t$answerString\t$l"))
      case None => Seq(Seq(s"$passage\t$leftContext\t$rightContext\t$answerString"))
    }
  }
}
