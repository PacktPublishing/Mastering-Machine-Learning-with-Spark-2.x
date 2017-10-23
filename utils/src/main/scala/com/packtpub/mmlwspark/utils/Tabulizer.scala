package com.packtpub.mmlwspark.utils

import org.apache.spark.mllib.linalg.Vector
/**
  * A simple console based table.
  */
object Tabulizer {
  def table(cells: Seq[Product]): String =
    table(cells.map(p => p.productIterator.toList), false)

  def table(header: Seq[String], cells: Seq[Product], format: Map[Int, String]): String =
    table(Seq(header) ++ cells.map(p => {
      p.productIterator.toList.zipWithIndex.map { case (v, i) =>
          format.get(i).map(f => String.format(f, v.asInstanceOf[Object])).getOrElse(v)
      }
    }), true)

  def table[A, B](cells: scala.collection.Map[A, B]): String = {
    val header = cells.keys.toSeq
    val values = header.map(cells(_))
    table(Seq(header) ++ Seq(values), true)
  }

  /*def table(header: Seq[String], cells: Seq[Seq[Any]]): String =
    table(Seq(header) ++ cells, header = true)*/

  def table(vector: Vector, cols: Int, format: String = "%.3f"): String =
    table(vector.toArray.map(format.format(_)), cols, None)

  def table(list: Seq[Any], cols: Int, header: Option[Seq[String]]): String =
    table(tblize(header.map(_ ++ list).getOrElse(list), cols), header.isDefined)

  def table(cells: Seq[Seq[Any]], header: Boolean): String = {
    val colSizes = cells
      .map(_.map(v => if (v != null) v.toString.length else 1))
      .reduce((v1, v2) => v1.zip(v2).map { case (v1, v2) => if (v1 > v2) v1 else v2 })
    val rowSeparator = colSizes.map("-" * _).mkString("+", "+", "+")
    def valueFormatter(v: Any, size: Int): String =
      ("%" + size + "s").format(if (v != null) v else "-")
    val rows = cells
      .map(row => row.zip(colSizes)
        .map { case (v, size) => valueFormatter(v, size) }.mkString("|", "|", "|"))
    if (header)
      s"""
         #$rowSeparator
         #${rows.head}
         #$rowSeparator
         #${rows.tail.mkString("\n")}
         #$rowSeparator
      """.stripMargin('#')
    else
      s"""
         #$rowSeparator
         #${rows.mkString("\n")}
         #$rowSeparator
      """.stripMargin('#')
  }

  def tblize(list: Seq[Product], horizontal: Boolean, cols: Int): Seq[Seq[Any]] = {
    val arity = list.head.productArity
    tblize(list.flatMap(_.productIterator.toList), cols = arity * cols)
  }

  def tblize(list: Seq[Any], cols: Int = 4): Seq[Seq[Any]] = {
    val nrow = list.length / cols + (if (list.length % cols == 0) 0 else 1)
    list.sliding(cols, cols)
      .map(s => if (s.length == cols || s.length == list.length) s else s.padTo(cols, null))
      .foldLeft(Seq[Seq[Any]]()) { case (a, s) => a ++ Seq(s) }
  }
}
