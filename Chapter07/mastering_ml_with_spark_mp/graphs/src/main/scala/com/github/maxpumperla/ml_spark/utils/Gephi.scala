package com.github.maxpumperla.ml_spark.utils

import org.apache.spark.graphx.Graph

/**
  * Created by maxpumperla on 01/06/17.
  */
object Gephi {

  def toGexf[VD, ED](g: Graph[VD, ED]): String = {
    val header =
      """<?xml version="1.0" encoding="UTF-8"?>
        |<gexf xmlns="http://www.gexf.net/1.2draft" version="1.2">
        |  <meta>
        |    <description>A gephi graph in GEXF format</description>
        |  </meta>
        |    <graph mode="static" defaultedgetype="directed">
      """.stripMargin

    val vertices = "<nodes>\n" + g.vertices.map(
      v => s"""<node id=\"${v._1}\" label=\"${v._2}\"/>\n"""
    ).collect.mkString + "</nodes>\n"

    val edges = "<edges>\n" + g.edges.map(
      e => s"""<edge source=\"${e.srcId}\" target=\"${e.dstId}\" label=\"${e.attr}\"/>\n"""
    ).collect.mkString + "</edges>\n"

    val footer = "</graph>\n</gexf>"

    header + vertices + edges + footer
  }
}
