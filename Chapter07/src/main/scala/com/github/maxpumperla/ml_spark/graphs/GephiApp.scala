package com.github.maxpumperla.ml_spark.graphs

import java.io.PrintWriter

import com.github.maxpumperla.ml_spark.utils.Gephi.toGexf
import org.apache.spark._
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD

object GephiApp extends App {

    val conf = new SparkConf()
      .setAppName("Gephi Test Writer")
      .setMaster("local[4]")
    val sc = new SparkContext(conf)

    val vertices: RDD[(VertexId, String)] = sc.parallelize(
      Array((1L, "Anne"),
        (2L, "Bernie"),
        (3L, "Chris"),
        (4L, "Don"),
        (5L, "Edgar")))

    val edges: RDD[Edge[String]] = sc.parallelize(
      Array(Edge(1L, 2L, "likes"),
        Edge(2L, 3L, "trusts"),
        Edge(3L, 4L, "believes"),
        Edge(4L, 5L, "worships"),
        Edge(1L, 3L, "loves"),
        Edge(4L, 1L, "dislikes")))

    val graph: Graph[String, String] = Graph(vertices, edges)

    val pw = new PrintWriter("./graph.gexf")
    pw.write(toGexf(graph))
    pw.close()
}


