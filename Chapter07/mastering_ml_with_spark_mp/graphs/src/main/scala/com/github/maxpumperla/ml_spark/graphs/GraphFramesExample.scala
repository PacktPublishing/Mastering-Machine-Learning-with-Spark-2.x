package com.github.maxpumperla.ml_spark.graphs

import org.apache.spark.graphx.{Edge, Graph, VertexId}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
//import org.graphframes._

/**
  * Created by maxpumperla on 07/06/17.
  */
object GraphFramesExample extends App {

    val conf = new SparkConf()
      .setAppName("RDD graph")
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

    val friendGraph: Graph[String, String] = Graph(vertices, edges)

//    val friendGraphFrame = GraphFrame.fromGraphX(friendGraph)
//
//    friendGraphFrame.find("(v1)-[e1]->(v2); (v2)-[e2]->(v3)").filter(
//      "e1.attr = 'trusts' OR v3.attr = 'Chris'"
//    ).collect.foreach(println)

}
