package com.github.maxpumperla.ml_spark.graphs

import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}


/**
  * Created by maxpumperla on 02/06/17.
  */
object GraphFromRdd extends App {

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
     friendGraph.vertices.collect.foreach(println)

     friendGraph.edges.map( e => e.srcId > e.dstId ).count()

     val mappedEdgeGraph: Graph[String, Boolean] = friendGraph.mapEdges( e => e.srcId > e.dstId )

     val inDegVertexRdd: VertexRDD[Int] = friendGraph.aggregateMessages[Int](
       sendMsg = ec => ec.sendToDst(1),
       mergeMsg = (msg1, msg2) => msg1+msg2
     )
     assert(inDegVertexRdd.collect.deep == friendGraph.inDegrees.collect.deep)

     friendGraph.staticPageRank(numIter = 10).vertices.collect.foreach(println)
     friendGraph.pageRank(tol = 0.0001, resetProb = 0.15)

}
