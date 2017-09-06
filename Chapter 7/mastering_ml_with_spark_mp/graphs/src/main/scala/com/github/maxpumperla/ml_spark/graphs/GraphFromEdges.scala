package com.github.maxpumperla.ml_spark.graphs

import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by maxpumperla on 12/06/17.
  */
object GraphFromEdges extends App {

  val conf = new SparkConf()
    .setAppName("Graph generation")
    .setMaster("local[4]")
  val sc = new SparkContext(conf)

  val edges: RDD[Edge[Long]] =
    sc.textFile("./rt_occupywallstnyc.edges").map { line =>
      val fields = line.split(",")
      Edge(fields(0).toLong, fields(1).toLong, fields(2).toLong)
    }
  val rtGraph: Graph[String, Long] = Graph.fromEdges(edges, defaultValue =  "")

  val order = rtGraph.numVertices
  val degree = rtGraph.numEdges

  val avgDegree = rtGraph.degrees.map(_._2).reduce(_ + _) / order.toDouble

  val vertexDegrees: VertexRDD[Int] = rtGraph.degrees
  val degrees: RDD[Int] = vertexDegrees.map(v => v._2)
  val sumDegrees: Int = degrees.reduce((v1, v2) => v1 + v2 )
  val avgDegreeAlt = sumDegrees / order.toDouble

  assert(avgDegree == avgDegreeAlt)

  val maxInDegree: (Long, Int) = rtGraph.inDegrees.reduce(
    (v1,v2) => if (v1._2 > v2._2) v1 else v2
  )

  rtGraph.edges.filter(e => e.dstId == 1783).map(_.srcId).distinct()

  val minOutDegree: (Long, Int) = rtGraph.outDegrees.reduce(
    (v1,v2) => if (v1._2 < v2._2) v1 else v2
  )

  val triplets: RDD[EdgeTriplet[String, Long]] = rtGraph.triplets

  val tweetStrings = triplets.map(
    t => t.dstId + " retweeted " + t.attr + " from " + t.srcId
  )
  tweetStrings.take(5).foreach(println)

  val vertexIdData: Graph[Long, Long] = rtGraph.mapVertices( (id, _) => id)

  val mappedTripletsGraph = rtGraph.mapTriplets(
    t => t.dstId + " retweeted " + t.attr + " from " + t.srcId
  )

  val connectedRt: Graph[VertexId, Long] = rtGraph.connectedComponents()

  val outDegreeGraph: Graph[Long, Long] =
    rtGraph.outerJoinVertices[Int, Long](rtGraph.outDegrees)(
      mapFunc = (id, origData, outDeg) => outDeg.getOrElse(0).toLong
    )

  val tenOrMoreRetweets = outDegreeGraph.subgraph(
    vpred = (id, deg) => deg >= 10
  )
  tenOrMoreRetweets.vertices.count
  tenOrMoreRetweets.edges.count

  val lessThanTenRetweets = rtGraph.mask(tenOrMoreRetweets)


}
