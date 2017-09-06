package com.github.maxpumperla.ml_spark.graphs

import org.apache.spark.graphx.lib.TriangleCount
import org.apache.spark.graphx.util.GraphGenerators
import org.apache.spark.graphx.{Graph, GraphLoader, PartitionStrategy, VertexId}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by maxpumperla on 02/06/17.
  */
object GraphGeneration extends App {

  val conf = new SparkConf()
    .setAppName("Graph generation")
    .setMaster("local[4]")
  val sc = new SparkContext(conf)

  val edgeListGraph = GraphLoader.edgeListFile(sc, "./edge_list.txt")

  val rawEdges: RDD[(VertexId, VertexId)] = sc.textFile("./edge_list.txt").map {
    line =>
      val field = line.split(" ")
      (field(0).toLong, field(1).toLong)
  }
  val edgeTupleGraph = Graph.fromEdgeTuples(
    rawEdges=rawEdges, defaultValue="")

  val gridGraph = GraphGenerators.gridGraph(sc, 5, 5)
  val starGraph = GraphGenerators.starGraph(sc, 11)
  val logNormalGraph  = GraphGenerators.logNormalGraph(
    sc, numVertices = 20, mu=1, sigma = 3
  )
  logNormalGraph.outDegrees.map(_._2).collect().sorted

  val actorGraph = GraphLoader.edgeListFile(
    sc, "./ca-hollywood-2009.txt", true
  ).partitionBy(PartitionStrategy.RandomVertexCut)
  actorGraph.edges.count()

  val actorComponents = actorGraph.connectedComponents().cache
  actorComponents.vertices.map(_._2).distinct().count

  val clusterSizes =actorComponents.vertices.map(
    v => (v._2, 1)).reduceByKey(_ + _)
  clusterSizes.map(_._2).max
  clusterSizes.map(_._2).min

  val smallActorGraph = GraphLoader.edgeListFile(sc, "./ca-hollywood-2009.txt")
  val strongComponents = smallActorGraph.stronglyConnectedComponents(numIter = 5)
  strongComponents.vertices.map(_._2).distinct().count

  val canonicalGraph = actorGraph.mapEdges(e => 1).removeSelfEdges().convertToCanonicalEdges()
  val partitionedGraph = canonicalGraph.partitionBy(PartitionStrategy.RandomVertexCut)

  actorGraph.triangleCount()
  val triangles = TriangleCount.runPreCanonicalized(partitionedGraph)

  actorGraph.staticPageRank(10)
  val actorPrGraph: Graph[Double, Double] = actorGraph.pageRank(0.0001)
  actorPrGraph.vertices.reduce((v1, v2) => {
    if (v1._2 > v2._2) v1 else v2
  })

  actorPrGraph.inDegrees.filter(v => v._1 == 33024L).collect.foreach(println)

  actorPrGraph.inDegrees.map(_._2).collect().sorted.takeRight(10)

  actorPrGraph.inDegrees.map(_._2).filter(_ >= 62).count

}
