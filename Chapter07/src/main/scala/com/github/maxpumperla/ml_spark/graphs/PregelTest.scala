package com.github.maxpumperla.ml_spark.graphs

import org.apache.spark.graphx._
import org.apache.spark.graphx.util.GraphGenerators
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by maxpumperla on 03/06/17.
  */
object PregelTest extends App {

  val conf = new SparkConf()
    .setAppName("Graph generation")
    .setMaster("local[4]")
  val sc = new SparkContext(conf)

  val graph = GraphGenerators.gridGraph(sc, 5, 5)

  val g = Pregel(graph.mapVertices((vid,vd) => 0), 0, activeDirection = EdgeDirection.Out)(
    (id:VertexId,vd:Int,a:Int) => math.max(vd,a),
    (et:EdgeTriplet[Int,Double]) => Iterator((et.dstId, et.srcAttr+1)),
    (a:Int,b:Int) => math.max(a,b))
  g.vertices.collect

}
