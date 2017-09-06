package com.github.maxpumperla.ml_spark.graphs

import org.apache.spark.graphx._

import scala.reflect.ClassTag

object ConnectedComponents extends Serializable with App {

  def run[VD: ClassTag, ED: ClassTag](graph: Graph[VD, ED],
                                      maxIterations: Int)
  : Graph[VertexId, ED] = {

    val idGraph: Graph[VertexId, ED] = graph.mapVertices((id, _) => id)

    def vprog(id: VertexId, attr: VertexId, msg: VertexId): VertexId = {
      math.min(attr, msg)
    }

    def sendMsg(edge: EdgeTriplet[VertexId, ED]): Iterator[(VertexId, VertexId)] = {
      if (edge.srcAttr < edge.dstAttr) {
        Iterator((edge.dstId, edge.srcAttr))
      } else if (edge.srcAttr > edge.dstAttr) {
        Iterator((edge.srcId, edge.dstAttr))
      } else {
        Iterator.empty
      }
    }

    def mergeMsg(v1: VertexId, v2: VertexId): VertexId = {
      math.min(v1, v2)
    }

    Pregel(
      graph = idGraph,
      initialMsg = Long.MaxValue,
      maxIterations,
      EdgeDirection.Either)(
      vprog,
      sendMsg,
      mergeMsg)
  }

  def run[VD: ClassTag, ED: ClassTag](graph: Graph[VD, ED])
  : Graph[VertexId, ED] = {
    run(graph, Int.MaxValue)
  }

}


