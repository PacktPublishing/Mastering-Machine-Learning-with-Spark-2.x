package com.github.maxpumperla.ml_spark.streaming

import org.apache.spark.mllib.fpm.PrefixSpan
import org.apache.spark.rdd.RDD
import org.apache.spark.streaming.dstream.DStream
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.spark.{SparkConf, SparkContext}

object MSNBCStreamingExample extends App {

    val conf = new SparkConf()
      .setAppName("MSNBC data initial streaming example")
      .setMaster("local[4]")
    val sc = new SparkContext(conf)
    val ssc = new StreamingContext(sc, batchDuration = Seconds(10))

    val transactions: RDD[Array[Int]] = sc.textFile("src/main/resources/msnbc990928.seq") map { line =>
      line.split(" ").map(_.toInt)
    }
    val trainSequences: RDD[Array[Array[Int]]] = transactions.map(_.map(Array(_))).cache()
    val prefixSpan = new PrefixSpan().setMinSupport(0.005).setMaxPatternLength(15)
    val psModel = prefixSpan.run(trainSequences)
    val freqSequences = psModel.freqSequences.map(_.sequence).collect()


    val rawSequences: DStream[String] = ssc.socketTextStream("localhost", 9999)

    val sequences: DStream[Array[Array[Int]]] = rawSequences
      .map(line => line.split(" ").map(_.toInt))
      .map(_.map(Array(_)))

    print(">>> Analysing new batch of data")
    sequences.foreachRDD(
      rdd => rdd.foreach(
        array => {
          println(">>> Sequence: ")
          println(array.map(_.mkString("[", ", ", "]")).mkString("[", ", ", "]"))
          freqSequences.count(_.deep == array.deep) match {
            case count if count > 0 => println("is frequent!")
            case _ => println("is not frequent.")
          }
        }
      )
    )
    print(">>> done")

    ssc.start()
    ssc.awaitTermination()

}
