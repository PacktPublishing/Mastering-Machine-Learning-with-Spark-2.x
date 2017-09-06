package com.github.maxpumperla.ml_spark.streaming

import org.apache.spark.mllib.fpm.{FPGrowth, PrefixSpan}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}


object MSNBCPatternMining extends App {

    val conf = new SparkConf()
      .setAppName("MSNBC.com data pattern mining")
      .setMaster("local[4]")
    val sc = new SparkContext(conf)

    val transactionTest = sc.parallelize(Array(Array("A", "B", "C"), Array("B", "C", "A")))
    val fp = new FPGrowth().setMinSupport(0.8).setNumPartitions(5)
    fp.run(transactionTest)

    val transactions: RDD[Array[Int]] = sc.textFile("./msnbc990928.seq") map { line =>
      line.split(" ").map(_.toInt)
    }

    // NOTE: Caching data is recommended
    val uniqueTransactions: RDD[Array[Int]] = transactions.map(_.distinct).cache()


    val fpGrowth = new FPGrowth().setMinSupport(0.01)
    val model = fpGrowth.run(uniqueTransactions)
    val count = uniqueTransactions.count()

    model.freqItemsets.collect().foreach { itemset =>
      if (itemset.items.length >= 3)
        println(itemset.items.mkString("[", ",", "]") + ", " + itemset.freq / count.toDouble )
    }

    val rules = model.generateAssociationRules(confidence = 0.4)
    rules.collect().foreach { rule =>
      println("[" + rule.antecedent.mkString(",") + "=>"
        + rule.consequent.mkString(",") + "]," + (100 * rule.confidence).round / 100.0)
    }

    val frontPageConseqRules = rules.filter(_.consequent.head == 1)
    frontPageConseqRules.count
    frontPageConseqRules.filter(_.antecedent.contains(2)).count
    rules.filter(_.antecedent.contains(7)).count


    val sequences: RDD[Array[Array[Int]]] = transactions.map(_.map(Array(_))).cache()

    val prefixSpan = new PrefixSpan().setMinSupport(0.005).setMaxPatternLength(15)
    val psModel = prefixSpan.run(sequences)

    psModel.freqSequences.map(fs => (fs.sequence.length, 1))
      .reduceByKey(_ + _)
      .sortByKey()
      .collect()
      .foreach(fs => println(s"${fs._1}: ${fs._2}"))

    psModel.freqSequences
      .map(fs => (fs.sequence.length, fs))
      .groupByKey()
      .map(group => group._2.reduce((f1, f2) => if (f1.freq > f2.freq) f1 else f2))
      .map(_.sequence.map(_.mkString("[", ", ", "]")).mkString("[", ", ", "]"))
      .collect.foreach(println)


    psModel.freqSequences
      .map(fs => (fs.sequence.map(_.mkString("[", ", ", "]")).mkString("[", ", ", "]"), 1))
      .reduceByKey(_ + _)
      .reduce( (f1, f2) => if (f1._2 > f2._2) f1 else f2 )


    psModel.freqSequences.reduce( (f1, f2) => if (f1.freq > f2.freq) f1 else f2 )
    psModel.freqSequences.filter(_.sequence.length == 1).map(_.sequence.toString).collect.foreach(println)

    psModel.freqSequences.collect().foreach {
      freqSequence =>
        println(
          freqSequence.sequence.map(_.mkString("[", ", ", "]")).mkString("[", ", ", "]") + ", " + freqSequence.freq
        )
    }
}
