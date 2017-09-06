package water.book.chap4

/**
  * Chapter 3
  */
object Chapter3 extends App {

  import org.apache.spark.sql.{SQLContext, SparkSession}
  import org.apache.spark.{SparkConf, SparkContext}

  /* Simulate environment of Spark shell */
  val config = new SparkConf()
    .setMaster("local[*]")
    .setAppName("Chapter3")

  val sc = SparkContext.getOrCreate(config)
  val sqlContext = SparkSession.builder().getOrCreate().sqlContext

  script(sc, sqlContext)

  import sqlContext.implicits._

  def script(sc: SparkContext, sqlContext: SQLContext): Unit = {
    // @Snippet
    val path = "../data/*"
    val dataFiles = sc.wholeTextFiles(path)
    println(s"Number of input files: ${dataFiles.count}")

    // @Snippet
    val allColumnNames = Array(
      "timestamp", "activityId", "hr") ++ Array(
      "hand", "chest", "ankle").flatMap(sensor =>
                                          Array(
                                            "temp",
                                            "accel1X", "accel1Y", "accel1Z",
                                            "accel2X", "accel2Y", "accel2Z",
                                            "gyroX", "gyroY", "gyroZ",
                                            "magnetX", "magnetY", "magnetZ",
                                            "orient", "orientX", "orientY", "orientZ").
                                            map(name => s"${sensor}_${name}"))

    // @Snippet
    val ignoredColumns =
    Array(0,
          3 + 13, 3 + 14, 3 + 15, 3 + 16,
          20 + 13, 20 + 14, 20 + 15, 20 + 16,
          37 + 13, 37 + 14, 37 + 15, 37 + 16)

    // @Snippet
    val rawData = dataFiles.flatMap { case (path, content) =>
      content.split("\n")
                                    }.map { row =>
      row.split(" ").
        map(_.trim).
        map(v => if (v.toUpperCase == "NAN") Double.NaN else v.toDouble).
        zipWithIndex.
        collect {
                  case (cell, idx) if !ignoredColumns.contains(idx) => cell
                }
                                          }
    rawData.cache()

    println(s"Number of rows: ${rawData.count}")

    // @Snippet
    val columnNames = allColumnNames.
      zipWithIndex.
      filter { case (_, idx) => !ignoredColumns.contains(idx) }.
      map { case (name, _) => name }

    println(s"Column names:\n${columnNames.mkString(", ")}")

    // @Snippet
    val activitiesMap = Map(
      1 -> "lying", 2 -> "sitting", 3 -> "standing", 4 -> "walking", 5 -> "running",
      6 -> "cycling", 7 -> "Nordic walking", 9 -> "watching TV", 10 -> "computer work",
      11 -> "car driving", 12 -> "ascending stairs", 13 -> "descending stairs", 16 -> "vacuum cleaning",
      17 -> "ironing", 18 -> "folding laundry", 19 -> "house cleaning", 20 -> "playing soccer",
      24 -> "rope jumping", 0 -> "other")


    // @Snippet
    val dataActivityId = rawData.map(l => l(0).toInt)

    val activityIdCounts = dataActivityId.
      map(n => (n, 1)).
      reduceByKey(_ + _)

    val activityCounts = activityIdCounts.
      collect.
      sortBy { case (activityId, count) =>
        -count
             }.map { case (activityId, count) =>
      (activitiesMap(activityId), count)
                   }

    println(s"""|Activities distribution:
                |${activityCounts.mkString("\n")}""".stripMargin)

    // @Snippet
    val nanCountPerRow = rawData.map { row =>
      row.foldLeft(0) { case (acc, v) =>
        acc + (if (v.isNaN) 1 else 0)
                      }
                                     }
    val nanTotalCount = nanCountPerRow.sum
    val ncols = rawData.take(1)(0).length
    val nrows = rawData.count

    val nanRatio = 100.0 * nanTotalCount / (ncols * nrows)

    println(f"""|NaN count = ${nanTotalCount}%.0f
                |NaN ratio = ${nanRatio}%.2f %%""".stripMargin)

    // @Snippet
    val nanRowDistribution = nanCountPerRow.
      map( count => (count, 1)).
      reduceByKey(_ + _).sortBy(-_._1)

    println(s"#NaN / #Rows \n${nanRowDistribution.collect.mkString("\n")}")

    // @Snippet
    val nanRowThreshold = 26
    val badRows = nanCountPerRow.zipWithIndex.zip(rawData).filter(_._1._1 > nanRowThreshold).sortBy(-_._1._1)
    println(s"Bad rows (#NaN, Row Idx, Row):\n${badRows.collect.map(x => (x._1, x._2.mkString(","))).mkString("\n")}")

    // @Snippet
    val nanCountPerColumn = rawData.map { row =>
      row.map(v => if (v.isNaN) 1 else 0)
                                        }.reduce((v1, v2) => v1.indices.map(i => v1(i) + v2(i)).toArray)

    println(s"Number of missing values per column:\n${columnNames.zip(nanCountPerColumn).mkString(", ")}")
    println(s"${columnNames.zip(nanCountPerColumn).map(v => "(%s, %.2f%%)".format(v._1, 100.0 * v._2 / nrows)).mkString(", ")}")

    // @Snippet
    val heartRateColumn = rawData.
      map(row => row(1)).
      filter(!_.isNaN).
      map(_.toInt)

    val heartRateValues = heartRateColumn.collect
    val meanHeartRate = heartRateValues.sum / heartRateValues.size
    scala.util.Sorting.quickSort(heartRateValues)
    val medianHeartRate = heartRateValues(heartRateValues.length / 2)

    println(s"Mean heart rate: ${meanHeartRate}")
    println(s"Median heart rate: ${medianHeartRate}")

    // @Snippet
    // Increment the key-value pair's value based on passed v
    // if v's key exists in l or append v to l.
    def inc[K,V](l: Seq[(K, V)], v: (K, V)) // (3)
                (implicit num: Numeric[V]): Seq[(K,V)] =
    if (l.exists(_._1 == v._1)) l.map(e => e match {
      case (v._1, n) => (v._1, num.plus(n, v._2))
      case t => t
    }) else l ++ Seq(v)

    val distribTemplate = activityIdCounts.collect.map { case (id, _) => (id, 0) }.toSeq
    val nanColumnDistribV1 = rawData.map { row => // (1)
      val activityId = row(0).toInt
      row.drop(1).map { v =>
        if (v.isNaN) inc(distribTemplate, (activityId, 1)) else distribTemplate
                      } // Tip: Make sure that we are returning same type
    }.reduce { (v1, v2) =>  // (2)
      v1.indices.map(idx => v1(idx).foldLeft(v2(idx))(inc)).toArray
    }

    println(s"""|NaN Column x Response distribution:
                |column, ${distribTemplate.map(v => activitiesMap(v._1)).mkString(", ")}
                |${columnNames.drop(1).zip(nanColumnDistribV1).map(v => v._1 + ", " + v._2.map(_._2).mkString(", ")).mkString("\n")}""".stripMargin)

    // @Snippet
    val nanColumnDistribV2 = rawData.map(row => {
      val activityId = row(0).toInt
      (activityId, row.drop(1).map(v => if (v.isNaN) 1 else 0))
    }).reduceByKey( (v1, v2) =>
                      v1.indices.map(idx => v1(idx) + v2(idx)).toArray
    ).map { case (activityId, d) =>
      (activitiesMap(activityId), d)
          }.collect

    println(s"""|NaN Column x Response distribution:
                |${columnNames.mkString(",")}
                |${nanColumnDistribV2.map(v=>v._1 + "," + v._2.mkString(",")).mkString("\n")}""".stripMargin)

    // @Snippet
    val imputedValues = columnNames.map {
                                          _ match {
                                            case "hr" => 60.0
                                            case _ => 0.0
                                          }
                                        }

    // @Snippet
    import org.apache.spark.rdd.RDD
    def imputeNaN(
                   data: RDD[Array[Double]],
                   values: Array[Double]): RDD[Array[Double]] = {
      data.map { row =>
        row.indices.map { i =>
          if (row(i).isNaN) values(i)
          else row(i)
                        }.toArray
               }
    }

    // @Snippet
    def filterBadRows(
                       rdd: RDD[Array[Double]],
                       nanCountPerRow: RDD[Int],
                       nanThreshold: Int): RDD[Array[Double]] = {
      rdd.zip(nanCountPerRow).filter { case (row, nanCount) =>
        nanCount < nanThreshold
                                     }.map { case (row, _) =>
        row
                                           }
    }

    // @Snippet
    val activityId2Idx = activityIdCounts.
      map(_._1).
      collect.
      zipWithIndex.
      toMap

    // @Snippet
    val processedRawData = imputeNaN(
      filterBadRows(rawData, nanCountPerRow, nanThreshold = 26),
      imputedValues)

    // @Snippet
    println(s"Number of rows before/after: ${rawData.count} / ${processedRawData.count}")

    // @Snippet
    import org.apache.spark.mllib
    import org.apache.spark.mllib.regression.LabeledPoint
    import org.apache.spark.mllib.linalg.Vectors
    import org.apache.spark.mllib.tree.RandomForest
    import org.apache.spark.mllib.util.MLUtils

    val data = processedRawData.map { r =>
      val activityId = r(0).toInt
      val activityIdx = activityId2Idx(activityId)
      val features = r.drop(1)
      LabeledPoint(activityIdx, Vectors.dense(features))
                                    }

    // @Snippet
    val splits = data.randomSplit(Array(0.8, 0.2))
    val (trainingData, testData) = (splits(0), splits(1))

    // @Snippet
    import org.apache.spark.mllib.tree.configuration._
    import org.apache.spark.mllib.tree.impurity._
    val rfStrategy = new Strategy(
      algo = Algo.Classification,
      impurity = Entropy,
      maxDepth = 10,
      maxBins = 20,
      numClasses = activityId2Idx.size,
      categoricalFeaturesInfo = Map[Int, Int](),
      subsamplingRate = 0.68)

    val rfModel = RandomForest.trainClassifier(
      input = trainingData,
      strategy = rfStrategy,
      numTrees = 50,
      featureSubsetStrategy = "auto",
      seed = 42)

    // @Snippet
    import org.apache.spark.mllib.evaluation._
    import org.apache.spark.mllib.tree.model._
    def getMetrics(model: RandomForestModel, data: RDD[LabeledPoint]):
    MulticlassMetrics = {
      val predictionsAndLabels = data.map(example =>
                                            (model.predict(example.features), example.label)
      )
      new MulticlassMetrics(predictionsAndLabels)
    }

    // @Snippet
    val rfModelMetrics = getMetrics(rfModel, testData)

    // @Snippet
    println(s"""|Confusion matrix (Rows x Columns = Actual x Predicted):
                |${rfModelMetrics.confusionMatrix}""".stripMargin)

    // @Snippet
    def idx2Activity(idx: Double): String =
    activityId2Idx.
      find(e => e._2 == idx.asInstanceOf[Int]).
      map(e => activitiesMap(e._1)).
      getOrElse("UNKNOWN")

    val rfCMLabels = rfModelMetrics.labels.map(idx2Activity(_))
    println(s"""|Labels:
                |${rfCMLabels.mkString(", ")}""".stripMargin)

    // @Snippet
    val rfCM = rfModelMetrics.confusionMatrix
    val rfCMTotal = rfCM.toArray.sum
    val rfAccuracy = (0 until rfCM.numCols).map(i => rfCM(i,i)).sum / rfCMTotal
    println(f"RandomForest accuracy = ${rfAccuracy*100}%.2f %%")

    // @Snippet
    import org.apache.spark.mllib.linalg.Matrix
    def colSum(m: Matrix, colIdx: Int) = (0 until m.numRows).map(m(_, colIdx)).sum
    def rowSum(m: Matrix, rowIdx: Int) = (0 until m.numCols).map(m(rowIdx, _)).sum
    val rfCMActDist = (0 until rfCM.numRows).map(rowSum(rfCM, _)/rfCMTotal)
    val rfCMPredDist = (0 until rfCM.numCols).map(colSum(rfCM, _)/rfCMTotal)

    println(s"""|Class distribution
                |Class, Actual, Predicted
                |${rfCMLabels.zip(rfCMActDist.zip(rfCMPredDist).map(p => f"${p._1*100}%.2f%%, ${p._2*100}%.2f%%")).mkString("\n")}
  """.stripMargin)

    // @Snippet
    def rfPrecision(m: Matrix, feature: Int) = m(feature, feature) / colSum(m, feature)
    def rfRecall(m: Matrix, feature: Int) = m(feature, feature) / rowSum(m, feature)
    def rfF1(m: Matrix, feature: Int) = 2 * rfPrecision(m, feature) * rfRecall(m, feature) / (rfPrecision(m, feature) + rfRecall(m, feature))

    val rfPerClassSummary = rfCMLabels.indices.map { i =>
      (rfCMLabels(i), rfRecall(rfCM, i), rfPrecision(rfCM, i), rfF1(rfCM, i))
                                                   }

    println(s"""|Per class summary
                |Label, Recall, Precision, F-1
                |${rfPerClassSummary.map(r => f"${r._1}%s, ${r._2}%.4f, ${r._3}%.4f, ${r._4}%.4f").mkString("\n")}""".stripMargin)

    // @Snippet
    val rfPerClassSummary2 = rfCMLabels.indices.map { i =>
      (rfCMLabels(i), rfModelMetrics.recall(i), rfModelMetrics.precision(i), rfModelMetrics.fMeasure(i))
                                                    }

    // @Snippet
    val rfMacroRecall = rfCMLabels.indices.map(i => rfRecall(rfCM, i)).sum/rfCMLabels.size
    val rfMacroPrecision = rfCMLabels.indices.map(i => rfPrecision(rfCM, i)).sum/rfCMLabels.size
    val rfMacroF1 = rfCMLabels.indices.map(i => rfF1(rfCM, i)).sum/rfCMLabels.size

    println(f"""|Macro statistics
                |Recall, Precision, F-1
                |${rfMacroRecall}%.4f, ${rfMacroPrecision}%.4f, ${rfMacroF1}%.4f""".stripMargin)

    // @Snippet
    println(f"""|Weighted statistics
                |Recall, Precision, F-1
                |${rfModelMetrics.weightedRecall}%.4f, ${rfModelMetrics.weightedPrecision}%.4f, ${rfModelMetrics.weightedFMeasure}%.4f
                |""".stripMargin)

    // @Snippet
    import org.apache.spark.mllib.linalg.Matrices

    val rfOneVsAll = rfCMLabels.indices.map { i =>
      val icm = rfCM(i,i)
      val irowSum = rowSum(rfCM, i)
      val icolSum = colSum(rfCM, i)
      Matrices.dense(2,2,
                     Array(
                       icm, irowSum - icm,
                       icolSum - icm, rfCMTotal - irowSum - icolSum + icm))
                                            }
    println(rfCMLabels.indices.map(i => s"${rfCMLabels(i)}\n${rfOneVsAll(i)}").mkString("\n"))

    // @Snippet
    val rfOneVsAllCM = rfOneVsAll.foldLeft(Matrices.zeros(2,2))( (acc, m) =>
                                                                   Matrices.dense(2, 2,
                                                                                  Array(acc(0, 0) + m(0, 0),
                                                                                        acc(1, 0) + m(1, 0),
                                                                                        acc(0, 1) + m(0, 1),
                                                                                        acc(1, 1) + m(1, 1)))
    )

    println(s"Sum of oneVsAll CM:\n${rfOneVsAllCM}")

    // @Snippet
    println(f"Average accuracy: ${(rfOneVsAllCM(0,0) + rfOneVsAllCM(1,1))/rfOneVsAllCM.toArray.sum}%.4f")

    // @Snippet
    println(f"Micro-averaged metrics: ${rfOneVsAllCM(0,0)/(rfOneVsAllCM(0,0)+rfOneVsAllCM(1,0))}%.4f")

    // @Snippet
    import org.apache.spark.h2o._
    val h2oContext = H2OContext.getOrCreate(sc)

    val trainHF = h2oContext.asH2OFrame(trainingData, "trainHF")
    trainHF.setNames(columnNames)
    trainHF.update()
    val testHF = h2oContext.asH2OFrame(testData, "testHF")
    testHF.setNames(columnNames)
    testHF.update()

    // @Snippet
    println(s"""|Distribution of activitiId:
                |${testData.map(row => (row.label, 1)).reduceByKey(_ + _).collect.sortBy(_._1).mkString(", ")}
                |""".stripMargin)

    // @Snippet
    trainHF.replace(0, trainHF.vec(0).toCategoricalVec).remove
    trainHF.update
    testHF.replace(0, testHF.vec(0).toCategoricalVec).remove
    testHF.update

    // @Snippet
    val domain = trainHF.vec(0).domain.map(i => idx2Activity(i.toDouble))
    trainHF.vec(0).setDomain(domain)
    water.DKV.put(trainHF.vec(0))
    testHF.vec(0).setDomain(domain)
    water.DKV.put(testHF.vec(0))

    // @Snippet
    import _root_.hex.tree.drf.DRF
    import _root_.hex.tree.drf.DRFModel
    import _root_.hex.tree.drf.DRFModel.DRFParameters
    import _root_.hex.ScoreKeeper._
    import _root_.hex.ConfusionMatrix
    import water.Key.make

    val drfParams = new DRFParameters
    drfParams._train = trainHF._key
    drfParams._valid = testHF._key
    drfParams._response_column = "activityId"
    drfParams._max_depth = 20
    drfParams._ntrees = 50
    drfParams._score_each_iteration = true
    drfParams._stopping_rounds = 2
    drfParams._stopping_metric = StoppingMetric.misclassification
    drfParams._stopping_tolerance = 1e-3
    drfParams._seed = 42

    val drfModel = new DRF(drfParams, make[DRFModel]("drfModel")).trainModel.get

    // @Snippet
    println(s"Number of trees: ${drfModel._output._ntrees}")

    // @Snippet
    val drfCM = drfModel._output._validation_metrics.cm

    def h2oCM2SparkCM(h2oCM: ConfusionMatrix): Matrix = {
      Matrices.dense(h2oCM.size, h2oCM.size, h2oCM._cm.flatMap(x => x))
    }
    val drfSparkCM = h2oCM2SparkCM(drfCM)

    // @Snippet
    val drfPerClassSummary = drfCM._domain.indices.map { i =>
      (drfCM._domain(i), rfRecall(drfSparkCM, i), rfPrecision(drfSparkCM, i), rfF1(drfSparkCM, i))
                                                       }

    println(s"""|Per class summary
                |Label, Recall, Precision, F-1
                |${drfPerClassSummary.map(r => f"${r._1}%s, ${r._2}%.4f, ${r._3}%.4f, ${r._4}%.4f").mkString("\n")}""".stripMargin)

    // @Snippet
    drfParams._ntrees = 20
    drfParams._stopping_rounds = 0
    drfParams._checkpoint = drfModel._key

    val drfModel20 = new DRF(drfParams, make[DRFModel]("drfModel20")).trainModel.get

    println(s"Number of trees: ${drfModel20._output._ntrees}")
  }
}
