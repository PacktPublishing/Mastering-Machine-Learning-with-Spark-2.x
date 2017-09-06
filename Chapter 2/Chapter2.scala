package org.apache.spark.book.examples

/**
  * Chapter 2
  */
object Chapter2 extends {

  import org.apache.spark.{SparkConf,  SparkContext}
  import org.apache.spark.sql.{SQLContext, SparkSession}

  /* Simulate environment of Spark shell */
  val config = new SparkConf()
    .setMaster("local[*]")
    .setAppName("Chapter2")

  val sc = SparkContext.getOrCreate(config)
  val sqlContext = SparkSession.builder().getOrCreate().sqlContext

  script(sc, sqlContext)

  import sqlContext.implicits._

  def script(sc: SparkContext, sqlContext: SQLContext): Unit = {
    // @Snippet
    import org.apache.spark.rdd.RDD
    import org.apache.spark.mllib.regression.LabeledPoint
    import org.apache.spark.mllib.linalg._
    import org.apache.spark.mllib.linalg.distributed.RowMatrix
    import org.apache.spark.mllib.evaluation._
    import org.apache.spark.mllib.tree._

    // @Snippet
    val rawData = sc.textFile("/tmp/data.csv")
    rawData.count

    // @Snippet
    rawData.take(2)

    // @Snippet
    val data = rawData.map(line => line.split(',').map(_.toDouble))

    // @Snippet
    val response: RDD[Int] = data.map(row => row(0).toInt)
    val features: RDD[Vector] = data.map(line => Vectors.dense(line.slice(1, line.size)))

    // @Snippet
    val featuresMatrix = new RowMatrix(features)
    val featuresSummary = featuresMatrix.computeColumnSummaryStatistics()

    // @Snippet
    println(s"Higgs Features Mean Values = ${featuresSummary.mean}")
    println(s"Higgs Features Variance Values = ${featuresSummary.variance}")

    // @Snippet
    val nonZeros = featuresSummary.numNonzeros
    println(s"Non-zero values count per column: $nonZeros")

    // @Snippet
    val numRows = featuresMatrix.numRows
    val numCols = featuresMatrix.numCols
    val colsWithZeros = nonZeros
      .toArray
      .zipWithIndex
      .filter { case (rows, idx) => rows != numRows }

    // @Snippet
    val sparsity = nonZeros.toArray.sum / (numRows * numCols)

    // @Snippet
    response.distinct.collect

    // @Snippet
    response.map(v => (v,1)).countByKey

    // @Snippet
    import org.apache.spark.h2o._
    val h2oContext = H2OContext.getOrCreate(sc)
    val h2oResponse = h2oContext.asH2OFrame(response, "response")
    h2oContext.openFlow

    // @Snippet
    val higgs = response.zip(features).map { case (response, features) => LabeledPoint(response, features) }
    higgs.setName("higgs").cache()

    // @Snippet
    val trainTestSplits = higgs.randomSplit(Array(0.8, 0.2))
    val (trainingData, testData) = (trainTestSplits(0), trainTestSplits(1))

    // @SkipCode
    println("""|
              | === Tree Model ===
              |""".stripMargin)

    // @Snippet
    val dtNumClasses = 2
    val dtCategoricalFeaturesInfo = Map[Int, Int]()
    val dtImpurity = "gini"
    val dtMaxDepth = 5
    val dtMaxBins = 10

    /// @Snippet
    val dtreeModel = DecisionTree.trainClassifier(trainingData,
                                                  dtNumClasses,
                                                  dtCategoricalFeaturesInfo,
                                                  dtImpurity,
                                                  dtMaxDepth,
                                                  dtMaxBins)

    println(s"Decision Tree Model:\n${dtreeModel.toDebugString}")

    // @Snippet
    val treeLabelAndPreds = testData.map { point =>
      val prediction = dtreeModel.predict(point.features)
      (point.label.toInt, prediction.toInt)
                                         }

    val treeTestErr = treeLabelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count()
    println(f"Tree Model: Test Error = ${treeTestErr}%.3f")

    // @Snippet
    val cm = treeLabelAndPreds.combineByKey(
      createCombiner = (label: Int) => if (label == 0) (1,0) else (0,1),
      mergeValue = (v: (Int,Int), label: Int) => if (label == 0) (v._1 +1, v._2) else (v._1, v._2 + 1),
      mergeCombiners = (v1: (Int,Int), v2: (Int,Int)) => (v1._1 + v2._1, v1._2 + v2._2)).collect

    // @Snippet
    val (tn, tp, fn, fp) = (cm(0)._2._1, cm(1)._2._2, cm(1)._2._1, cm(0)._2._2)
    println(f"""Confusion Matrix
                |   ${0}%5d ${1}%5d  ${"Err"}%10s
                |0  ${tn}%5d ${fp}%5d ${tn+fp}%5d ${fp.toDouble/(tn+fp)}%5.4f
                |1  ${fn}%5d ${tp}%5d ${fn+tp}%5d ${fn.toDouble/(fn+tp)}%5.4f
                |   ${tn+fn}%5d ${fp+tp}%5d ${tn+fp+fn+tp}%5d ${(fp+fn).toDouble/(tn+fp+fn+tp)}%5.4f
                |""".stripMargin)

    // @Snippet
    type Predictor = {
      def predict(features: Vector): Double
    }

    def computeMetrics(model: Predictor, data: RDD[LabeledPoint]): BinaryClassificationMetrics = {
      val predAndLabels = data.map(newData => (model.predict(newData.features), newData.label))
      new BinaryClassificationMetrics(predAndLabels)
    }

    val treeMetrics = computeMetrics(dtreeModel, testData)
    println(f"Tree Model: AUC on Test Data = ${treeMetrics.areaUnderROC()}%.3f")

    // @SkipCode
    println("""|
              | === Random Forest Model ===
              |""".stripMargin)

    // @Snippet
    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees = 10
    val featureSubsetStrategy = "auto"
    val impurity = "gini"
    val maxDepth = 5
    val maxBins = 10

    val rfModel = RandomForest.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

    // @Snippet
    def computeError(model: Predictor, data: RDD[LabeledPoint]): Double = {
      val labelAndPreds = data.map { point =>
        val prediction = model.predict(point.features)
        (point.label, prediction)
                                   }
      labelAndPreds.filter(r => r._1 != r._2).count.toDouble/data.count
    }

    val rfTestErr = computeError(rfModel, testData)
    println(f"RF Model: Test Error = ${rfTestErr}%.3f")

    // @Snippet
    val rfMetrics = computeMetrics(rfModel, testData)
    println(f"RF Model: AUC on Test Data = ${rfMetrics.areaUnderROC}%.3f")

    // @Snippet
    val rfGrid =
    for (
      gridNumTrees <- Array(15, 20);
      gridImpurity <- Array("entropy", "gini");
      gridDepth <- Array(20, 30);
      gridBins <- Array(20, 50)) yield {
      val gridModel = RandomForest.trainClassifier(trainingData, 2, Map[Int, Int](), gridNumTrees, "auto", gridImpurity, gridDepth, gridBins)
      val gridAUC = computeMetrics(gridModel, testData).areaUnderROC
      val gridErr = computeError(gridModel, testData)
      ((gridNumTrees, gridImpurity, gridDepth, gridBins), gridAUC, gridErr)
    }

    // @Snippet
    val rfParamsMaxAUC = rfGrid.maxBy(g => g._2)
    println(f"Parameters ${rfParamsMaxAUC._1}%s producing max AUC = ${rfParamsMaxAUC._2}%.3f (error = ${rfParamsMaxAUC._3}%.3f)")

    // @SkipCode
    println("""|
              | === Gradient Boosted Trees Model ===
              |""".stripMargin)

    // @Snippet
    import org.apache.spark.mllib.tree.GradientBoostedTrees
    import org.apache.spark.mllib.tree.configuration.BoostingStrategy
    import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel
    import org.apache.spark.mllib.util.MLUtils

    val gbmStrategy = BoostingStrategy.defaultParams("Classification")
    gbmStrategy.setNumIterations(10)
    gbmStrategy.treeStrategy.setNumClasses(2)
    gbmStrategy.treeStrategy.setMaxDepth(10)
    gbmStrategy.setLearningRate(0.1)
    gbmStrategy.treeStrategy.setCategoricalFeaturesInfo(Map[Int, Int]())

    val gbmModel = GradientBoostedTrees.train(trainingData, gbmStrategy)

    // @Snippet
    val gbmTestErr = computeError(gbmModel, testData)
    println(f"GBM Model: Test Error = ${gbmTestErr}%.3f")

    val gbmMetrics = computeMetrics(dtreeModel, testData)
    println(f"GBM Model: AUC on Test Data = ${gbmMetrics.areaUnderROC()}%.3f")

    // @Snippet
    val gbmGrid =
    for (
      gridNumIterations <- Array(5, 10, 50, 100);
      gridDepth <- Array(2, 3, 5, 7);
      gridLearningRate <- Array(0.1, 0.01, 0.001))
      yield {
        gbmStrategy.setNumIterations(gridNumIterations)
        gbmStrategy.treeStrategy.setMaxDepth(gridDepth)
        gbmStrategy.setLearningRate(gridLearningRate)

        val gridModel = GradientBoostedTrees.train(trainingData, gbmStrategy)
        val gridAUC = computeMetrics(gridModel, testData).areaUnderROC
        val gridErr = computeError(gridModel, testData)
        ((gridNumIterations, gridDepth, gridLearningRate), gridAUC, gridErr)
      }

    // @Snippet
    val gbmParamsMaxAUC = gbmGrid.maxBy(g => g._2)
    println(f"Parameters ${gbmParamsMaxAUC._1}%s producing max AUC = ${gbmParamsMaxAUC._2}%.3f (error = ${gbmParamsMaxAUC._3}%.3f)")

    // @SkipCode
    println("""|
              | === Deep Learning Model ===
              |""".stripMargin)

    // @Snippet
    val trainingHF = h2oContext.asH2OFrame(trainingData.toDF, "trainingHF")
    val testHF = h2oContext.asH2OFrame(testData.toDF, "testHF")

    // @Snippet
    trainingHF.replace(0, trainingHF.vecs()(0).toCategoricalVec).remove()
    water.DKV.put(trainingHF)
    testHF.replace(0, testHF.vecs()(0).toCategoricalVec).remove()
    water.DKV.put(testHF)

    // @Snippet
    import _root_.hex.deeplearning._
    import _root_.hex.deeplearning.DeepLearningModel.DeepLearningParameters
    import _root_.hex.deeplearning.DeepLearningModel.DeepLearningParameters.Activation

    val dlParams = new DeepLearningParameters()
    dlParams._train = trainingHF._key
    dlParams._valid = testHF._key
    dlParams._response_column = "label"
    dlParams._epochs = 1
    dlParams._activation = Activation.RectifierWithDropout
    dlParams._hidden = Array[Int](500, 500, 500)

    // @Snippet
    val dlBuilder = new DeepLearning(dlParams)
    val dlModel = dlBuilder.trainModel.get

    // @Snippet
    print(dlModel)

    // @Snippet
    val testPredictions = dlModel.score(testHF)

    // @Snippet
    import water.support.ModelMetricsSupport._
    import _root_.hex.ModelMetricsBinomial
    val dlMetrics = modelMetrics[ModelMetricsBinomial](dlModel, testHF)

    println(dlMetrics)
  }
}
