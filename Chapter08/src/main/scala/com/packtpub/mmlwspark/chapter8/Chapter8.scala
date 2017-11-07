package com.packtpub.mmlwspark.chapter8

// @formatter:off
import java.io.{FileOutputStream}

import hex.tree.gbm.GBMModel
import org.apache.spark.SparkContext
import org.apache.spark.ml.h2o.features.ColRemover
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{SQLContext, SparkSession}

/**
  * Mastering Machine Learning With Spark: Chapter 8
  *
  * Model training
  */
object Chapter8 extends App {

  val spark = SparkSession.builder()
    .master("local[*]")
    .appName("Chapter8")
    .getOrCreate()

  val sc = spark.sparkContext
  sc.setLogLevel("WARN")
  script(spark, sc, spark.sqlContext)
  

def script(spark: SparkSession, sc: SparkContext, sqlContext: SQLContext): Unit = {

    // @Snippet
    import sqlContext.implicits._
    import org.apache.spark.h2o._
    val h2oContext = H2OContext.getOrCreate(sc)

    // Load data
    val USE_SMALLDATA = false
    val DATASET_DIR = s"${sys.env.get("DATADIR").getOrElse("data")}"
    val DATASETS = if (!USE_SMALLDATA) Array("LoanStats3a.csv", "LoanStats3b.csv") else Array("loans.csv")
    import java.net.URI

    import water.fvec.H2OFrame
    val loanDataHf = new H2OFrame(DATASETS.map(name => URI.create(s"${DATASET_DIR}/${name}")):_*)

    // @Snippet
    import com.packtpub.mmlwspark.utils.Tabulizer.table
    val idColumns = Seq("id", "member_id")
    println(s"Columns with Ids: ${table(idColumns, 4, None)}")

    // @Snippet
    val constantColumns = loanDataHf.names().indices
      .filter(idx => loanDataHf.vec(idx).isConst || loanDataHf.vec(idx).isBad)
      .map(idx => loanDataHf.name(idx))
    println(s"Constant and bad columns: ${table(constantColumns, 4, None)}")

    // @Snippet
    val stringColumns = loanDataHf.names().indices
      .filter(idx => loanDataHf.vec(idx).isString)
      .map(idx => loanDataHf.name(idx))
    println(s"String columns:${table(stringColumns, 4, None)}")

    // @Snippet
    val loanProgressColumns = Seq("funded_amnt", "funded_amnt_inv", "grade", "initial_list_status",
                                  "issue_d", "last_credit_pull_d", "last_pymnt_amnt", "last_pymnt_d",
                                  "next_pymnt_d", "out_prncp", "out_prncp_inv", "pymnt_plan",
                                  "recoveries", "sub_grade", "total_pymnt", "total_pymnt_inv",
                                  "total_rec_int", "total_rec_late_fee", "total_rec_prncp")
    // @Snippet
    val columnsToRemove = (idColumns ++ constantColumns ++ stringColumns ++ loanProgressColumns)
  
    // @Snippet
    val categoricalColumns = loanDataHf.names().indices
      .filter(idx => loanDataHf.vec(idx).isCategorical)
      .map(idx => (loanDataHf.name(idx), loanDataHf.vec(idx).cardinality()))
      .sortBy(-_._2)
    
    println(s"Categorical columns:${table(categoricalColumns)}")

    // @Snippet
    // @See toNumericMnths

    // @Snippet
    // @See toNumericRate

    // @Snippet
    val naColumns = loanDataHf.names().indices
      .filter(idx => loanDataHf.vec(idx).naCnt() > 0)
      .map(idx =>
             (loanDataHf.name(idx),
               loanDataHf.vec(idx).naCnt(),
               f"${100*loanDataHf.vec(idx).naCnt()/loanDataHf.numRows().toFloat}%2.1f%%")
      ).sortBy(-_._2)
    println(s"Columns with NAs (#${naColumns.length}):${table(naColumns)}")
    
    // @Snippet
    // @See toBinaryLoanStatus

    // @Snippet
    // @See basicDataCleanup

    // @Snippet
    import com.packtpub.mmlwspark.chapter8.Chapter8Library._
    val loanDataDf = h2oContext.asDataFrame(loanDataHf)(sqlContext)
    val loanStatusBaseModelDf = basicDataCleanup(
      loanDataDf
        .where("loan_status is not null")
        .withColumn("loan_status", toBinaryLoanStatusUdf($"loan_status")),
      colsToDrop = Seq("title") ++ columnsToRemove)

    // @Snippet
    val loanStatusDfSplits = loanStatusBaseModelDf.randomSplit(Array(0.7, 0.3), seed = 42)

    val trainLSBaseModelHf = toHf(loanStatusDfSplits(0).drop("emp_title", "desc"), "trainLSBaseModelHf")(h2oContext)
    val validLSBaseModelHf = toHf(loanStatusDfSplits(1).drop("emp_title", "desc"), "validLSBaseModelHf")(h2oContext)

    // @Snippet
    import _root_.hex.tree.drf.DRFModel.DRFParameters
    import _root_.hex.tree.drf.{DRF, DRFModel}
    import _root_.hex.ScoreKeeper.StoppingMetric
    import com.packtpub.mmlwspark.utils.Utils.let

    val loanStatusBaseModelParams = let(new DRFParameters) { p =>
      p._response_column = "loan_status"
      p._train = trainLSBaseModelHf._key
      p._ignored_columns = Array("int_rate")
      p._stopping_metric = StoppingMetric.logloss
      p._stopping_rounds = 1
      p._stopping_tolerance = 0.1
      p._ntrees = 100
      p._balance_classes = true
      p._score_tree_interval = 20
    }

    //loanStatusBaseModelParams._nfolds = 5
    val loanStatusBaseModel1 = new DRF(loanStatusBaseModelParams, water.Key.make[DRFModel]("loanStatusBaseModel1"))
      .trainModel()
      .get()

    // @Snippet
    loanStatusBaseModelParams._ignored_columns = Array("int_rate", "collection_recovery_fee", "zip_code")
    val loanStatusBaseModel2 = new DRF(loanStatusBaseModelParams, water.Key.make[DRFModel]("loanStatusBaseModel2"))
      .trainModel()
      .get()
  h2oContext.openFlow()

    // @Snippet
    import _root_.hex.ModelMetrics
    val lsBaseModel2PredHf = loanStatusBaseModel2.score(validLSBaseModelHf)
    val lsBaseModel2PredModelMetrics = ModelMetrics.getFromDKV(loanStatusBaseModel2, validLSBaseModelHf)
    println(lsBaseModel2PredModelMetrics)

    // @Snippet
    // @See profitMoenyLoss

    // @Snippet
    // @See loanMoneyLoss

    // @Snippet
    // @See totalLoss

    // @Snippet
    import _root_.hex.AUC2.ThresholdCriterion
    val predVActHf: Frame = lsBaseModel2PredHf.add(validLSBaseModelHf)
    water.DKV.put(predVActHf)
    val predVActDf = h2oContext.asDataFrame(predVActHf)(sqlContext)
    val DEFAULT_THRESHOLDS = Array(0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)

    println(
      table(Array("Threshold", "Profit Loss", "Count", "Loan loss", "Count", "Total loss"),
            (DEFAULT_THRESHOLDS :+
                  ThresholdCriterion.min_per_class_accuracy.max_criterion(lsBaseModel2PredModelMetrics.auc_obj()))
             .map(threshold => totalLoss(predVActDf, threshold)),
            Map(1 -> "%,.2f", 3 -> "%,.2f", 5 -> "%,.2f")))

    // @Snippet
    def findMinLoss(model: DRFModel,
                    validHf: H2OFrame,
                    defaultThresholds: Array[Double]): (Double, Double, Double, Double) = {
      import _root_.hex.ModelMetrics
      import _root_.hex.AUC2.ThresholdCriterion
      // Score model
      val modelPredHf = model.score(validHf)
      val modelMetrics = ModelMetrics.getFromDKV(model, validHf)
      val predVActHf: Frame = modelPredHf.add(validHf)
      water.DKV.put(predVActHf)
      //
      val predVActDf = h2oContext.asDataFrame(predVActHf)(sqlContext)
      val min = (DEFAULT_THRESHOLDS :+ ThresholdCriterion.min_per_class_accuracy.max_criterion(modelMetrics.auc_obj()))
        .map(threshold => totalLoss(predVActDf, threshold)).minBy(_._6)
      ( /* Threshold */ min._1, /* Total loss */ min._6, /* Profit loss */ min._2, /* Loan loss */ min._4)
    }

    val minLossModel2 = findMinLoss(loanStatusBaseModel2, validLSBaseModelHf, DEFAULT_THRESHOLDS)
    println(f"Min total loss for model 2: ${minLossModel2._2}%,.2f (threshold = ${minLossModel2._1})")

    // @Snippet # Text columns
    // @see unifyTextColumn
    
    // @Snippet
    val ALL_NUM_REGEXP = java.util.regex.Pattern.compile("\\d*")
    val tokenizeTextColumn = (minLen: Int) => (stopWords: Array[String]) => (w: String) => {
      if (w != null)
        w.split(" ").map(_.trim).filter(_.length >= minLen).filter(!ALL_NUM_REGEXP.matcher(_).matches()).filter(!stopWords.contains(_)).toSeq
      else
        Seq.empty[String]
    }
    import org.apache.spark.ml.feature.StopWordsRemover
    val tokenizeUdf = udf(tokenizeTextColumn(3)(StopWordsRemover.loadDefaultStopWords("english")))
    
    // @Snippet
    val empTitleColumnDf = loanStatusBaseModelDf
      .withColumn("emp_title", unifyTextColumnUdf($"emp_title"))
      .withColumn("emp_title_tokens", tokenizeUdf($"emp_title"))
    
    // @Snippet
    println("Number of unique values in emp_title column: " +
            empTitleColumnDf.select("emp_title").groupBy("emp_title").count().count())
    println("Number of unique tokens with freq > 100 in emp_title column: " +
            empTitleColumnDf.select("emp_title_tokens").rdd.flatMap(row => row.getSeq[String](0).map(w => (w, 1)))
              .reduceByKey(_ + _).filter(_._2 > 100).count)

    // @Snippet
    import org.apache.spark.ml.feature.Word2Vec
    val empTitleW2VModel = new Word2Vec()
      .setInputCol("emp_title_tokens")
      .setOutputCol("emp_title_w2vVector")
      .setMinCount(1)
      .fit(empTitleColumnDf)
    
    // @Snippet
    val empTitleColumnWithW2V = empTitleW2VModel.transform(empTitleColumnDf)
    empTitleColumnWithW2V.printSchema()

    // @Snippet
    import org.apache.spark.ml.clustering.KMeans
    val K = 500
    val empTitleKmeansModel = new KMeans()
      .setFeaturesCol("emp_title_w2vVector")
      .setK(K)
      .setSeed(42L)
      //.setMaxIter(7)
      .setPredictionCol("emp_title_cluster")
      .fit(empTitleColumnWithW2V)
    val clustered = empTitleKmeansModel.transform(empTitleColumnWithW2V)
    clustered.printSchema()

    // @Snippet
    println(
      s"""Words in cluster '133':
         |${clustered.select("emp_title").where("emp_title_cluster = 133").take(10).mkString(", ")}
         |""".stripMargin)

    // @Snippet
    // NOTE: in the word processing we used full data! not only training data!
    import org.apache.spark.ml.Pipeline
    import org.apache.spark.sql.types._
    import org.apache.spark.ml.UDFTransformer

    val empTitleTransformationPipeline = new Pipeline()
      .setStages(Array(
        new UDFTransformer("unifier", unifyTextColumn, StringType, StringType)
          .setInputCol("emp_title").setOutputCol("emp_title_unified"),
        new UDFTransformer("tokenizer",
                           tokenizeTextColumn(3)(StopWordsRemover.loadDefaultStopWords("english")),
                           StringType, ArrayType(StringType, true))
          .setInputCol("emp_title_unified").setOutputCol("emp_title_tokens"),
        empTitleW2VModel,
        empTitleKmeansModel,
        new ColRemover().setKeep(false).setColumns(Array("emp_title", "emp_title_unified", "emp_title_tokens", "emp_title_w2vVector"))
      ))

    // @Snippet
    val empTitleTransformer = empTitleTransformationPipeline.fit(loanStatusBaseModelDf)
    empTitleTransformer.write.overwrite.save("/tmp/kolo42")

    // @Snippet
    val trainLSBaseModel3Df = empTitleTransformer.transform(loanStatusDfSplits(0))
    val validLSBaseModel3Df = empTitleTransformer.transform(loanStatusDfSplits(1))
    val trainLSBaseModel3Hf = toHf(trainLSBaseModel3Df.drop("desc"), "trainLSBaseModel3Hf")(h2oContext)
    val validLSBaseModel3Hf = toHf(validLSBaseModel3Df.drop("desc"), "validLSBaseModel3Hf")(h2oContext)

    // @Snippet
    loanStatusBaseModelParams._train = trainLSBaseModel3Hf._key
    val loanStatusBaseModel3 = new DRF(loanStatusBaseModelParams, water.Key.make[DRFModel]("loanStatusBaseModel3"))
      .trainModel()
      .get()

    // @Snippet
    val minLossModel3 = findMinLoss(loanStatusBaseModel3, validLSBaseModel3Hf, DEFAULT_THRESHOLDS)
    println(f"Min total loss for model 3: ${minLossModel3._2}%,.2f (threshold = ${minLossModel3._1})")

    
    // @Snippet
    import org.apache.spark.sql.types._
    val descColUnifier = new UDFTransformer(
      "unifier", unifyTextColumn, StringType, StringType)
      .setInputCol("desc")
      .setOutputCol("desc_unified")

    val descColTokenizer = new UDFTransformer(
        "tokenizer",
        tokenizeTextColumn(3)(StopWordsRemover.loadDefaultStopWords("english")),
        StringType, ArrayType(StringType, true)
      ).setInputCol("desc_unified")
      .setOutputCol("desc_tokens")

    // @Snippet
    import org.apache.spark.ml.feature.CountVectorizer
    val descCountVectorizer = new CountVectorizer()
      .setInputCol("desc_tokens")
      .setOutputCol("desc_vector")
      .setMinDF(1)
      .setMinTF(1)
    
    // @Snippet
    import org.apache.spark.ml.feature.IDF
    val descIdf = new IDF()
      .setInputCol("desc_vector")
      .setOutputCol("desc_idf_vector")
      .setMinDocFreq(1)

    // @Snippet
    import org.apache.spark.ml.Pipeline
    val descFreqPipeModel = new Pipeline()
      .setStages(
        Array(descColUnifier,
              descColTokenizer,
              descCountVectorizer,
              descIdf)
      ).fit(loanStatusBaseModelDf)

    // @Snippet
    val descFreqDf = descFreqPipeModel.transform(loanStatusBaseModelDf)
    import org.apache.spark.ml.feature.IDFModel
    import org.apache.spark.ml.feature.CountVectorizerModel
    val descCountVectorizerModel = descFreqPipeModel.stages(2).asInstanceOf[CountVectorizerModel]
    val descIdfModel = descFreqPipeModel.stages(3).asInstanceOf[IDFModel]
    val descIdfScores = descIdfModel.idf.toArray
    val descVocabulary = descCountVectorizerModel.vocabulary
    println(
      s"""
        ~Size of 'desc' column vocabulary: ${descVocabulary.length}
        ~Top ten highest scores:
        ~${table(descVocabulary.zip(descIdfScores).sortBy(-_._2).take(10))}
      """.stripMargin('~'))

    // @Snippet
    import org.apache.spark.ml.linalg.{Vector, Vectors}
    import org.apache.spark.sql.Row
    val rowAdder = (toVector: Row => Vector) => (r1: Row, r2: Row) => {
      Row(Vectors.dense((toVector(r1).toArray, toVector(r2).toArray).zipped.map((a, b) => a + b)))
    }

    val descTargetGoodLoan = descFreqDf
      .where("loan_status == 'good loan'")
      .select("desc_vector")
      .reduce(rowAdder((row:Row) => row.getAs[Vector](0))).getAs[Vector](0).toArray

    val descTargetBadLoan = descFreqDf
      .where("loan_status == 'bad loan'")
      .select("desc_vector")
      .reduce(rowAdder((row:Row) => row.getAs[Vector](0))).getAs[Vector](0).toArray

    // @Snippet
    val descTargetsWords = descTargetGoodLoan.zip(descTargetBadLoan)
      .zip(descVocabulary.zip(descIdfScores)).map(t => (t._1._1, t._1._2, t._2._1, t._2._2))
    println(
      s"""
         ~Words used only in description of good loans:
         ~${table(descTargetsWords.filter(t => t._1 > 0 && t._2 == 0).sortBy(-_._1).take(10))}
         ~
         ~Words used only in description of bad loans:
         ~${table(descTargetsWords.filter(t => t._1 == 0 && t._2 > 0).sortBy(-_._1).take(10))}
       """.stripMargin('~'))

    // @Snippet
    def descWordScore = (freqGoodLoan: Double, freqBadLoan: Double, wordIdfScore: Double) =>
      Math.abs(freqGoodLoan - freqBadLoan) * wordIdfScore * wordIdfScore

    // @Snippet
    val numOfGoodLoans = loanStatusBaseModelDf.where("loan_status == 'good loan'").count()
    val numOfBadLoans = loanStatusBaseModelDf.where("loan_status == 'bad loan'").count()

    val descDiscriminatingWords = descTargetsWords.filter(t => t._1 > 0 && t. _2 > 0).map(t => {
          val freqGoodLoan = t._1 / numOfGoodLoans
          val freqBadLoan = t._2 / numOfBadLoans
          val word = t._3
          val idfScore = t._4
          (word, freqGoodLoan*100, freqBadLoan*100, idfScore, descWordScore(freqGoodLoan, freqBadLoan, idfScore))
        })

    println(
      table(Seq("Word", "Freq Good Loan", "Freq Bad Loan", "Idf Score", "Score"),
        descDiscriminatingWords.sortBy(-_._5).take(100),
      Map(1 -> "%.2f", 2 -> "%.2f")))

    // @Snippet
    // @see descWordEncoder

    // @Snippet
    val trainLSBaseModel4Df = trainLSBaseModel3Df.withColumn("desc_denominating_words", descWordEncoderUdf($"desc")).drop("desc")
    val validLSBaseModel4Df = validLSBaseModel3Df.withColumn("desc_denominating_words", descWordEncoderUdf($"desc")).drop("desc")
    val trainLSBaseModel4Hf = toHf(trainLSBaseModel4Df, "trainLSBaseModel4Hf")(h2oContext)
    val validLSBaseModel4Hf = toHf(validLSBaseModel4Df, "validLSBaseModel4Hf")(h2oContext)
    loanStatusBaseModelParams._train = trainLSBaseModel4Hf._key
    val loanStatusBaseModel4 = new DRF(loanStatusBaseModelParams, water.Key.make[DRFModel]("loanStatusBaseModel4"))
      .trainModel()
      .get()

    // @Snippet
    val minLossModel4 = findMinLoss(loanStatusBaseModel4, validLSBaseModel4Hf, DEFAULT_THRESHOLDS)
    println(f"Min total loss for model 4: ${minLossModel4._2}%,.2f (threshold = ${minLossModel4._1})")

    // @Snippet
    println(
      s"""
        ~Results:
        ~${table(Seq("Threshold", "Total loss", "Profit loss", "Loan loss"),
                 Seq(minLossModel2, minLossModel3, minLossModel4),
                 Map(1 -> "%,.2f", 2 -> "%,.2f", 3 -> "%,.2f"))}
      """.stripMargin('~'))

    //
    // Int Rate model
    //
    // @Snippet
    val intRateDfSplits = loanStatusDfSplits.map(df => {
      df
        .where("loan_status == 'good loan'")
        .drop("emp_title", "desc", "loan_status")
        .withColumn("int_rate", toNumericRateUdf(col("int_rate")))
    })
    val trainIRHf = toHf(intRateDfSplits(0), "trainIRHf")(h2oContext)
    val validIRHf = toHf(intRateDfSplits(1), "validIRHf")(h2oContext)

    // @Snippet
    // @Snippet
    import _root_.hex.tree.gbm.GBMModel.GBMParameters
    val intRateModelParam = let(new GBMParameters()) { p =>
      p._train = trainIRHf._key
      p._valid = validIRHf._key
      p._response_column = "int_rate"
      p._score_tree_interval  = 20
    }
    // @Snippet
    import _root_.hex.grid.{GridSearch}
    import water.Key
    import scala.collection.JavaConversions._

    val intRateHyperSpace: java.util.Map[String, Array[Object]] = Map[String, Array[AnyRef]](
      "_ntrees" -> (1 to 10).map(v => Int.box(100*v)).toArray,
      "_max_depth" -> (2 to 7).map(Int.box).toArray,
      "_learn_rate" -> Array(0.1, 0.01).map(Double.box),
      "_col_sample_rate" -> Array(0.3, 0.7, 1.0).map(Double.box),
      "_learn_rate_annealing" -> Array(0.8, 0.9, 0.95, 1.0).map(Double.box)
    )

    // @Snippet
    import _root_.hex.grid.HyperSpaceSearchCriteria.RandomDiscreteValueSearchCriteria
    val intRateHyperSpaceCriteria = let(new RandomDiscreteValueSearchCriteria) { c =>
      c.set_stopping_metric(StoppingMetric.RMSE)
      c.set_stopping_tolerance(0.1)
      c.set_stopping_rounds(1)
      c.set_max_runtime_secs(4 * 60 /* seconds */)
    }

    // @Snippet
    val intRateGrid = GridSearch.startGridSearch(Key.make("intRateGridModel"),
                                                 intRateModelParam,
                                                 intRateHyperSpace,
                                                 new GridSearch.SimpleParametersBuilderFactory[GBMParameters],
                                                 intRateHyperSpaceCriteria).get()

    // @Snippet
    val intRateModel = intRateGrid.getModels.minBy(_._output._validation_metrics.rmse()).asInstanceOf[GBMModel]
    println(intRateModel._output._validation_metrics)

    // @Snippet
    // @See scoreLoanStatus

    // @Snippet
    val prediction = scoreLoan(loanStatusDfSplits(0),
                               empTitleTransformer,
                               loanStatusBaseModel4,
                               minLossModel4._4,
                               intRateModel)(h2oContext)
    prediction.show(10)

    // @Snippet
    import java.io.File
    val MODELS_DIR = s"${sys.env.get("MODELSDIR").getOrElse("models")}"
    val destDir = new File(MODELS_DIR)
    empTitleTransformer.write.overwrite.save(new File(destDir, "empTitleTransformer").getAbsolutePath)

    // @Snippet
    loanStatusBaseModel4.getMojo.writeTo(new FileOutputStream(new File(destDir, "loanStatusModel.mojo")))
    intRateModel.getMojo.writeTo(new FileOutputStream(new File(destDir, "intRateModel.mojo")))

    // @Snippet
    saveSchema(loanDataDf.schema, new File(destDir, "inputSchema.json"))

    sc.stop()
  }
}