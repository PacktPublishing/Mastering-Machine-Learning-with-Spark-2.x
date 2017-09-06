package org.apache.spark.examples.h2o

import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{MLWritable, MLWriter}
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * Chapter 4 source code.
  */
object Chapter4 extends App {

  val spark = SparkSession.builder()
     .master("local[*]")
     .appName("Chapter4")
     .config("spark.some.config.option", "some-value").getOrCreate()

  val sc = spark.sparkContext
  sc.setLogLevel("WARN")

  val USE_FULL_DATA = false
  val DATASET_DIR = "/Users/michal/Devel/projects/spark-book/repo/MMLS/chapter4/data/aclImdb/train"
  val MODELS_DIR = "/tmp/ch4models/"
  val FILE_SELECTOR = if (USE_FULL_DATA) "*.txt" else "111*.txt"

  // @Snippet
  val positiveReviews = spark.sqlContext.read.textFile(s"$DATASET_DIR/pos/$FILE_SELECTOR").toDF("reviewText")
  println(s"Number of positive reviews: ${positiveReviews.count}")

  // Show the first five reviews
  println("Positive reviews:")
  positiveReviews.show(5, truncate = true)

  // @Snippet
  val negativeReviews = spark.sqlContext.read.textFile(s"$DATASET_DIR/neg/$FILE_SELECTOR").toDF("reviewText")
  println(s"Number of negative reviews: ${negativeReviews.count}")

  // @Snippet
  import org.apache.spark.sql.functions._
  val pos = positiveReviews.withColumn("label", lit(1.0))
  val neg = negativeReviews.withColumn("label", lit(0.0))
  // FIXME: explain monotonically incresing id
  var movieReviews = pos.union(neg).withColumn("row_id", monotonically_increasing_id)

  println("All reviews:")
  movieReviews.show(5)

  // @Snippet
  movieReviews.show(1, truncate = false)

  // @Snippet
  import org.apache.spark.ml.feature.StopWordsRemover
  val stopWords = StopWordsRemover.loadDefaultStopWords("english") ++ Array("ax", "arent", "re")

  // @Snippet
  val MIN_TOKEN_LENGTH = 3
  val toTokens = (minTokenLen: Int, stopWords: Array[String], review: String) =>
    review.split("""\W+""")
      .map(_.toLowerCase.replaceAll("[^\\p{IsAlphabetic}]", ""))
      .filter(w => w.length > minTokenLen)
      .filter(w => !stopWords.contains(w))

  // @Snippet
  import spark.implicits._
  val toTokensUDF = udf(toTokens.curried(MIN_TOKEN_LENGTH)(stopWords))     // TODO: explain why separating toToken and toTokenUDF
  movieReviews = movieReviews.withColumn("reviewTokens", toTokensUDF('reviewText))

  // @Snippet
  val RARE_TOKEN = 2
  val rareTokens = movieReviews.select("reviewTokens")
    .flatMap(r => r.getAs[Seq[String]]("reviewTokens"))
    .map((v:String) => (v, 1))
    .groupByKey(t => t._1)
    .reduceGroups((a,b) => (a._1, a._2 + b._2))
    .map(_._2)
    .filter(t => t._2 < RARE_TOKEN)
    .map(_._1)
    .collect()

  println(s"Rare tokens count: ${rareTokens.size}")
  println(s"Rare tokens: ${rareTokens.take(10).mkString(", ")}")

  // @Snippet
  val rareTokensFilter = (rareTokens: Array[String], tokens: Seq[String]) => tokens.filter(token => !rareTokens.contains(token))
  val rareTokensFilterUDF = udf(rareTokensFilter.curried(rareTokens))
  movieReviews = movieReviews.withColumn("reviewTokens", rareTokensFilterUDF('reviewTokens))

  println("Movie reviews tokens:")
  movieReviews.show(5)

  // @Snippet
  import org.apache.spark.ml.feature.HashingTF

  val hashingTF = new HashingTF
  hashingTF.setInputCol("reviewTokens")
           .setOutputCol("tf")
           .setNumFeatures(1 << 12) // 2^12
           .setBinary(false)
  val tfTokens = hashingTF.transform(movieReviews)
  println("Vectorized movie reviews:")
  tfTokens.show(5)

  // @Snippet
  import org.apache.spark.ml.feature.IDF
  val idf = new IDF
  idf.setInputCol(hashingTF.getOutputCol)
    .setOutputCol("tf-idf")
  val idfModel = idf.fit(tfTokens)

  // @Snippet
  val tfIdfTokens = idfModel.transform(tfTokens)

  println("Vectorized and scaled movie reviews:")
  tfIdfTokens.show(5)

  // @Snippet
  import org.apache.spark.ml.linalg.Vector
  val vecTf = tfTokens.take(1)(0).getAs[Vector]("tf").toSparse
  val vecTfIdf = tfIdfTokens.take(1)(0).getAs[Vector]("tf-idf").toSparse
  println(s"Both vectors contains the same layout of non-zeros: ${java.util.Arrays.equals(vecTf.indices, vecTfIdf.indices)}")
  println(s"${vecTf.values.zip(vecTfIdf.values).take(5).mkString("\n")}")

  // @Snippet
  val splits = tfIdfTokens.select("row_id", "label", idf.getOutputCol).randomSplit(Array(0.7, 0.1, 0.1, 0.1), seed = 42)
  val (trainData, testData, transferData, validationData) = (splits(0), splits(1), splits(2), splits(3))
  trainData.cache()
  testData.cache()

  // @Snippet
  // Grid Search of Hyper-Parameters for Decision Tree
  import org.apache.spark.ml.classification.DecisionTreeClassifier
  import org.apache.spark.ml.classification.DecisionTreeClassificationModel
  import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
  import java.io.File

  val dtModelPath = s"$MODELS_DIR/dtModel"
  val dtModel = if (new File(dtModelPath).exists()) {
    DecisionTreeClassificationModel.load(dtModelPath)
  } else {
    val dtGridSearch =
      for (
        dtImpurity <- Array("entropy", "gini");
        dtDepth <- Array(3, 5))
        yield {
          println(s"Training decision tree: impurity $dtImpurity, depth: $dtDepth")
          val dtModel = new DecisionTreeClassifier()
            .setFeaturesCol(idf.getOutputCol)
            .setLabelCol("label")
            .setImpurity(dtImpurity)
            .setMaxDepth(dtDepth)
            .setMaxBins(10)
            .setSeed(42)
            .setCacheNodeIds(true)
            .fit(trainData)
          val dtPrediction = dtModel.transform(testData)
          val dtAUC = new BinaryClassificationEvaluator().setLabelCol("label")
            .evaluate(dtPrediction)
          println(s"  DT AUC on test data: $dtAUC")
          ((dtImpurity, dtDepth), dtModel, dtAUC)
        }
    println(dtGridSearch.sortBy(-_._3).take(5).mkString("\n"))
    val bestModel = dtGridSearch.sortBy(-_._3).head._2
    bestModel.write.overwrite.save(dtModelPath)
    bestModel
  }

  // @Snippet
  import org.apache.spark.ml.classification.{NaiveBayes, NaiveBayesModel}

  val nbModelPath = s"$MODELS_DIR/nbModel"
  val nbModel = if (new File(nbModelPath).exists()) {
    NaiveBayesModel.load(nbModelPath)
  } else {
    val model = new NaiveBayes()
      .setFeaturesCol(idf.getOutputCol)
      .setLabelCol("label")
      .setSmoothing(1.0)
      .setModelType("multinomial") // Note: input data are multinomial
      .fit(trainData)
    val nbPrediction = model.transform(testData)
    val nbAUC = new BinaryClassificationEvaluator().setLabelCol("label").evaluate(nbPrediction)
    println(s"Naive Bayes AUC: $nbAUC")
    model.write.overwrite.save(nbModelPath)
    model
  }

  // @Snippet
  import org.apache.spark.ml.classification.{RandomForestClassifier, RandomForestClassificationModel}

  val rfModelPath = s"$MODELS_DIR/rfModel"
  val rfModel = if (new File(rfModelPath).exists()) {
    RandomForestClassificationModel.load(rfModelPath)
  } else {
    val rfGridSearch =
      for (
        rfNumTrees <- Array(10, 15);
        rfImpurity <- Array("entropy", "gini");
        rfDepth <- Array(3, 5))
        yield {
          println(
            s"Training random forest: numTrees: $rfNumTrees, impurity $rfImpurity, depth: $rfDepth")
          val rfModel = new RandomForestClassifier()
            .setFeaturesCol(idf.getOutputCol)
            .setLabelCol("label")
            .setNumTrees(rfNumTrees)
            .setImpurity(rfImpurity)
            .setMaxDepth(rfDepth)
            .setMaxBins(10)
            .setSubsamplingRate(0.67)
            .setSeed(42)
            .setCacheNodeIds(true)
            .fit(trainData)

          val rfPrediction = rfModel.transform(testData)
          val rfAUC = new BinaryClassificationEvaluator().setLabelCol("label")
            .evaluate(rfPrediction)
          println(s"  RF AUC on test data: $rfAUC")
          ((rfNumTrees, rfImpurity, rfDepth), rfModel, rfAUC)
        }
    println(rfGridSearch.sortBy(-_._3).take(5).mkString("\n"))
    val bestModel = rfGridSearch.sortBy(-_._3).head._2 // Stress that the model is minimal because of defined gird space^
    bestModel.write.overwrite.save(rfModelPath)
    bestModel
  }

  // @Snippet
  import org.apache.spark.ml.classification.{GBTClassifier, GBTClassificationModel}
  val gbmModelPath = s"$MODELS_DIR/gbmModel"
  val gbmModel = if (new File(gbmModelPath).exists()) {
    GBTClassificationModel.load(gbmModelPath)
  } else {
    val model = new GBTClassifier()
      .setFeaturesCol(idf.getOutputCol)
      .setLabelCol("label")
      .setMaxIter(20) // FIXME higher
      .setMaxDepth(6) // FIXME 4
      .setCacheNodeIds(true)
      .fit(trainData)

    val gbmPrediction = model.transform(testData)
    gbmPrediction.show()
    val gbmAUC = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol(model.getPredictionCol)
      .evaluate(gbmPrediction)
    // TODO: gbm does not output raw prediction right now
    println(s"  GBM AUC on test data: $gbmAUC")
    model.write.overwrite.save(gbmModelPath)
    model
  }

  // @Snippet
  import org.apache.spark.ml.PredictionModel

  val models = Seq(("NB", nbModel), ("DT", dtModel), ("RF", rfModel), ("GBM", gbmModel))
  def slData(inputData: DataFrame, responseColumn: String, baseModels: Seq[(String, PredictionModel[_, _])]): DataFrame = {
    baseModels.map { case (name, model) =>
      model.transform(inputData)
        .select("row_id", model.getPredictionCol )
        .withColumnRenamed("prediction", s"${name}_prediction")
    }.reduceLeft((a, b) => a.join(b, Seq("row_id"), "inner"))
      .join(inputData.select("row_id", responseColumn), Seq("row_id"), "inner")
  }

  val mlTrainData = slData(transferData, "label", models).drop("row_id")
  mlTrainData.show()

  // @Snippet
  val mlTestData = slData(validationData, "label", models).drop("row_id")

  // @Snippet
  import org.apache.spark.h2o._
  val hc = H2OContext.getOrCreate(sc)
  val mlTrainHF = hc.asH2OFrame(mlTrainData, "metaLearnerTrain")
  val mlTestHF = hc.asH2OFrame(mlTestData, "metaLearnerTest")

  import water.fvec.Vec
  val toEnumUDF = (name: String, vec: Vec) => vec.toCategoricalVec
  mlTrainHF(toEnumUDF, 'label).update()
  mlTestHF(toEnumUDF, 'label).update()

  import org.apache.spark.ml.h2o.models.H2ODeepLearning
  val metaLearningModel = new H2ODeepLearning()(hc, spark.sqlContext)
    .setTrainKey(mlTrainHF.key)
    .setValidKey(mlTestHF.key)
    .setResponseColumn("label")
    .setEpochs(10)
    .setHidden(Array(100, 100, 50))
    .fit(null)

  // @Snippet
  import org.apache.spark.ml.{Pipeline, UnaryTransformer}
  import org.apache.spark.sql.types._
  class UDFTransformer[T, U](override val uid: String,
                             f: T => U,
                             inType: DataType,
                             outType: DataType)
    extends UnaryTransformer[T, U, UDFTransformer[T, U]] with MLWritable {

    override protected def createTransformFunc: T => U = f

    override protected def validateInputType(inputType: DataType): Unit = require(inputType == inType)

    override protected def outputDataType: DataType = outType

    override def write: MLWriter = new MLWriter {
      override protected def saveImpl(path: String): Unit = {}
    }
  }

  // @Snippet
  val tokenizerTransformer = new UDFTransformer[String, Array[String]](
    "tokenizer",
    toTokens.curried(MIN_TOKEN_LENGTH)(stopWords),
    StringType, new ArrayType(StringType, true))

  // @Snippet
  val rareTokensFilterTransformer = new UDFTransformer[Seq[String], Seq[String]](
    "rareWordsRemover",
    rareTokensFilter.curried(rareTokens),
    new ArrayType(StringType, true), new ArrayType(StringType, true))

  // @Snippet
  import org.apache.spark.ml.Transformer
  private class ColumnSelector(override val uid: String, val columnsToSelect: Array[String])
    extends Transformer with MLWritable {

    override def transform(dataset: Dataset[_]): DataFrame = {
      dataset.select(columnsToSelect.map(dataset.col): _*)
    }

    override def transformSchema(schema: StructType): StructType = {
      StructType(schema.fields.filter(col => columnsToSelect.contains(col.name)))
    }

    override def copy(extra: ParamMap): ColumnSelector = defaultCopy(extra)

    override def write: MLWriter = new MLWriter {
      override protected def saveImpl(path: String): Unit = {}
    }
  }
  // @Snippet
  val columnSelector = new ColumnSelector("columnSelector", Array(dtModel, nbModel, rfModel, gbmModel).map(_.getPredictionCol))

  // @Snippet
  val superLearnerPipeline = new Pipeline()
    .setStages(Array(
      // Tokenize
      tokenizerTransformer
        .setInputCol("reviewText")
        .setOutputCol("allReviewTokens"),
      // Remove rare items
      rareTokensFilterTransformer
        .setInputCol("allReviewTokens")
        .setOutputCol("reviewTokens"),
      hashingTF,
      idfModel,
      dtModel
        .setPredictionCol(s"DT_${dtModel.getPredictionCol}")
        .setRawPredictionCol(s"DT_${dtModel.getRawPredictionCol}")
        .setProbabilityCol(s"DT_${dtModel.getProbabilityCol}"),
      nbModel
        .setPredictionCol(s"NB_${nbModel.getPredictionCol}")
        .setRawPredictionCol(s"NB_${nbModel.getRawPredictionCol}")
        .setProbabilityCol(s"NB_${nbModel.getProbabilityCol}"),
      rfModel
        .setPredictionCol(s"RF_${rfModel.getPredictionCol}")
        .setRawPredictionCol(s"RF_${rfModel.getRawPredictionCol}")
        .setProbabilityCol(s"RF_${rfModel.getProbabilityCol}"),
      gbmModel // Note: GBM does not have full API of PredictionModel
        .setPredictionCol(s"GBM_${gbmModel.getPredictionCol}"),
      columnSelector,
      metaLearningModel
    ))

  // @Snippet
  val superLearnerModel  = superLearnerPipeline.fit(pos)
  val testx = superLearnerModel.transform(sc.parallelize(Seq("Although I love this movie, I can barely watch it, it is so real. So, I put it on tonight and hid behind my bank of computers. I remembered it vividly, but just wanted to see if I could find something I hadn't seen before........I didn't: that's because it's so real to me.<br /><br />Another \"user\" wrote the ages of the commentators should be shown with their summary. I'm all for that ! It's absolutely obvious that most of these people who've made comments about \"Midnight Cowboy\" may not have been born when it was released. They are mentioning other movies Jon Voight and Dustin Hoffman have appeared in, at a later time. I'll be just as ruinously frank: I am 82-years-old. If you're familiar with some of my other comments, you'll be aware that I was a professional female-impersonator for 60 of those years, and also have appeared in film - you'd never recognize me, even if you were familiar with my night-club persona. Do you think I know a lot about the characters in this film ? YOU BET I DO !!....<br /><br />....and am not the least bit ashamed. If you haven't run-into some of them, it's your loss - but, there's a huge chance you have, but just didn't know it. So many moms, dads, sons and daughters could surprise you. It should be no secret MANY actors/actresses have emerged from the backgrounds of \"Midnight Cowboy\". Who is to judge ? I can name several, current BIG-TIME stars who were raised on the seedy streets of many cities, and weren't the least bit damaged by their time spent there. I make no judgment, because these are humans, just as we all are - love, courage, kindness, compassion, intelligence, humility: you name the attributes, they are all there, no matter what the package looks like.<br /><br />The \"trivia\" about Hoffman actually begging on the streets to prove he could do the role of \"Ratzo\" is a gem - he can be seen driving his auto all around Los Angeles - how do you think he gets his input? I can also name lots of male-stars who have stood on the streets and cruised the bars for money. Although the nightclub I last worked in for 26 years was world-famous and legit, I can also name some HUGE stars that had to be constantly chased out our back-street, looking to make a pick-up.<br /><br />This should be no surprise today, although it's definitely action in Hollywood and other cities, large and small. Wake-up and smell the roses. They smell no less sweet because they are of a different hue.<br /><br />Some of the \"users\" thought \"Joe Buck\" had been molested by his grandma. Although I saw him in her bed with a boyfriend, I didn't find any incidence of that. Believe-it-or-not, kids haven't ALWAYS had their own rooms - because that is a must today should tell you something kinda kinky may be going-on in the master-bedroom. Whose business? Hoffman may have begged for change on the streets, but some of the \"users\" point-out that Jon Voight was not a major star for the filming of \"Midnight Cowboy\" - his actual salary would surprise you. I think he was robbed ! No one can doubt the clarity he put into his role, nor that it MADE him a star for such great work as \"Deliverance\". He defined a potent man who had conquered his devils and was the better for it: few people commented he had been sodomized in this movie. The end of the 60s may have been one of the first films to be so open, but society has always been hip.<br /><br />I also did not find any homosexuality between \"Ratzo\" and \"Joe\" - they were clearly opposites, unappealing to one another. They found a much purely higher relationship - true friendship. If you didn't understand that at the end of the movie, then you've wasted your time. \"Joe's\" bewilderment, but unashamed devotion was apparent. Yes, Voight deserved an Oscar for this role - one that John Wayne could never pull-off, and he was as handsome in his youth.<br /><br />Hoffman is Hoffman - you expect fireworks. He gave them superbly. Wayne got his Oscar. Every character in this film was beautifully defined - if you don't think they are still around, you are mistaken. \"The party\" ? - attend some of the \"raves\" younger people attend.....if you can get in. Look at the lines of people trying to get into the hot clubs - you'll see every outrageous personality.<br /><br />Brenda Viccaro was the epitome of society's sleek women who have to get down to the nitty-gritty at times. If you were shocked by her brilliant acting, thinking \"this isn't real\", look at today's \"ladies\" who live on the brink of disrepute....and are admired for it.<br /><br />The brutality \"Joe\" displayed in robbing the old guy, unfortunately, is also a part of life. You don't have to condone it, but it's not too much different than any violence. \"Joe\" pointedly named his purpose - in that situation, I'd have handed-over the money quicker than he asked for it. That's one of the scenes that makes this movie a break-through, one which I do not watch. I get heartbroken for both.....<br /><br />John Schlesinger certainly must have been familiar with this sordidness to direct this chillingly beautiful eye-opener- Waldo Salt didn't write from clairvoyance. Anyone who had any part of getting it to the screen must have realized they were making history, and should be proud for the honesty of it. Perhaps \"only in America\" can we close our eyes to unpleasant situations, while other movie-makers make no compunction in presenting it to the public. Not looking doesn't mean it isn't there - give me the truth every time. Bravo! to all")).toDF("reviewText"))
  testx.printSchema()
  testx.show()
  //val pipelineModelPath = s"$MODELS_DIR/pipeline"
  //pipelineModel.write.overwrite.save(pipelineModelPath)

  sc.stop()
}
