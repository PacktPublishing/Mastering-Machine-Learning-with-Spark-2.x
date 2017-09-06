// @Snippet
val DATASET_DIR = s"${sys.env.get("DATADIR").getOrElse("data")}/aclImdb/train"
val FILE_SELECTOR = "*.txt"
case class Review(label: Int, reviewText: String)

val positiveReviews = spark.read.textFile(s"$DATASET_DIR/pos/$FILE_SELECTOR")
    .map(line => Review(1, line)).toDF
val negativeReviews = spark.read.textFile(s"$DATASET_DIR/neg/$FILE_SELECTOR")
  .map(line => Review(0, line)).toDF
var movieReviews = positiveReviews.union(negativeReviews)

// @Snippet
// @Snippet
import org.apache.spark.ml.feature.StopWordsRemover
val stopWords = StopWordsRemover.loadDefaultStopWords("english") ++ Array("ax", "arent", "re")

val MIN_TOKEN_LENGTH = 3
val toTokens = (minTokenLen: Int, stopWords: Array[String], review: String) =>
  review.split("""\W+""")
    .map(_.toLowerCase.replaceAll("[^\\p{IsAlphabetic}]", ""))
    .filter(w => w.length > minTokenLen)
    .filter(w => !stopWords.contains(w))

// @Snippet
import spark.implicits._
val toTokensUDF = udf(toTokens.curried(MIN_TOKEN_LENGTH)(stopWords))
movieReviews = movieReviews.withColumn("reviewTokens", toTokensUDF('reviewText))

// @Snippet
val word2vec = new Word2Vec()
  .setInputCol("reviewTokens")
  .setOutputCol("reviewVector")
  .setMinCount(1)
val w2vModel = word2vec.fit(movieReviews)

// @Snippet
w2vModel.findSynonyms("funny", 5).show()

// @Snippet
w2vModel.getVectors.where("word = 'funny'").show(truncate = false)

// @Snippet
val testDf = Seq(Seq("funny"), Seq("movie"), Seq("funny", "movie")).toDF("reviewTokens")
w2vModel.transform(testDf).show(truncate=false)

// @Snippet
val inputData = w2vModel.transform(movieReviews)

// @Snippet
val trainValidSplits = inputData.randomSplit(Array(0.8, 0.2))
val (trainData, validData) = (trainValidSplits(0), trainValidSplits(1))

// @Snippet
val gridSearch =
  for (
    hpImpurity <- Array("entropy", "gini");
    hpDepth <- Array(5, 20);
    hpBins <- Array(10, 50))
    yield {
      println(s"Building model with: impurity=${hpImpurity}, depth=${hpDepth}, bins=${hpBins}")
      val model = new DecisionTreeClassifier()
        .setFeaturesCol("reviewVector")
        .setLabelCol("label")
        .setImpurity(hpImpurity)
        .setMaxDepth(hpDepth)
        .setMaxBins(hpBins)
        .fit(trainData)
      
      val preds = model.transform(validData)
      val auc = new BinaryClassificationEvaluator().setLabelCol("label")
        .evaluate(preds)
      (hpImpurity, hpDepth, hpBins, auc)
    }

// @Snippet
import com.packtpub.mmlwspark.utils.Tabulizer.table
println(table(Seq("Impurity", "Depth", "Bins", "AUC"),
              gridSearch.sortBy(_._4).reverse,
              Map.empty[Int,String]))

// @Snippet
import org.apache.spark.h2o._
val hc = H2OContext.getOrCreate(sc)
val trainHf = hc.asH2OFrame(trainData, "trainData")
val validHf = hc.asH2OFrame(validData, "validData")

// @Snippet
hc.openFlow()

// @Snippet
w2vModel.findSynonyms("drama", 5).show()

// @Snippet
val newW2VModel = new Word2Vec()
  .setInputCol("reviewTokens")
  .setOutputCol("reviewVector")
  .setMinCount(3)
  .setMaxIter(250)
  .setStepSize(0.02)
  .fit(movieReviews)

// @Snippet
newW2VModel.findSynonyms("drama", 5).show()

