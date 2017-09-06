// @Snippet
val trainPositiveFiles = sc.wholeTextFiles("../data/aclImdb/train/pos/*.txt")
// Extract data field
val trainPositiveReviews = trainPositiveFiles.map { case (file, data) => data }
println(s"Number of positive reviews: ${trainPositiveReviews.count}")

// Show the first five reviews
trainPositiveReviews.take(5).foreach(println)

// @Snippet
val trainNegativeFiles = sc.wholeTextFiles("../data/aclImdb/train/neg/*.txt")
val trainNegativeReviews = trainNegativeFiles.map { case (file, data) => data }

println(s"Number of negative reviews: ${trainPositiveReviews.count}")

// @Snippet

