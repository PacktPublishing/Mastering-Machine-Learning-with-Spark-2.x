# Mastering Machine Learning with Spark 2.x
This is the code repository for [Mastering Machine Learning with Spark 2.x](https://www.packtpub.com/big-data-and-business-intelligence/mastering-machine-learning-spark-2x?utm_source=github&utm_medium=repository&utm_campaign=9781785283451), published by [Packt](https://www.packtpub.com/?utm_source=github). It contains all the supporting project files necessary to work through the book from start to finish.
## About the Book
This book gives you access to transform data into actionable knowledge. The book commences by defining machine learning primitives by the MLlib and H2O libraries. You will learn how to use Binary classification to detect the Higgs Boson particle in the huge amount of data produced by CERN particle collider and classify daily health activities using ensemble Methods for Multi-Class Classification.
## Instructions and Navigation
All of the code is organized into folders. Each folder starts with a number followed by the application name. For example, Chapter02.

Chapter 01 does not contain code.

The code will look like the following:
```
import org.apache.spark.ml.feature.StopWordsRemover 
val stopWords= StopWordsRemover.loadDefaultStopWords("english") ++ Array("ax", "arent", "re")
```

Code samples provided in this book use Apache Spark 2.1 and its Scala API. Furthermore, we utilize the Sparkling Water package to access the H2O machine learning library. In each chapter, we show how to start Spark using spark-shell, and also how to download the data necessary to run the code.



In summary, the basic requirements to run the code provided in this book include:

 Java 8
Spark 2.1

## Related Products
* [Machine Learning with Spark - Second Edition](https://www.packtpub.com/big-data-and-business-intelligence/machine-learning-spark-second-edition?utm_source=github&utm_medium=repository&utm_campaign=9781785889936)

* [Large Scale Machine Learning with Spark](https://www.packtpub.com/big-data-and-business-intelligence/large-scale-machine-learning-spark?utm_source=github&utm_medium=repository&utm_campaign=9781785888748)

* [Machine Learning with Spark](https://www.packtpub.com/big-data-and-business-intelligence/machine-learning-spark?utm_source=github&utm_medium=repository&utm_campaign=9781783288519)

### Suggestions and Feedback
[Click here](https://docs.google.com/forms/d/e/1FAIpQLSe5qwunkGf6PUvzPirPDtuy1Du5Rlzew23UBp2S-P3wB-GcwQ/viewform) if you have any feedback or suggestions.
