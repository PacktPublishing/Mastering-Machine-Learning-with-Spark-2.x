## [Get this title for $10 on Packt's Spring Sale](https://www.packt.com/B04833?utm_source=github&utm_medium=packt-github-repo&utm_campaign=spring_10_dollar_2022)
-----
For a limited period, all eBooks and Videos are only $10. All the practical content you need \- by developers, for developers

# Mastering Machine Learning with Spark 2.x
This is the code repository for [Mastering Machine Learning with Spark 2.x](https://www.packtpub.com/big-data-and-business-intelligence/mastering-machine-learning-spark-2x?utm_source=github&utm_medium=repository&utm_campaign=9781785283451), published by [Packt](https://www.packtpub.com/?utm_source=github). It contains all the supporting project files necessary to work through the book from start to finish.


## About the Book
This book gives you access to transform data into actionable knowledge.
The book commences by defining machine learning primitives by the MLlib and H2O libraries.
You will learn how to use binary classification to detect the Higgs Boson particle in the huge
amount of data produced by CERN particle collider or classify daily health activities
using ensemble methods.


## Instructions and Navigation
All of the code is organized into folders. Each folder starts with a number followed by the application name. For example, Chapter02.

The repository includes the following chapters:
  - Chapter 2: Detecting Dark Matter - The Higgs-Boson Particle
  - Chapter 3: Ensemble Methods for Multi-class Classification
  - Chapter 4: Predicting Movie Reviews Using NLP and Spark Streaming
  - Chapter 5: Word2Vec for Prediction and Clustering
  - Chapter 6: Extracting Patterns from Clickstream Data
  - Chapter 7: Graph Analytics with GraphX
  - Chapter 8: Lending Club Loan Prediction

> Note: Chapter 01 does not contain code.

Code samples provided in this book use Apache Spark 2.1 and its Scala API.
Furthermore, we utilize the Sparkling Water package to access the H2O machine learning library.
In each chapter, we show how to start Spark using spark-shell, and also how to download the data
necessary to run the code.
Moreover, each chapter also contains code representing a regular Spark application.

## Requirements
In summary, the basic requirements to run the code provided in this book include:

  - Java 8
  - Spark 2.1

## Building
The project utilizes Gradle as build system. To build it, it is necessary to run:

```shell
> ./gradlew build
```

To list all project, you can use:

```shell
> ./gradlew projects
```

## Running
Each individual example can be run in the way it is described in the book, or directly
via Gradle, for example:

```shell
> ./gradlew :mastering-ml-w-spark-chapter02:run
```

## Related Products
* [Machine Learning with Spark - Second Edition](https://www.packtpub.com/big-data-and-business-intelligence/machine-learning-spark-second-edition?utm_source=github&utm_medium=repository&utm_campaign=9781785889936)

* [Large Scale Machine Learning with Spark](https://www.packtpub.com/big-data-and-business-intelligence/large-scale-machine-learning-spark?utm_source=github&utm_medium=repository&utm_campaign=9781785888748)

* [Machine Learning with Spark](https://www.packtpub.com/big-data-and-business-intelligence/machine-learning-spark?utm_source=github&utm_medium=repository&utm_campaign=9781783288519)
