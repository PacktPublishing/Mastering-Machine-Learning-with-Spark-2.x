import sbt.Keys._

name := "examples"

organization in ThisBuild := "com.github.maxpumperla"

version in ThisBuild := "0.0.1-SNAPSHOT"

scalaVersion in ThisBuild := "2.11.7"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "2.1.1",
  "org.apache.spark" %% "spark-mllib" % "2.1.1",
  "org.apache.spark" %% "spark-graphx" % "2.1.1",
  "org.apache.spark" %% "spark-sql" % "2.1.1",
)

resolvers += Resolver.url("SparkPackages", url("https://dl.bintray.com/spark-packages/maven"))
