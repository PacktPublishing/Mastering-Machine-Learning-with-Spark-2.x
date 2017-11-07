#!/usr/bin/env bash
set -e

TOPDIR=$(cd `dirname $0`/..; pwd)
DATADIR="$TOPDIR/data"
DATAFILE="$DATADIR/higgs100k.csv"
ROWS=100000

[ -d $DATADIR ] || mkdir -p $DATADIR

echo "Downloading $ROWS rows of HIGGS dataset to $DATAFILE"
curl -s https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz | gunzip - | head -n$ROWS > $DATAFILE
echo "Done"
