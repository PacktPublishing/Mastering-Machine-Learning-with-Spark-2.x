#!/usr/bin/env bash
set -e

TOPDIR=$(cd `dirname $0`/..; pwd)
DATADIR="$TOPDIR/data"
DATAFILE="$DATADIR/data.zip"

[ -d "$DATADIR" ] || mkdir -p "$DATADIR"

echo "Downloading Chapter 3 dataset to $DATADIR"
curl -s https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip -o "$DATAFILE"
( 
  cd "$DATADIR"
  unzip -j "$DATAFILE" "*/Optional/*.dat"
)
echo "Done"
