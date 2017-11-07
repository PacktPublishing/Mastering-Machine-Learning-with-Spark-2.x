#!/usr/bin/env bash
set -e

TOPDIR=$(cd `dirname $0`/..; pwd)
DATADIR="$TOPDIR/data"
DATADIR="/tmp/data"
DATAFILE="$DATADIR/data.zip"

[ -d "$DATADIR" ] || mkdir -p "$DATADIR"

echo "Downloading Chapter 3 dataset to $DATADIR"
curl -s "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz" -o "$DATAFILE"
( 
  cd "$DATADIR"
  tar -xf "$DATAFILE"
)
echo "Done"
