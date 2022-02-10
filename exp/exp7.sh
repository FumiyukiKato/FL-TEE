#! /bin/bash
set -ex

src/bin/bench -v -a non_oblivious -c 100 -d 10000 -k 100 -t 1
src/bin/bench -v -a advanced -c 100 -d 10000 -k 100 -t 1
src/bin/bench -v -a baseline -c 100 -d 10000 -k 100 -t 1
src/bin/bench -v -a path_oram -c 100 -d 10000 -k 100 -t 1

src/bin/bench -v -a non_oblivious -c 100 -d 50000 -k 500 -t 1
src/bin/bench -v -a advanced -c 100 -d 50000 -k 500 -t 1
src/bin/bench -v -a baseline -c 100 -d 50000 -k 500 -t 1
src/bin/bench -v -a path_oram -c 100 -d 50000 -k 500 -t 1

src/bin/bench -v -a non_oblivious -c 100 -d 100000 -k 1000 -t 1
src/bin/bench -v -a advanced -c 100 -d 100000 -k 1000 -t 1
src/bin/bench -v -a baseline -c 100 -d 100000 -k 1000 -t 1
src/bin/bench -v -a path_oram -c 100 -d 100000 -k 1000 -t 1

src/bin/bench -v -a non_oblivious -c 100 -d 500000 -k 5000 -t 1
src/bin/bench -v -a advanced -c 100 -d 500000 -k 5000 -t 1
src/bin/bench -v -a baseline -c 100 -d 500000 -k 5000 -t 1
src/bin/bench -v -a path_oram -c 100 -d 500000 -k 5000 -t 1

src/bin/bench -v -a non_oblivious -c 100 -d 1000000 -k 10000 -t 1
src/bin/bench -v -a advanced -c 100 -d 1000000 -k 10000 -t 1
src/bin/bench -v -a baseline -c 100 -d 1000000 -k 10000 -t 1
src/bin/bench -v -a path_oram -c 100 -d 1000000 -k 10000 -t 1

src/bin/bench -v -a non_oblivious -c 100 -d 5000000 -k 50000 -t 1
src/bin/bench -v -a advanced -c 100 -d 5000000 -k 50000 -t 1
src/bin/bench -v -a baseline -c 100 -d 5000000 -k 50000 -t 0
src/bin/bench -v -a path_oram -c 100 -d 5000000 -k 50000 -t 0

src/bin/bench -v -a non_oblivious -c 100 -d 10000000 -k 100000 -t 0
src/bin/bench -v -a advanced -c 100 -d 10000000 -k 100000 -t 0
src/bin/bench -v -a baseline -c 100 -d 10000000 -k 100000 -t 0
src/bin/bench -v -a path_oram -c 100 -d 10000000 -k 100000 -t 0