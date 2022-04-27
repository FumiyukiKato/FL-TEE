#! /bin/bash
set -ex

# bin/bench -v -a non_oblivious -c 100 -d 10000 -k 100 -t 1
# bin/bench -v -a advanced -c 100 -d 10000 -k 100 -t 1
# bin/bench -v -a baseline -c 100 -d 10000 -k 100 -t 1
# bin/bench -v -a path_oram -c 100 -d 10000 -k 100 -t 1

# bin/bench -v -a non_oblivious -c 100 -d 50000 -k 500 -t 1
# bin/bench -v -a advanced -c 100 -d 50000 -k 500 -t 1
# bin/bench -v -a baseline -c 100 -d 50000 -k 500 -t 1
# bin/bench -v -a path_oram -c 100 -d 50000 -k 500 -t 1

# bin/bench -v -a non_oblivious -c 100 -d 100000 -k 1000 -t 1
# bin/bench -v -a advanced -c 100 -d 100000 -k 1000 -t 1
# bin/bench -v -a baseline -c 100 -d 100000 -k 1000 -t 1
# bin/bench -v -a path_oram -c 100 -d 100000 -k 1000 -t 1

# bin/bench -v -a non_oblivious -c 100 -d 500000 -k 5000 -t 1
# bin/bench -v -a advanced -c 100 -d 500000 -k 5000 -t 1
# bin/bench -v -a baseline -c 100 -d 500000 -k 5000 -t 1
# bin/bench -v -a path_oram -c 100 -d 500000 -k 5000 -t 1

# bin/bench -v -a non_oblivious -c 100 -d 1000000 -k 10000 -t 1
# bin/bench -v -a advanced -c 100 -d 1000000 -k 10000 -t 1
# bin/bench -v -a baseline -c 100 -d 1000000 -k 10000 -t 1
# bin/bench -v -a path_oram -c 100 -d 1000000 -k 10000 -t 1

# bin/bench -v -a non_oblivious -c 100 -d 5000000 -k 50000 -t 1
# bin/bench -v -a advanced -c 100 -d 5000000 -k 50000 -t 1
# bin/bench -v -a baseline -c 100 -d 5000000 -k 50000 -t 1
# bin/bench -v -a path_oram -c 100 -d 5000000 -k 50000 -t 1

# bin/bench -v -a non_oblivious -c 100 -d 10000000 -k 100000 -t 1
# bin/bench -v -a advanced -c 100 -d 10000000 -k 100000 -t 1
# bin/bench -v -a baseline -c 100 -d 10000000 -k 100000 -t 1
# bin/bench -v -a path_oram -c 100 -d 10000000 -k 100000 -t 1

# # index-privacy r=10
# bin/bench -v -a non_oblivious -c 100 -d 10000 -k 1000 -t 1
# bin/bench -v -a non_oblivious -c 100 -d 50000 -k 5000 -t 1
# bin/bench -v -a non_oblivious -c 100 -d 100000 -k 10000 -t 1
# bin/bench -v -a non_oblivious -c 100 -d 500000 -k 50000 -t 1
# bin/bench -v -a non_oblivious -c 100 -d 1000000 -k 100000 -t 1
# bin/bench -v -a non_oblivious -c 100 -d 5000000 -k 500000 -t 1
# bin/bench -v -a non_oblivious -c 100 -d 10000000 -k 1000000 -t 1

# bin/bench -v -a advanced_simple -c 100 -d 10000 -k 100 -t 2
# bin/bench -v -a advanced_simple -c 100 -d 50000 -k 500 -t 2
# bin/bench -v -a advanced_simple -c 100 -d 100000 -k 1000 -t 2
# bin/bench -v -a advanced_simple -c 100 -d 500000 -k 5000 -t 2
# bin/bench -v -a advanced_simple -c 100 -d 1000000 -k 10000 -t 2
bin/bench -v -a advanced_simple -c 100 -d 5000000 -k 50000 -t 1
bin/bench -v -a advanced_simple -c 100 -d 10000000 -k 100000 -t 1