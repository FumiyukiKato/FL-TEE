#! /bin/bash
set -ex

# for n_users in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 25 30 40 50 60 80 100 200 300 400 500 1000 3000
# do
# bin/bench -v -a optimized -c 10000 -d 50890 -k 5089 -t 2 --sampling_ratio=0.3 --optimal_num_of_clients=$n_users
# done

# bin/bench -v -a non_oblivious -c 10000 -d 50890 -k 5089 -t 2 --sampling_ratio=0.3
# bin/bench -v -a baseline -c 10000 -d 50890 -k 5089 -t 2 --sampling_ratio=0.3
# bin/bench -v -a advanced -c 10000 -d 50890 -k 5089 -t 2 --sampling_ratio=0.3



for n_users in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 25 30 40 50 60 80 100 200 300 400 500 1000 2000 3000
do
bin/bench -v -a optimized -c 10000 -d 201588 -k 2015 -t 2 --sampling_ratio=0.3 --optimal_num_of_clients=$n_users
done

bin/bench -v -a non_oblivious -c 10000 -d 201588 -k 2015 -t 2 --sampling_ratio=0.3
bin/bench -v -a baseline -c 10000 -d 201588 -k 2015 -t 2 --sampling_ratio=0.3
bin/bench -v -a advanced -c 10000 -d 201588 -k 2015 -t 2 --sampling_ratio=0.3