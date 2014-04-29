#! /bin/bash

for n in 4 16 64 256 1024
do
    cd nehalem-matmul-$n
    zsim nehalem-matmul-$n.cfg
    cd ..
done
