#!/bin/bash
g++ -std=c++11 -O3 ./Q1.cpp -o ./Q1.out
./Q1.out "$1" "$2" > ans_one.txt
rm ./Q1.out -f 2> /dev/null