#!/bin/bash
g++ -std=c++11 -O3 ./Q2.cpp ./Jtol.cpp ./lodepng.cpp -o Q2.out
./Q2.out "$1" "$2"
rm ./Q2.out -f 2> /dev/null
