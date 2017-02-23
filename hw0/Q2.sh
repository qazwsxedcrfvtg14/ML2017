#!/bin/bash
g++ -std=c++11 -O3 ./Q2/main.cpp ./Q2/Jtol.cpp ./Q2/lodepng.cpp -o main.out
./main.out "$1" "$2"
