#!/bin/bash
g++ -std=c++11 -O3 ./Q1/main.cpp -o main.out
./main.out "$1" "$2" > ans_one.txt