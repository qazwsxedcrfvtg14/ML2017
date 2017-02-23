//Jtol.Linux.h v1.7.3.5-lite
#ifndef JTOL_H_
#define JTOL_H_
#include"lodepng.h"
#include<vector>
#include<string>
#include<algorithm>
#include<iostream>
#undef UNICODE
#define UNICODE
#define f first
#define s second
namespace Jtol{
    using namespace std;
    struct Color{
        unsigned char R,G,B,A;
        Color(){}
        Color(unsigned char r,unsigned char g,unsigned char b){R=r,G=g,B=b,A=255;}
        Color(unsigned char r,unsigned char g,unsigned char b,unsigned char a){R=r,G=g,B=b,A=a;}
        unsigned char L(){return max(R,max(G,B));}
        bool operator==(const Color &x){
            return R==x.R&&G==x.G&&B==x.B&&A==x.A;
            }
        };
    typedef vector<vector<Color>> Pic;
    Pic ReadPNG(string in);
    void WritePNG(string out,Pic pic);
    }
#endif
