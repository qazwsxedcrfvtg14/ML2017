#define _USE_MATH_DEFINES
#include"Jtol.h"
using namespace Jtol;
using namespace std;
int main(int argc,char** argv){
    string fa,fb;
    if(argv[1]) fa=argv[1];
    else fa="lena.png";
    if(argv[2]) fb=argv[2];
    else fb="lena_modified.png";
    if(fa=="")fa="lena.png";
    if(fb=="")fb="lena_modified.png";
    Pic a=ReadPNG(fa),b=ReadPNG(fb);
    if(a.size()==0||b.size()==0||a[0].size()==0||b[0].size()==0){
        puts("Bad PNG!");
        exit(0);
        }
    if(a.size()!=b.size()||a[0].size()!=b[0].size()){
        puts("File size not match!");
        exit(0);
        }
    for(unsigned int i=0;i<a.size();i++)
        for(unsigned int j=0;j<a[i].size();j++)
            if(a[i][j]==b[i][j])
                b[i][j]=Color(0,0,0,0);
    WritePNG("ans_two.png",b);
    return 0;
    }
