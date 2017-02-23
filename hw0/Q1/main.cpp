#include<iostream>
#include<fstream>
#include<string>
#include<sstream>
#include<vector>
#include<algorithm>
using namespace std;
vector<string>split(string s,string cut){
    vector<string>ve;
    if(s=="")return ve;
    auto clen=cut.length();
    while(true){
        auto pos=s.find(cut);
        if(pos==string::npos){
            ve.push_back(s);
            break;
            }
        else{
            if(pos)
                ve.push_back(s.substr(0,pos));
            else
                ve.push_back("");
            if(pos+clen!=s.length())
                s=s.substr(pos+clen);
            else
                s="";
            }
        }
    return ve;
    }
int StrToInt(string x){
    stringstream str;
    int s;
    str<<x;
    str>>s;
    return s;
    }
vector<vector<int>>a,b,c;
int main(int argc,char** argv){
    string fa,fb;
    if(argv[1]) fa=argv[1];
    else fa="matrixA.txt";
    if(argv[2]) fb=argv[2];
    else fb="matrixB.txt";
    if(fa=="")fa="matrixA.txt";
    if(fb=="")fb="matrixB.txt";
    fstream fin;
    string s;
    fin.open(fa,ios::in);
    if(!fin.is_open()){
        fprintf(stderr,"%s not found!\n",fa.c_str());
        exit(0);
        }
    for(int i=0;getline(fin,s)&&s!="";i++){
        auto v=split(s,",");
        a.push_back(vector<int>());
        for(auto &x:v)
            a.back().push_back(StrToInt(x));
        }
    fin.close();
    fin.open(fb,ios::in);
    if(!fin.is_open()){
        fprintf(stderr,"%s not found!\n",fb.c_str());
        exit(0);
        }
    for(int i=0;getline(fin,s)&&s!="";i++){
        auto v=split(s,",");
        b.push_back(vector<int>());
        for(auto &x:v)
            b.back().push_back(StrToInt(x));
        }
    fin.close();
    if(a[0].size()!=b.size()){
        fprintf(stderr,"Matrix is not multipliable!\n");
        exit(0);
        }
    vector<int>ans;
    for(unsigned int i=0;i<a.size();i++){
        //c.push_back(vector<int>());
        for(unsigned int j=0;j<b[0].size();j++){
            //c[i].push_back(0);
            ans.push_back(0);
            for(unsigned int k=0;k<b.size();k++){
                //c[i][j]+=a[i][k]*b[k][j];
                ans.back()+=a[i][k]*b[k][j];
                }
            }
        }
    sort(ans.begin(),ans.end());
    for(int x:ans)
        printf("%d\n",x);
    return 0;
    }
