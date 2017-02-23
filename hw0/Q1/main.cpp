#include<iostream>
#include<fstream>
#include<string>
#include<sstream>
#include<vector>
#include<algorithm>
using namespace std;
template<unsigned int N,typename T=int>
struct cub{
    T a[N][N];
    int n,m;
    cub(int _n,int _m,T k=0){
        n=_n;
        m=_m;
        for(int i=0;i<n;i++)
            for(int j=0;j<m;j++)
                a[i][j]=0;
        if(k)
            for(int i=0;i<n;i++)
                a[i][i]=k;
        }
    cub(T k=0){
        n=N;
        m=N;
        for(int i=0;i<n;i++)
            for(int j=0;j<m;j++)
                a[i][j]=0;
        if(k)
            for(int i=0;i<n;i++)
                a[i][i]=k;
        }
    void rebuild(int _n=0,int _m=0,T k=0){
        n=_n;
        m=_m;
        for(int i=0;i<n;i++)
            for(int j=0;j<m;j++)
                a[i][j]=0;
        if(k)
            for(int i=0;i<n;i++)
                a[i][i]=k;
        }
    inline T* operator [](int x){
        return a[x];
        }
    cub operator +(cub b){
        if(n!=b.n||m!=b.m)throw("+ error");
        cub c(n,m);
        for(int i=0;i<n;i++)
            for(int j=0;j<m;j++)
                c[i][j]=a[i][j]+b[i][j];
        return c;
        }
    cub operator -(cub b){
        if(n!=b.n||m!=b.m)throw("- error");
        cub c(n,m);
        for(int i=0;i<n;i++)
            for(int j=0;j<m;j++)
                c[i][j]=a[i][j]-b[i][j];
        return c;
        }
    cub operator *(cub b){
        if(m!=b.n)throw("* error");
        cub c(n,b.m);
        for(int i=0;i<n;i++)
            for(int j=0;j<b.m;j++)
                for(int k=0;k<m;k++)
                    c[i][j]+=a[i][k]*b[k][j];
        return c;
        }
    cub operator *(T b){
        cub c(n,m);
        for(int i=0;i<n;i++)
            for(int j=0;j<m;j++)
                c[i][j]=a[i][j]*b;
        return c;
        }
    cub del(int x,int y){
        if(x<0||y<0||x>=n||y>=m)throw("del error");
        cub c(n-1,m-1);
        for(int i=0,ci=0;i<n;i++,ci++){
            if(i==x){ci--;continue;}
            for(int j=0,cj=0;j<m;j++,cj++){
                if(j==y){cj--;continue;}
                c[ci][cj]=a[i][j];
                }
            }
        return c;
        }
    T det(){
        if(n!=m)throw("det error");
        if(n==1)return a[0][0];
        if(n==2)return a[0][0]*a[1][1]-a[0][1]*a[1][0];
        T ret=0;
        int t=1;
        for(int i=0;i<n;i++){
            ret+=t*a[0][i]*del(0,i).det();
            t=-t;
            }
        return ret;
        }
    cub gus(){
        if(n>m)throw("gus error");
        cub mtx=*this;
        for(int i=0;i<n;i++){
            for(int j=i;j<n;j++)
                if(mtx[j][i].p){
                    for(int k=0;k<m;k++)
                        swap(mtx[i][k],mtx[j][k]);
                    break;
                    }
            if(!mtx[i][i].p)continue;
            for(int k=n;k>=i;k--)
                mtx[i][k]=mtx[i][k]/mtx[i][i];
            for(int j=i+1;j<n;j++)
                for(int k=m-1;k>=i;k--)
                    mtx[j][k]=mtx[j][k]-mtx[i][k]*mtx[j][i];
            }
        for(int i=n-1;i>=0;i--){
            for(int j=i-1;j>=0;j--){
                for(int k=n;k<m;k++)
                    mtx[j][k]=mtx[j][k]-mtx[i][k]*mtx[j][i];
                mtx[j][i]=0;
                }
            }
        return mtx;
        }
    cub Tr(){
        cub c(m,n);
        for(int i=0;i<n;i++){
            for(int j=0;j<m;j++){
                c[j][i]=a[i][j];
                }
            }
        return c;
        }
    cub operator -(){
        if(!det())throw("~ error");
        cub d(n,n);
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                d[i][j]=del(j,i).det()*(((i+j)&1)?-1:1);
                }
            }
        return d;
        }
    cub operator ~(){
        return (-*this)*(1.0/det());
        }
    cub operator /(cub b){
        return (*this*(-b));//*b.det();
        }
    cub operator %(cub b){
        return ((-*this)*b);//*det();
        }
    cub operator ^(long long int b){
        cub ret(n,m,1),now=*this;
        while(b){
            if(b&1)ret=ret*now;
            now=now*now;
            b>>=1;
            }
        return ret;
        }
    void print(){
        printf("[ ");
        for(int i=0;i<n;i++){
            if(i)printf("; ");
            for(int j=0;j<m;j++)
                printf("%d ",a[i][j]);
            }
        printf("]\n");
        }
    };
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
cub<50> a(1,50),b(50,10),c(1,10);
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
    for(int i=0;i<1;i++){
        getline(fin,s);
        auto v=split(s,",");
        for(int j=0;j<50;j++)
            a[i][j]=StrToInt(v[j]);
        }
    fin.close();
    fin.open(fb,ios::in);
    if(!fin.is_open()){
        fprintf(stderr,"%s not found!\n",fb.c_str());
        exit(0);
        }
    for(int i=0;i<50;i++){
        getline(fin,s);
        auto v=split(s,",");
        for(int j=0;j<10;j++)
            b[i][j]=StrToInt(v[j]);
        }
    fin.close();
    c=a*b;
    sort(c[0],c[0]+10);
    for(int i=0;i<1;i++)
        for(int j=0;j<10;j++)
            printf("%d\n",c[i][j]);
    return 0;
    }
