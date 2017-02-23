//Jtol.Linux.cpp v1.7.3.5-lite
#include"lodepng.h"
#include"Jtol.h"
#undef UNICODE
#define UNICODE
#define f first
#define s second
//#pragma comment(linker, "/subsystem:console /entry:WinMainCRTStartup")
namespace Jtol{
    using namespace std;
    Pic ReadPNG(string in){
        std::vector<unsigned char> png;
        std::vector<unsigned char> image; //the raw pixels
        unsigned width, height;
        lodepng::State state; //optionally customize this one
        unsigned error = lodepng::load_file(png, in.c_str()); //load the image file with given filename
        if(!error) error = lodepng::decode(image, width, height, state, png);
        //State state contains extra information about the PNG such as text chunks, ...
        //if there's an error, display it
        if(error) std::cout << "decoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
        //the pixels are now in the vector "image", 4 bytes per pixel, ordered RGBARGBA..., use it as texture, draw it, ...
        //printf("%d %d\n",width,height);
        Pic pic;
        pic.resize(height);
        for(unsigned int i=0;i<height;i++){
            pic[i].resize(width);
            for(unsigned int j=0;j<width;j++)
                pic[i][j]=Color(image[(i*width+j)*4],image[(i*width+j)*4+1],image[(i*width+j)*4+2],image[(i*width+j)*4+3]);
            }
        return pic;
        }
    void WritePNG(string out,Pic pic){
        std::vector<unsigned char> image;
        unsigned width=pic[0].size();
        unsigned height=pic.size();
        for(unsigned int i=0;i<height;i++)
            for(unsigned int j=0;j<width;j++)
                image.push_back(pic[i][j].R),
                image.push_back(pic[i][j].G),
                image.push_back(pic[i][j].B),
                image.push_back(pic[i][j].A);
        std::vector<unsigned char> png;
        lodepng::State state; //optionally customize this one
        unsigned error = lodepng::encode(png, image, width, height, state);
        if(!error) lodepng::save_file(png,out.c_str());
        //if there's an error, display it
        if(error) std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
        }
    }
