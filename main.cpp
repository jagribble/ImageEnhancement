#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

void showImage(const char *file, const char *name){
    Mat image;

    image = imread(file, CV_LOAD_IMAGE_COLOR);   // Read the file

    if(! image.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << endl ;
    }

    namedWindow( name, WINDOW_AUTOSIZE );// Create a window for display.
    imshow( name, image );                   // Show our image inside it.

    waitKey(0); // wait for the enter key to be pressed
}

void makeDFT(const char *file, const char *name){

    Mat image;

    image = imread(file, CV_LOAD_IMAGE_GRAYSCALE);   // Read the file

    if(! image.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << endl ;
    }

    Mat outputImage;

    dft(image,outputImage);

    namedWindow( name, WINDOW_AUTOSIZE );// Create a window for display.
    imshow( name, outputImage );
    waitKey(0);


}

void makeFFT(){

}


/**
 *  (0,0) in top left corner
 *
 *  Take the average from each of the 9 pixels in the neighbourhood
 *  _ _ _
 * |_|_|_|
 * |_|x|_|
 * |_|_|_|
 * **/
float getNeighbourhood(int x,int y,Mat image){
    // work out the values of each pixel in the neighbourhood
    int m11 = image.at<uchar>(x-1,y-1);
    int m12 = image.at<uchar>(x,y-1);
    int m13 = image.at<uchar>(x+1,y-1);
    int m21 = image.at<uchar>(x-1,y);
    int m22 = image.at<uchar>(x,y);
    int m23 = image.at<uchar>(x+1,y);
    int m31 = image.at<uchar>(x-1,y+1);
    int m32 = image.at<uchar>(x,y+1);
    int m33 = image.at<uchar>(x+1,y+1);
    // sum up all the pixels and take the average
    return (1/9.0)*(m11+m12+m13+m21+m22+m23+m31+m32+m33);
}

void neighbourhoodAverage(const char *file, const char *name){
    // neighbourhood averaging
    Mat image;
    Mat newImage;
    image = imread(file,CV_LOAD_IMAGE_GRAYSCALE);

    if(!image.data){
        // if the image failed to open output error message
        cout << "Image failed to open" << endl;
    } else{
        // create a new image the size of the old one with 8 bits on once channel (greyscale)
        newImage = Mat::zeros(image.rows,image.cols, CV_8UC1);
        for(int x=0;x<image.rows;x++){
            for(int y=0;y<image.cols;y++){
                // For each pixel get the neighbourhood of pixels and get the average from them
                    newImage.at<uchar>(x,y) = getNeighbourhood(x, y, image);
            }
        }
    }

    namedWindow( "neighbourhood Average image", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "neighbourhood Average image", newImage );
    namedWindow( "original image", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "original image", image );
    waitKey(0);



}

int main() {
    cout << "Hello, World!" << endl;
   // showImage("../PandaOriginal.bmp" ,  "Panda Original");
   // makeDFT("../PandaNoise.bmp","Panda Noise");
    neighbourhoodAverage("../PandaNoise.bmp","Panda Noise");
    return 0;
}