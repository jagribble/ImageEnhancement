#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

void showImage(const char *file, const char *name);
// function definitions for median filtering
void medianFilter(const String &file, Mat img);
int median(int x,int y,Mat image);
void getMatrix(int x,int y,Mat image, int matrix[]);

// function definitions for gaussian smoothing
Mat gaussianSmoothing(const char *file, const char *name);
float getGaussianSmoothing(int x,int y,Mat image);
// function definitions for neighbourhood averaging
Mat neighbourhoodAverage(const char *file, const char *name);
float getNeighbourhood(int x,int y,Mat image);


int main() {
    cout << "Hello, World!" << endl;
    showImage("../PandaNoise.bmp" ,  "Panda Noise");
    // makeDFT("../PandaNoise.bmp","Panda Noise");
    Mat neigbourhoodAvg =  neighbourhoodAverage("../PandaNoise.bmp","Panda Noise");
    Mat gaussian = gaussianSmoothing("../PandaNoise.bmp","Panda Noise");
    medianFilter("",neigbourhoodAvg);
    return 0;
}

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
 * Meadian filter
 * e.g
 *  _ _ _
 * |1|2|1|
 * |2|4|6|
 * |1|2|1|
 *
 * = 1,1,1,1,2,2,2,4,6 (sorted)
 *
 * median value = 2
 *
 * **/
void medianFilter(const String &file, Mat img){
    Mat image;
    Mat newImage;
    if(img.data != NULL){
        image = img;
        newImage = Mat::zeros(image.rows,image.cols, CV_8UC1);
        for(int x=0;x<image.rows;x++){
            for(int y=0;y<image.cols;y++){

                newImage.at<uchar>(x,y) = median(x,y,image);
            }
        }
    }
    if(!file.empty() ){


    image = imread(file, CV_LOAD_IMAGE_GRAYSCALE);

    if(!image.data){
        // if the image failed to open output error message
        cout << "Image failed to open" << endl;
    } else{
        // create a new image the size of the old one with 8 bits on once channel (greyscale)
        newImage = Mat::zeros(image.rows,image.cols, CV_8UC1);
        for(int x=0;x<image.rows;x++){
            for(int y=0;y<image.cols;y++){

                newImage.at<uchar>(x,y) = median(x,y,image);
            }
        }

    }
    }
    namedWindow( "median filter", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "median filter", newImage );
    waitKey(0);
}

void getMatrix(int x,int y,Mat image, int matrix[]){
    matrix[0] = image.at<uchar>(x-1,y-1);
    matrix[1] = image.at<uchar>(x,y-1);
    matrix[2] = image.at<uchar>(x+1,y-1);
    matrix[3] = image.at<uchar>(x-1,y);
    matrix[4] = image.at<uchar>(x,y);
    matrix[5] = image.at<uchar>(x+1,y);
    matrix[6] = image.at<uchar>(x-1,y+1);
    matrix[7] = image.at<uchar>(x,y+1);
    matrix[8] = image.at<uchar>(x+1,y+1);
}

int median(int x,int y,Mat image){
    int matrix[8];
    int sortedMatrix[8];
    // get neighbourhood matrix
    getMatrix(x,y,image,matrix);
    // sort the array so you are able to grab the median
    sort(matrix,matrix+9);
    //cout << matrix[0] <<" ,"<< matrix[1] <<" ," << matrix[2] <<" ," << matrix[3] <<" ," << matrix[4] <<" ,"<< matrix[5]  <<" ,"<< matrix[6] <<" ," << matrix[7] <<" ," << matrix[8] <<endl;
    return matrix[4];
}




/**
 *  (0,0) in top left corner
 *
 *  Take the average from each of the 9 pixels in the neighbourhood
 *  _  _  _
 * |16|26|16|
 * |26|41|26|
 * |16|26|16|
 * **/


Mat gaussianSmoothing(const char *file, const char *name){
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
                newImage.at<uchar>(x,y) = getGaussianSmoothing(x, y, image);
            }
        }
    }

    namedWindow( "gaussian Smoothing image", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "gaussian Smoothing image", newImage );

    waitKey(0);

    return newImage;
}

float getGaussianSmoothing(int x,int y,Mat image){
    // work out the values of each pixel in the neighbourhood
    int m11 = image.at<uchar>(x-1,y-1)*16;
    int m12 = image.at<uchar>(x,y-1)*26;
    int m13 = image.at<uchar>(x+1,y-1)*16;
    int m21 = image.at<uchar>(x-1,y)*26;
    int m22 = image.at<uchar>(x,y)*41;
    int m23 = image.at<uchar>(x+1,y)*26;
    int m31 = image.at<uchar>(x-1,y+1)*16;
    int m32 = image.at<uchar>(x,y+1)*26;
    int m33 = image.at<uchar>(x+1,y+1)*16;
    // sum up all the pixels and take the average
    return (1/209.0)*(m11+m12+m13+m21+m22+m23+m31+m32+m33);
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
Mat neighbourhoodAverage(const char *file, const char *name){
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

    waitKey(0);
    return newImage;
}

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

