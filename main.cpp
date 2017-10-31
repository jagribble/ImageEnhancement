#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

void showImage(Mat image, const char *name);
float meanSquaredError(Mat original, Mat filtered);

//function defintions for dft
Mat makeDFT(Mat I);
Mat inverseDFT(Mat source);
Mat createMask(int width,int height,int xSize, int ySize,int lineSize);
void swapQuadrants(Mat& img);
Mat applyFilter(Mat maskFilter,Mat dft);
Mat frequencyDomainLowPass(Mat original, int size);

//function defintions for image sharpening
Mat imageSharpening(Mat image,int threshold);
float sobelMask(int x,int y, Mat image,int threshold);

// function definitions for image Averaging
Mat imageAveraging(Mat images[],int n);
float pixelAverage(int x, int y, Mat images[],int n);

// function definitions for median filtering
Mat medianFilter(Mat img);
int median(int x,int y,Mat image);
void getMatrix(int x,int y,Mat image, int matrix[]);

// function definitions for gaussian smoothing
Mat gaussianSmoothing(Mat image);
float getGaussianSmoothing(int x,int y,Mat image);
// function definitions for neighbourhood averaging
Mat neighbourhoodAverage(Mat image);
float getNeighbourhood(int x,int y,Mat image);


int main() {
    cout << "Hello, World!" << endl;
    Mat imageNoise = imread("../PandaNoise.bmp",CV_LOAD_IMAGE_GRAYSCALE);
    Mat imageOriginal = imread("../PandaOriginal.bmp",CV_LOAD_IMAGE_GRAYSCALE);
    showImage(imageNoise ,  "Panda Noise");
   // Mat dftOriginal = makeDFT(imageOriginal);

    frequencyDomainLowPass(imageNoise,30);
    // makeDFT("../PandaNoise.bmp","Panda Noise");
    cout << "neighbourhoodAverage"<<endl;
    Mat neigbourhoodAvg =  neighbourhoodAverage(imageNoise);
    meanSquaredError(imageOriginal,neigbourhoodAvg);
    cout << "gaussianSmoothing"<<endl;
    Mat gaussian = gaussianSmoothing(imageNoise);
    meanSquaredError(imageOriginal,gaussian);
    cout << "meadian filtering"<<endl;
    Mat median = medianFilter(neigbourhoodAvg);
    meanSquaredError(imageOriginal,median);
    cout << "image sharpening"<<endl;
    Mat sharpen = imageSharpening(median,85);
    meanSquaredError(imageOriginal,sharpen);
    cout << "image averaging"<<endl;
    Mat images[4] = {neigbourhoodAvg,gaussian,median,sharpen};
    Mat average = imageAveraging(images,3);
    meanSquaredError(imageOriginal,average);

    showImage(imageOriginal,"Panda original");
    meanSquaredError(imageOriginal,imageNoise);
    return 0;
}

void showImage(Mat image, const char *name){

    namedWindow( name, WINDOW_AUTOSIZE );// Create a window for display.
    imshow( name, image );                   // Show our image inside it.

    waitKey(0); // wait for the enter key to be pressed
}



// https://docs.opencv.org/2.4/doc/tutorials/core/discrete_fourier_transform/discrete_fourier_transform.html
Mat makeDFT(Mat I){
    Mat padded;                            //expand input image to optimal size
    int m = getOptimalDFTSize( I.rows );
    int n = getOptimalDFTSize( I.cols ); // on the border add zero values
    copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));

    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

    dft(complexI, complexI);            // this way the result may fit in the source matrix

    // compute the magnitude and switch to logarithmic scale
    // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
    split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
    Mat magI = planes[0];

    magI += Scalar::all(1);                    // switch to logarithmic scale
    log(magI, magI);

    swapQuadrants(magI);


    normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
    // viewable image form (float between values 0 and 1).


    imshow("spectrum magnitude", magI);
    waitKey();

    return complexI;

}

void swapQuadrants(Mat& img){
// crop the spectrum, if it has an odd number of rows or columns
    img = img(Rect(0, 0, img.cols & -2, img.rows & -2));

    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = img.cols/2;
    int cy = img.rows/2;

    Mat q0(img, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(img, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(img, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(img, Rect(cx, cy, cx, cy)); // Bottom-Right

    Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
}



/**
 * Inverse DFT function
 * @param source Mat returned from dft with both Complex and real components
 * **/
Mat inverseDFT(Mat source){
    Mat inverse = Mat::zeros(source.rows,source.cols, CV_8UC1);
    dft(source,inverse,DFT_INVERSE|DFT_REAL_OUTPUT|DFT_SCALE);
    normalize(inverse, inverse, 0, 1, CV_MINMAX);
    imshow("inverse dft",inverse);
    waitKey();
    return inverse;

}

/**
 * Create a mask at a point and size.
 * @param lineSize if -1 filled in
 * **/
Mat createMask(int width,int height,int xSize, int ySize,int lineSize){
    Mat mask = Mat::zeros(width, height, CV_32F);;
    ellipse(mask, Point(width/2, height/2), Size(xSize, ySize), 0, 0, 360, Scalar(255, 0, 0), lineSize, 8);
    normalize(mask, mask, 0, 1, CV_MINMAX);
    return mask;
}

Mat applyFilter(Mat maskFilter,Mat dft){
    Mat result;
    // multiply the dft and the mask together
    mulSpectrums(dft, maskFilter, result, 0);
    return result;
}

Mat frequencyDomainLowPass(Mat original, int size){
    // get the dft of the image
    Mat dft = makeDFT(original);
    // create the low pass mask
    Mat mask = createMask(original.rows,original.cols,size,size,-1);
    imshow("mask",mask);
    waitKey();
    // swap the quedarants of the mask to match the dft
    swapQuadrants(mask);
    imshow("swaped quadrant mask",mask);
    waitKey();
    Mat padded;                            //expand input image to optimal size
    int m = getOptimalDFTSize( mask.rows );
    int n = getOptimalDFTSize( mask.cols ); // on the border add zero values
    copyMakeBorder(mask, padded, 0, m - mask.rows, 0, n - mask.cols, BORDER_CONSTANT, Scalar::all(0));
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat maskComplex;
    merge(planes, 2, maskComplex);
    // apply the filter to the dft
    Mat filteredDFT = applyFilter(maskComplex,dft);

    //take the inverse of the filtered DFT
    inverseDFT(filteredDFT);
    return filteredDFT;
}


/**
 * Mean-Squared-Error (MSE)
 * For each pixel of image A and B:
 *      difference = A(x,y) - B(x,y)
 *      sum += difference^2
 *
 * MSE = sum/(width x height)
 * **/
float meanSquaredError(Mat original, Mat filtered){
    int sum = 0;
    int no = 0;
    for (int x=0;x<original.rows;x++){
        for(int y=0;y<original.cols;y++){
            int difference = original.at<uchar>(x,y)-filtered.at<uchar>(x,y);
            sum+= pow(difference,2);
            no++;
        }
    }
    float mse = sum/(original.rows * original.cols);
    cout << "MSE = " << mse <<endl;
    cout << "Max MSE = " << (pow(255,2)*no)/(original.rows * original.cols)<<endl;
    return mse;
}



void makeFFT(){

}

/**
 * Image Sharpening
 *
 *Sobel algorithm used
 *
 *  _  _  _
 * |-1|-2|-1|
 * | 0| 0| 0|
 * | 1| 2| 1|
 *  _  _  _
 * |-1| 0| 1|
 * |-2| 0| 2|
 * |-1| 0| 1|
 *
 * **/
Mat imageSharpening(Mat image,int threshold){
    Mat newImage;
    Mat edgeImage;
    newImage = Mat::zeros(image.rows,image.cols, CV_8UC1);
    edgeImage = Mat::zeros(image.rows,image.cols, CV_8UC1);
    // create edge outline of image
    for(int x=0;x<image.rows;x++){
        for(int y=0;y<image.cols;y++){
            edgeImage.at<uchar>(x,y) = sobelMask(x,y,image,threshold);
        }
    }
    namedWindow( "edge mask", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "edge mask", edgeImage );
    waitKey(0);
    // add the outline to the image with a scaling factor of 0.25 to not approach the max value
    for(int x=0;x<image.rows;x++){
        for(int y=0;y<image.cols;y++){
            newImage.at<uchar>(x,y) = image.at<uchar>(x,y) + (edgeImage.at<uchar>(x,y)*0.25) ;
        }
    }
    namedWindow( "sharpened image", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "sharpened image", newImage );
    waitKey(0);

    return newImage;
}

/**create the outline of the image
 *  _ _ _
 * |a|b|c|
 * |d|e|f|
 * |g|h|i|
 *
 * S(e) = 1/8 [|(a + 2b + c)-(g + 2h +i)|]
+ |(a + 2d + g) - (c + 2f + i)|]
 * **/
float sobelMask(int x,int y, Mat image,int threshold){
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
    // if the pixel is less than the threshold then make the pixel at (x,y) 0
    if(m22<threshold){
        return 0;
    }
    // cout << m11 <<" ,"<< m12 <<" ," << m13 <<" ," << m21 <<" ," << m22 <<" ,"<< m23  <<" ,"<< m31 <<" ," << m32 <<" ," << m33 <<endl;
    //cout<< (1/8)*abs((m11+(2*m12)+m13)-(m31+(2*m32)+m33))+abs((m11+(2*m21)+m31)-(m13+(2*m23)+m33)) <<endl;
    // get the sobel operation using the equation and matrix data
    float sobelOp = (1/8)*abs((m11+(2*m12)+m13)-(m31+(2*m32)+m33))+abs((m11+(2*m21)+m31)-(m13+(2*m23)+m33));
    return sobelOp;
}


/**
 * Image Averaging
 *
 * Many images pixel added together and divided by the amount of noisy pictures
 *
 * g(x,y) = 1/N*(f(x,y)+n(x,y)+..)
 * **/
Mat imageAveraging(Mat images[],int n){
    Mat newImage;
    newImage = Mat::zeros(images[0].rows,images[0].cols, CV_8UC1);
    for(int x=0;x<images[0].rows;x++){
        for(int y=0;y<images[0].cols;y++){
            newImage.at<uchar>(x,y) = pixelAverage(x,y,images,n);
        }
    }
    namedWindow( "image averaging", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "image averaging", newImage );
    waitKey(0);
    return newImage;
}

float pixelAverage(int x, int y, Mat images[],int n){
    int total = 0;
    for(int i=0;i<n;i++){
        total += images[i].at<uchar>(x,y);
    }
    return total/n;
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
Mat medianFilter(Mat img){
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

    namedWindow( "median filter", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "median filter", newImage );
    waitKey(0);
    return newImage;
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


Mat gaussianSmoothing(Mat image){
    // neighbourhood averaging
    Mat newImage;

    // create a new image the size of the old one with 8 bits on once channel (greyscale)
    newImage = Mat::zeros(image.rows,image.cols, CV_8UC1);
    for(int x=0;x<image.rows;x++){
        for(int y=0;y<image.cols;y++){
            // For each pixel get the neighbourhood of pixels and get the average from them
            newImage.at<uchar>(x,y) = getGaussianSmoothing(x, y, image);
        }
    }

    namedWindow( "gaussian Smoothing image", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "gaussian Smoothing image", newImage );

    waitKey(0);

    return newImage;
}


/**
 * _  _  _  _
 * |1 |4 |7 |4 |1 |
 * |4 |16|26|16|4 |
 * |7 |26|41|26|7 |
 * |4 |16|26|16|4 |
 * |1 |4 |7 |4 |1 |
 * */
float getGaussianSmoothing(int x,int y,Mat image){
    // work out the values of each pixel in the neighbourhood
    int m11 = image.at<uchar>(x-2,y-2)*1;
    int m12 = image.at<uchar>(x-1,y-2)*4;
    int m13 = image.at<uchar>(x,y-2)*7;
    int m14 = image.at<uchar>(x+1,y-2)*4;
    int m15 = image.at<uchar>(x+2,y-2)*1;
    int m21 = image.at<uchar>(x-2,y-1)*4;
    int m22 = image.at<uchar>(x-1,y-1)*16;
    int m23 = image.at<uchar>(x,y-1)*26;
    int m24 = image.at<uchar>(x+1,y-1)*16;
    int m25 = image.at<uchar>(x+2,y-1)*4;
    int m31 = image.at<uchar>(x-2,y)*7;
    int m32 = image.at<uchar>(x-1,y)*26;
    int m33 = image.at<uchar>(x,y)*41;
    int m34 = image.at<uchar>(x+1,y)*26;
    int m35 = image.at<uchar>(x+2,y)*7;
    int m41 = image.at<uchar>(x-2,y)*4;
    int m42 = image.at<uchar>(x-1,y+1)*16;
    int m43 = image.at<uchar>(x,y+1)*26;
    int m44 = image.at<uchar>(x+1,y+1)*16;
    int m45 = image.at<uchar>(x+2,y+1)*4;
    int m51 = image.at<uchar>(x-2,y+2)*1;
    int m52 = image.at<uchar>(x-1,y+2)*4;
    int m53 = image.at<uchar>(x,y+2)*7;
    int m54 = image.at<uchar>(x+1,y+2)*4;
    int m55 = image.at<uchar>(x+2,y+2)*1;
    // sum up all the pixels and take the average
    return (1/273.0)*(m11+m12+m13+m14+m15+m21+m22+m23+m24+m25+m31+m32+m33+m34+m35+m41+m42+m43+m44+m45+m51+m52+m53+m54+m55);
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
Mat neighbourhoodAverage(Mat image){
    // neighbourhood averaging

    Mat newImage;
    // create a new image the size of the old one with 8 bits on once channel (greyscale)
    newImage = Mat::zeros(image.rows,image.cols, CV_8UC1);
    for(int x=0;x<image.rows;x++){
        for(int y=0;y<image.cols;y++){
            // For each pixel get the neighbourhood of pixels and get the average from them
                newImage.at<uchar>(x,y) = getNeighbourhood(x, y, image);
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

