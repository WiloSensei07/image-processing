
#include "opencv2/opencv.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "iostream"

// include functions of static library
#include "improcfuncs.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{

    std::string image_path = argv[1];

    Mat imgcol = imread(image_path, IMREAD_COLOR);

    //Mat imgray;

    if(imgcol.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    //exemple de déclaration d'une srucrure matricielle de couleur de 03 canaux

    cv::Mat imgblue(imgcol.rows, imgcol.cols, CV_8UC3);
    cv::Mat imggreen(imgcol.rows, imgcol.cols, CV_8UC3);
    cv::Mat imgred(imgcol.rows, imgcol.cols, CV_8UC3);

    //exemple de colour slice
    ip::colourslice(imgcol, imgblue, imggreen, imgred);
    cv::namedWindow("display blue channel", WINDOW_AUTOSIZE);
    cv::imshow("display blue channel", imgblue);
    cv::namedWindow("display green channel", WINDOW_AUTOSIZE);
    cv::imshow("display green channel", imggreen);
    cv::namedWindow("display red channel", WINDOW_AUTOSIZE);
    cv::imshow("display red channel", imgred);

    //conversion de RGV à nuance de gris
    cv::Mat imgray(imgcol.rows, imgcol.cols, CV_8UC1, Scalar(0));
    ip::convert2gray(imgcol, imgray);
    cv::namedWindow("display gray channel", WINDOW_AUTOSIZE);
    cv::imshow("display gray channel", imgray);

    //conversion RGV à HSI
    cv::Mat imghsi(imgcol.rows, imgcol.cols, CV_8UC3);
    ip::convert2HSI(imgcol, imghsi);
    cv::namedWindow("display hsi channel", WINDOW_AUTOSIZE);
    cv::imshow("display hsi channel", imghsi);


    //une variation d'OPENCV
    Mat cvimghsi;
    cv::cvtColor(imgcol, cvimghsi, cv::COLOR_BGR2HSV);
    cv::namedWindow("display opencv hsi channel", 1);
    cv::imshow("display opencv hsi channel", imghsi);

cv::waitKey(0);
}
