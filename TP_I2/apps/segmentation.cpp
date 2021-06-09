
#include "opencv2/opencv.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "iostream"

// include functions of static library
#include "improcfuncs.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv) {

    std::string image_path = argv[1];

    Mat img = imread(image_path, IMREAD_COLOR);

    Mat imgray;



    if(img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    cv::imshow("Display window", img);

    cv::cvtColor(img,imgray,cv::COLOR_BGR2GRAY);

    // Appel des fonctions de la bibliothèque
//    ip::point_based_accessing(imgray);

    cv::Mat imgout(imgray.rows, imgray.cols, CV_8UC1, Scalar(0));

    ip::maxGradiant(imgray);// Ajouté
    ip::Roberts_cross_filter(imgray); //Ajouté
    ip::Central_difference_filter(imgray); //Ajouté
    ip::Prewitt_filter(imgray); //Ajouté
    ip::Sobel_filter(imgray); //Ajouté
    ip::Enhancement_filter(imgray); //Ajouté
    ip::Laplacian_filter(imgray); //Ajouté
    ip::thresholdingOTSU(imgray); //Ajouté

    // attente d'une touche de clavier
    cv::waitKey(0);

    return 0;
}
