
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

    cv::namedWindow("Input Image", 1);
    cv::imshow("Input Image", img);

    cv::cvtColor(img,imgray,cv::COLOR_BGR2GRAY);

    //cv::Mat imagesorti1(imgray.rows, imgray.cols, CV_8UC1, Scalar(0)); // mes déclarations
    //cv::Mat imagesorti2(imgray.rows, imgray.cols, CV_8UC1, Scalar(0));

    //ip::minFilter(imgray,imagesorti1,3); // minFilter
    //ip::maxFilter(imagesorti1,imagesorti2,3); // pour exécuter l'image de sortie du minFilter sur les maxFilter

    // Appel des fonctions de la bibliothèque

    ip::show_histogram("hist1",imgray);

    ip::computeHistogram(imgray);

    ip::convolution(imgray);

    //ip::add_Salt_Pepper_Noise(imgray, 0.1, 0.1);
    //ip::add_Gaussian_Noise(imgray, 0.0, 4.0);
    //ip::gaussianKernel(imgray, 0.01);

    cv::Mat imgout(imgray.rows, imgray.cols, CV_8UC1, Scalar(0));
    cv::Mat maximgout(imgray.rows, imgray.cols, CV_8UC1, Scalar(0)); // image de sortie avec maxFilter
    cv::Mat minimgout(imgray.rows, imgray.cols, CV_8UC1, Scalar(0)); // image de sortie avec minFilter

    //ip::medianFilter(imgray,imgout,5);
    ip::maxFilter(imgray,maximgout,3);
    ip::minFilter(imgray,minimgout,3);
    //ip::boxFilter(imgray,minimgout,3);

    ip::robertsFilter(imgray);

    // attente d'une touche de clavier
    cv::waitKey(0);

    return 0;
}
