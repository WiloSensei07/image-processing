
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


    //ip::log_transform(imgray);

    //ip::exp_transform(imgray);

    //ip::gamma(imgray,1.3);

    ip::extension_dynamique(imgray, 84, 144);

    //ip::show_histogram("hist1",imgray);

    ip::computeHistogram(imgray);

    // attente d'une touche de clavier
    cv::waitKey(0);

    return 0;
}
