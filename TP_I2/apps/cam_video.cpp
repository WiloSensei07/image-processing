#include "opencv2/opencv.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "iostream"

int main(int argc, char** argv) {
    // open the first webcam plugged in the computer
    cv::VideoCapture camera(0);
    if (!camera.isOpened()) {
        std::cerr << "ERROR: Could not open camera" << std::endl;
        return 1;
    }

    // create a window to display the images from the webcam
    cv::namedWindow("Webcam" , 0);

    // this will contain the image from the webcam
    cv::Mat frame;       
    
    // display the frame until you press a key
    while (1) {

         // capture the next frame from the webcam
         camera >> frame;

        // show the image on the window
        cv::imshow("Webcam", frame);
        // wait (10ms) for a key to be pressed
        if (cv::waitKey(10) >= 0)
            break;
    }

    cv::destroyWindow("Webcam");
    return 0;
}
