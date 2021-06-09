
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;
int main(int argc, char** argv)
{
//    std::string image_path = "/home/gokhool/Cours/Traitement_Image/TP_I2/datasets/DIP3E_Original_Images_CH03/fractured_spine.tif";

    std::string image_path = argv[1];

    Mat img = imread(image_path, IMREAD_COLOR);
    if(img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    // imprimer la structure matricielle de l'image
//    cout<<img<<endl;

    int imwidth = img.cols;
    int imheight = img.rows;

    // Affichage de la taille de l'image
    cout<<"taille de l'image : "<<imwidth<< " X " <<imheight<<endl;

    //Affichage en couleur dans le terminale
    cout << "\033[1;31m taille de l'image : " <<imwidth<<  " X " <<imheight<<"\033[0m"<<endl ;

    imshow("Display window", img);
    int k = waitKey(0); // Wait for a keystroke in the window
    if(k == 's')
    {
        imwrite("fractured_spine.png", img); // image stocké dans le répertoire bin si on ne specifie pas le chemin exacte
    }
    return 0;
}
