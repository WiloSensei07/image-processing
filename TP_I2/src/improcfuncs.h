
#ifndef __IMAGEPROCFUNCS_HPP__
#define __IMAGEPROCFUNCS_HPP__

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "iostream"
#include <algorithm>

namespace ip{
//Implémentation des fonctions de la bibliothèque statique

    // traitement pixellique
    void point_based_accessing(const cv::Mat &image);
    void log_transform(const cv::Mat &image);
    void exp_transform(const cv::Mat &image);
    void gamma(const cv::Mat &image, double alpha);// Ajouté
    void extension_dynamique(const cv::Mat &image, double gmin, double gmax);// Ajouté
    void show_histogram(std::string wname, const cv::Mat &image);
    void computeHistogram(cv::Mat &image);
    void contrastStretching(const cv::Mat &image); // Ajouté

    //filtrage 2d
    void add_Salt_Pepper_Noise(cv::Mat &srcArr, float psel, float ppoivre );
    void add_Gaussian_Noise(cv::Mat &srcArr,double mean,double sigma);
    void gaussianKernel(cv::Mat &gaussK, const double &lambda_g);
    void convolution(cv::Mat &imgin);
    void convolve(cv::Mat &imgin, cv::Mat &imgout, const cv::Mat &mask);
    void medianFilter(cv::Mat &imgin, cv::Mat &imgout, int ksize);
    void maxFilter(cv::Mat &imgin, cv::Mat &imgout, int ksize); // Ajouté
    void minFilter(cv::Mat &imgin, cv::Mat &imgout, int ksize); // Ajouté
    void boxFilter(cv::Mat &imgin, cv::Mat &imgout, int ksize); // Ajouté

    void robertsFilter(cv::Mat &imgin);

    //Segmentation
    void convolve_2(cv::Mat &imgin, cv::Mat &imgout, const cv::Mat &mask, const cv::Mat &mask2);// Ajouté
    void maxGradiant(cv::Mat &imgin);// Ajouté
    void Roberts_cross_filter(cv::Mat &imgin); //Ajouté
    void Central_difference_filter(cv::Mat &imgin); //Ajouté
    void Prewitt_filter(cv::Mat &imgin); //Ajouté
    void Sobel_filter(cv::Mat &imgin); //Ajouté
    void Enhancement_filter(cv::Mat &imgin); //Ajouté
    void Laplacian_filter(cv::Mat &imgin); //Ajouté
    void thresholdingOTSU(cv::Mat &imgin);//Ajouté

    //Morphology
    void erosion(cv::Mat &imgin, cv::Mat &imgout, int ksize);// Ajouté
    void dilation(cv::Mat &imgin, cv::Mat &imgout, int ksize);// Ajouté
    void opening(cv::Mat &imgin, cv::Mat &imgout, int ksize);// Ajouté
    void closing(cv::Mat &imgin, cv::Mat &imgout, int ksize);// Ajouté

    //colour
    void convert2gray(cv::Mat &imgin, cv::Mat &imgout); // Ajouté
    void colourslice(cv::Mat &imgin, cv::Mat &imgblue ,cv::Mat &imggreen, cv::Mat &imgred); // Ajouté
    void convert2HSI (cv::Mat &imgin, cv::Mat &imghsi); // Ajouté
}

#endif
