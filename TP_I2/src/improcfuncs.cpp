
#include "improcfuncs.h"

using namespace  std;
using namespace cv;

void ip::point_based_accessing(const Mat &image)
{
    int value = 0;

    if(image.type() == CV_8UC1)
    {
        for(int i = 0 ; i< image.rows; i++)
            for(int j = 0 ; j< image.cols; j++)
            {
                value = (int)image.at<uchar>(i,j);

                cout<<value<<"\t";

            }
    }

    cv::Mat mask = (Mat_<double>(3,3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
    cout << "mask = " << endl << " " << mask << endl << endl;

    float data[9] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    cv::Mat your_matrix = cv::Mat(3, 3, CV_32F, data);

    cout << your_matrix.at<float>(1,2) << endl;
    cout << your_matrix << endl;


    cout<<endl;
}

void ip::log_transform(const cv::Mat &image)
{
    int value = 0;

    double scale = log(255.0);//facteur d'échelle

    // definition du tableau de correspondences
    std::vector<float> lut(256);
    lut[0] = 0;

    for (int i = 1; i<256; i++)
        lut[i] = (int) (255*log( i/scale));

    Mat imglog(image.rows, image.cols, CV_8UC1, Scalar(0));

        for(int i = 0 ; i< image.rows; i++)
            for(int j = 0 ; j< image.cols; j++)
            {
                 value = image.at<uchar>(i,j);
                 imglog.at <uchar>(i,j) = (int) lut[value];
            }


        cv::namedWindow("image log", 1);
        imshow("image log", imglog );
}


void ip::exp_transform(const cv::Mat &image)
{
    int value = 0;

    double scale = exp(2.55);//facteur d'échelle

    // definition du tableau de correspondences
    std::vector<float> lut(255);
    lut[0] = 0;

    for (int i = 1; i<=255; i++)
        lut[i] = (int) (255*exp( (i/100)/scale));

    Mat imgexp(image.rows, image.cols, CV_8UC1, Scalar(0));

        for(int i = 0 ; i< image.rows; i++)
            for(int j = 0 ; j< image.cols; j++)
            {
                 value = image.at<uchar>(i,j);
                 imgexp.at <uchar>(i,j) = (int) lut[value];
            }


        cv::namedWindow("image exp", 1);
        imshow("image exp", imgexp );
        show_histogram("hist1",imgexp);
}

void ip::gamma(const cv::Mat &image, double gamma)
{
    int value = 0;

    double scale = exp(2.55);//facteur d'échelle

    // definition du tableau de correspondences
    std::vector<float> lut(255);
    lut[0] = 0;

    for (int i = 1; i<=255; i++){
        lut[i] = (int) pow(i, gamma);

        if(lut[i]>255){
            lut[i]=255;
        }
    }

    Mat imggamma(image.rows, image.cols, CV_8UC1, Scalar(0));

        for(int i = 0 ; i< image.rows; i++)
            for(int j = 0 ; j< image.cols; j++)
            {
                 value = image.at<uchar>(i,j);
                 imggamma.at <uchar>(i,j) = (int) lut[value];
            }


        cv::namedWindow("image gamma", 1);
        imshow("image gamma", imggamma );
        show_histogram("histo gamma", imggamma);
}

void ip::extension_dynamique(const cv::Mat &image, double gmin, double gmax)
{
    int value = 0;

    double scale = exp(2.55); //facteur d'échelle

    Mat imgextension(image.rows, image.cols, CV_8UC1, Scalar(0));

        for(int i = 0 ; i< image.rows; i++)
            for(int j = 0 ; j< image.cols; j++)
            {
                 value = image.at<uchar>(i,j);
                 double temp = (int)(255*(value-gmin)/(gmax-gmin));
                 if(temp<0){
                     imgextension.at<uchar>(i,j)=0;
                 }
                 else if(temp>255){
                     imgextension.at<uchar>(i,j)=255;
                 }
                 else{
                     imgextension.at<uchar>(i,j)=temp;
                 }
            }


        cv::namedWindow("image contrast_stretching)", 1);
        imshow("image contrast_stretching)", imgextension );
        show_histogram("histo image pour contrast_stretching", image);
        show_histogram("histo contrast_stretching", imgextension);
}


void ip::show_histogram(std::string wname, const cv::Mat &image)
{
    // Set histogram bins count
    int bins = 256;
    int histSize[] = {bins};
    // Set ranges for histogram bins
    float lranges[] = {0, 256};
    const float* ranges[] = {lranges};
    // create matrix for histogram
    cv::Mat hist;
    int channels[] = {0};

    // create matrix for histogram visualization
    int const hist_height = 256;
    cv::Mat3b hist_image = cv::Mat3b::zeros(hist_height, bins);

    cv::calcHist(&image, 1, channels, cv::Mat(), hist, 1, histSize, ranges, true, false);

    double max_val=0;
    cv::minMaxLoc(hist, 0, &max_val);

    // visualize each bin
    for(int b = 0; b < bins; b++) {
        float const binVal = hist.at<float>(b);
        int   const height = cvRound(binVal*hist_height/max_val);
        cv::line
                ( hist_image   // matrice sur laquelle on va écrire
                  , cv::Point(b, hist_height-height), cv::Point(b, hist_height) // coord(x,y)
                  , cv::Scalar::all(255) // valeur de l'intensité
                  );
    }

     cv::namedWindow(wname, 1);
     imshow(wname, hist_image );


}


void ip::computeHistogram( cv::Mat &image)
{

    int imsize = image.rows * image.cols ;

    //initialisation du vecteur
    std::vector<float> hist (256, 0);

    // initialisation du pointeur à l'image
    uchar *ptr = image.ptr<uchar>(0);

    for(int i = 0; i< imsize; i++)
        ++hist[ptr[i]]; //on incremente l'index de l'histogram

    // normalise histogram
    std::vector<float> normhist (256, 0);
    for(size_t i = 0; i< hist.size(); i++){
            normhist[i] = hist[i]/imsize;

            cout<<i<<"\t"<<normhist[i]<<endl;

    }

    // compute cummulative distribution function
    std::vector<float> cfd (256, 0);
    cfd[0] = normhist[0];
    for(size_t i = 1; i< normhist.size(); i++){
        cfd[i] = normhist[i] + cfd[i-1];
    }

    // define lookup table (LUT)
    std::vector<float> lut(256,0);

    //Allocate LUT
    for (int i = 0; i<256; i++)
        lut[i] = (int) (255.0*cfd[i]);

    int value(0);
    Mat imgequalise(image.rows, image.cols, CV_8UC1, Scalar(0));

    for(int i = 0 ; i< image.rows; i++)
        for(int j = 0 ; j< image.cols; j++)
        {
            value = image.at<uchar>(i,j);
            imgequalise.at <uchar>(i,j) = (int) lut[value];
        }

    cv::namedWindow("image exp", 1);
    cv::imshow("image exp", imgequalise );

    show_histogram("hist2",imgequalise);

}

void ip::convolution(cv::Mat &imgin)
{
      cv::Mat mask = (cv::Mat_<float>(5,5) << 2, 4, 5,  4, 2,
                                           4, 9, 12, 9 ,4,
                                           5, 12, 15, 12, 5,
                                           4, 9, 12, 9 ,4,
                                           2, 4, 5,  4, 2);

      float normaliser = 159.0;   //256
      mask = mask/normaliser;

      cv::Mat mask2 = (cv::Mat_<float>(5,5) << 1, 4, 6,  4, 1,
                                            4, 16, 24, 16 ,4,
                                            6, 24, 36, 24, 6,
                                            4, 16, 24, 16 ,4,
                                            1, 4, 6,  4, 1);

      float normaliser2 = 256.0;
      mask2 = mask2/normaliser2;

      cout << "mask = " << endl << " " << mask<< endl << endl;
      cout << "mask2 = " << endl << " " << mask2<< endl << endl;

      cv::Mat imgout(imgin.rows, imgin.cols, CV_8UC1, Scalar(0));

      cv::Mat imgout2(imgin.rows, imgin.cols, CV_8UC1, Scalar(0));

      cv::Mat imgout3(imgin.rows, imgin.cols, CV_8UC1, Scalar(0));


      cv::Mat maskg(15, 15, CV_32F, Scalar(0));
      gaussianKernel(maskg,4.0);

      cout<<"Mask de 15/15 "<<maskg<<endl;

       // notre fonction à nous
       convolve(imgin, imgout, maskg);
       cv::namedWindow("gaussian blurr", 1);
       cv::imshow("gaussian blurr", imgout );

       //equivalent de la fonction en opencv
       cv::filter2D( imgin, imgout2, imgin.depth(), mask );
       cv::namedWindow("OCV filter2D", 1);
       cv::imshow("OCV filter2D", imgout2 );

       // Autre variant d'OPENCV
       cv::GaussianBlur(imgin,imgout3,Size(5,5),2.0,2.0,BORDER_DEFAULT);
       cv::namedWindow("OCV GaussianBlur", 1);
       cv::imshow("OCV GaussianBlur", imgout3 );
}

void ip::convolve(cv::Mat &imgin, cv::Mat &imgout, const cv::Mat &mask)
{

     CV_Assert(imgin.depth() == CV_8U);  // accept only uchar images

    int halfwx = floor(mask.rows/2);
    int halfwy = floor(mask.cols/2);
    float sum = 0;

// balayage de l'image en 2D
    for(int i = halfwx ; i< imgin.rows - halfwx ; i++)
        for(int j = halfwx ; j< imgin.cols - halfwx; j++)
        {
            // initialisation de la somme
            sum = 0;
            // glissage du masque/filtre dans l'image
            for (int wx = -halfwx; wx<=halfwx; wx++ )
                for (int wy = -halfwy; wy<=halfwy; wy++ )
                {
                    int u_wx = i + wx;
                    int v_wy = j + wy;

                     sum += imgin.at<uchar>(u_wx,v_wy) *
                            mask.at <float>(wx +halfwx,wy +halfwy);
          }

            // si somme plus que 255, valeur max  = 255
            imgout.at <uchar>(i,j) = (int) (sum > 255)? 255: sum;

        } // fin balayage de l'image

}

void ip::convolve_2(cv::Mat &imgin, cv::Mat &imgout, const cv::Mat &mask, const cv::Mat &mask2)
{

     CV_Assert(imgin.depth() == CV_8U);  // accept only uchar images

    int halfwx = floor(mask.rows/2);
    int halfwy = floor(mask.cols/2);
    float sum = 0;

// balayage de l'image en 2D
    for(int i = halfwx ; i< imgin.rows - halfwx ; i++)
        for(int j = halfwx ; j< imgin.cols - halfwx; j++)
        {
            // initialisation de la somme
            sum = 0;
            // glissage du masque/filtre dans l'image
            for (int wx = -halfwx; wx<=halfwx; wx++ )
                for (int wy = -halfwy; wy<=halfwy; wy++ )
                {
                    int u_wx = i + wx;
                    int v_wy = j + wy;

                    if(abs(mask.at <float>(wx +halfwx,wy +halfwy)) < abs(mask2.at <float>(wx +halfwx,wy +halfwy))){
                        sum += imgin.at<uchar>(u_wx,v_wy) *
                               mask2.at <float>(wx +halfwx,wy +halfwy);
                    }else {
                        sum += imgin.at<uchar>(u_wx,v_wy) *
                               mask.at <float>(wx +halfwx,wy +halfwy);
                    }
          }

            // si somme plus que 255, valeur max  = 255
            imgout.at <uchar>(i,j) = (int) (sum > 255)? 255: sum;

        } // fin balayage de l'image

}

void ip::maxGradiant(cv::Mat &imgin)
{
      cv::Mat mask = (cv::Mat_<float>(3,3) << 0,0,0,
                                             -1,1,0,
                                              0,0,0);

      cv::Mat mask2 = (cv::Mat_<float>(3,3) << 0,-1,0,
                                               0,1,0,
                                               0,0,0);

      cv::Mat mask3 = (cv::Mat_<float>(3,3) << 0,-1,0,
                                               -1,1,0,
                                               0,0,0);


//      cout << "mask = " << endl << " " << mask<< endl << endl;

      cv::Mat imgout(imgin.rows, imgin.cols, CV_8UC1, Scalar(0));


      cv::Mat maskg(15, 15, CV_32F, Scalar(0));
      gaussianKernel(maskg,4.0);

      cout<<maskg<<endl;

       // notre fonction à nous

      //convolve(imgin, imgout, mask1); //Pour afficher uniquement les traits verticaux
      //convolve(imgin, imgout, mask2); //Pour afficher uniquement les traits horizontaux
       convolve_2(imgin, imgout, mask, mask2);
       //ou encore
       //convolve(imgin, imgout, mask3);
       cv::namedWindow("Max gradiant", 1);
       cv::imshow("Max gradiant", imgout );
}

void ip::Roberts_cross_filter(cv::Mat &imgin)
{
      cv::Mat mask = (cv::Mat_<float>(3,3) << 0,0,0,
                                             0,-1,0,
                                              0,0,1);

      cv::Mat mask2 = (cv::Mat_<float>(3,3) << 0,0,0,
                                               0,0,-1,
                                               0,1,0);


//      cout << "mask = " << endl << " " << mask<< endl << endl;

      cv::Mat imgout(imgin.rows, imgin.cols, CV_8UC1, Scalar(0));


      cv::Mat maskg(15, 15, CV_32F, Scalar(0));
      gaussianKernel(maskg,4.0);

      cout<<maskg<<endl;

       // notre fonction à nous
       convolve_2(imgin, imgout, mask, mask2);
       cv::namedWindow("Roberts_cross_filter", 1);
       cv::imshow("Roberts_cross_filter", imgout );
}

void ip::Central_difference_filter(cv::Mat &imgin)
{
    cv::Mat mask = (cv::Mat_<float>(3,3) << 0,0,0,
                                           -1,0,1,
                                            0,0,0);

    cv::Mat mask2 = (cv::Mat_<float>(3,3) << 0,-1,0,
                                             0,0,0,
                                             0,1,0);


//      cout << "mask = " << endl << " " << mask<< endl << endl;

    cv::Mat imgout(imgin.rows, imgin.cols, CV_8UC1, Scalar(0));


    cv::Mat maskg(15, 15, CV_32F, Scalar(0));
    gaussianKernel(maskg,4.0);

    cout<<maskg<<endl;

     // notre fonction à nous
     convolve_2(imgin, imgout, mask, mask2);
     cv::namedWindow("Central_difference_filter", 1);
     cv::imshow("Central_difference_filter", imgout );

}

void ip::Prewitt_filter(cv::Mat &imgin)
{
    cv::Mat mask = (cv::Mat_<float>(3,3) << -1,0,1,
                                           -1,0,1,
                                            -1,0,1);

    cv::Mat mask2 = (cv::Mat_<float>(3,3) << -1,-1,-1,
                                             0,0,0,
                                             1,1,1);


//      cout << "mask = " << endl << " " << mask<< endl << endl;

    cv::Mat imgout(imgin.rows, imgin.cols, CV_8UC1, Scalar(0));


    cv::Mat maskg(15, 15, CV_32F, Scalar(0));
    gaussianKernel(maskg,4.0);

    cout<<maskg<<endl;

     // notre fonction à nous
     convolve_2(imgin, imgout, mask, mask2);
     cv::namedWindow("Prewitt_filter", 1);
     cv::imshow("Prewitt_filter", imgout );

}

void ip::Sobel_filter(cv::Mat &imgin)
{
    cv::Mat mask = (cv::Mat_<float>(3,3) << -1,0,1,
                                           -2,0,2,
                                            -1,0,1);

    cv::Mat mask2 = (cv::Mat_<float>(3,3) << -1,-2,-1,
                                             0,0,0,
                                             1,2,1);


//      cout << "mask = " << endl << " " << mask<< endl << endl;

    cv::Mat imgout(imgin.rows, imgin.cols, CV_8UC1, Scalar(0));


    cv::Mat maskg(15, 15, CV_32F, Scalar(0));
    gaussianKernel(maskg,4.0);

    cout<<maskg<<endl;

     // notre fonction à nous

     //convolve(imgin, imgout, mask1); //Pour afficher uniquement les traits verticaux
     //convolve(imgin, imgout, mask2); //Pour afficher uniquement les traits horizontaux
     convolve_2(imgin, imgout, mask, mask2);
     cv::namedWindow("Sobel_filter", 1);
     cv::imshow("Sobel_filter", imgout );

}

void ip::Enhancement_filter(cv::Mat &imgin)
{
    cv::Mat mask = (cv::Mat_<float>(3,3) << 0,-1,0,
                                           -1,5,-1,
                                            0,-1,0);

    cv::Mat mask2 = (cv::Mat_<float>(3,3) << -1,-1,-1,
                                             -1,9,-1,
                                             -1,-1,-1);


//      cout << "mask = " << endl << " " << mask<< endl << endl;

    cv::Mat imgout(imgin.rows, imgin.cols, CV_8UC1, Scalar(0));


    cv::Mat maskg(15, 15, CV_32F, Scalar(0));
    gaussianKernel(maskg,4.0);

    cout<<maskg<<endl;

     // notre fonction à nous
     convolve_2(imgin, imgout, mask, mask2);
     cv::namedWindow("Enhancement_filter", 1);
     cv::imshow("Enhancement_filter", imgout );

}

void ip::Laplacian_filter(cv::Mat &imgin)
{
    cv::Mat mask = (cv::Mat_<float>(3,3) << 0,-1,0,
                                           -1,4,-1,
                                            0,-1,0);

    cv::Mat mask2 = (cv::Mat_<float>(3,3) << -1,-1,-1,
                                             -1,8,-1,
                                             -1,-1,-1);


//      cout << "mask = " << endl << " " << mask<< endl << endl;

    cv::Mat imgout(imgin.rows, imgin.cols, CV_8UC1, Scalar(0));


    cv::Mat maskg(15, 15, CV_32F, Scalar(0));
    gaussianKernel(maskg,4.0);

    cout<<maskg<<endl;

     // notre fonction à nous
     convolve_2(imgin, imgout, mask, mask2);
     cv::namedWindow("Laplacian_filter", 1);
     cv::imshow("Laplacian_filter", imgout );

}

void ip::gaussianKernel(cv::Mat &gaussK, const double &lambda_g)
{

    int size = gaussK.rows;

    double x0 = floor(size / 2);
    double y0 = floor(size/ 2);

    double rx, ry, lg;

    lg = (1/(2*lambda_g *lambda_g));

    double norm = 0;

    for (int y = 0 ; y < size  ; y++)
        for(int x = 0 ; x < size ; x++)
        {
            // on centralise la gaussienne
            rx = (x-x0) * (x-x0) ;
            ry = (y-y0) * (y-y0) ;

            // application de la formule gaussienne
            gaussK.at<float>(x,y) = exp(-(rx  + ry)*lg);

            // la somme de tous les éléments de la matrice
            norm += exp(-(rx  + ry)*lg);
            gaussK.at<float>(x,y) = (1/(lambda_g * sqrt(2*M_PI)))*exp(-(rx  + ry)*lg);

        }

    // normalisation de la gaussienne centrée
    gaussK /= norm;
    cv::namedWindow("GaussianKernel Filter", 1);
    cv::imshow("GaussianKernel Filter", gaussK );
    return;

}


void ip::medianFilter(cv::Mat &imgin, cv::Mat &imgout, int ksize)
{

    CV_Assert(imgin.depth() == CV_8U);  // accept only uchar images

   int halfwx = floor(ksize/2);
   int halfwy = floor(ksize/2);


   for(int i = halfwx ; i< imgin.rows - halfwx ; i++)
       for(int j = halfwx ; j< imgin.cols - halfwx; j++)
       {
           // Déclaration de la liste vecmedian
           std::vector<int> vecmedian(ksize*ksize);
           int count =0;

           for (int wx = -halfwx; wx<=halfwx; wx++ )
               for (int wy = -halfwy; wy<=halfwy; wy++ )
               {
                   int u_wx = i + wx;
                   int v_wy = j + wy;
                    // instanciation de la liste
                    vecmedian[count++] = imgin.at<uchar>(u_wx,v_wy);
         }

           std::nth_element(vecmedian.begin(), vecmedian.begin() + vecmedian.size()/2, vecmedian.end());
           std::cout << "The median is " << vecmedian[vecmedian.size()/2] << '\n';

           imgout.at <uchar>(i,j) = vecmedian[vecmedian.size()/2];
       }

   cv::namedWindow("Median Filter", 1);
   cv::imshow("Median Filter", imgout );


}

void ip::maxFilter(cv::Mat &imgin, cv::Mat &maximgout, int ksize)
{

    CV_Assert(imgin.depth() == CV_8U);  // accept only uchar images

   int halfwx = floor(ksize/2);
   int halfwy = floor(ksize/2);


   for(int i = halfwx ; i< imgin.rows - halfwx ; i++)
       for(int j = halfwx ; j< imgin.cols - halfwx; j++)
       {
           // Déclaration de la liste vecmedian
           std::vector<int> vecmax(ksize*ksize);
           int count =0;

           for (int wx = -halfwx; wx<=halfwx; wx++ )
               for (int wy = -halfwy; wy<=halfwy; wy++ )
               {
                   int u_wx = i + wx;
                   int v_wy = j + wy;
                    // instanciation de la liste
                    vecmax[count++] = imgin.at<uchar>(u_wx,v_wy);
         }

           //std::nth_element(vecmedian.begin(), vecmedian.begin() + vecmedian.size()/2, vecmedian.end());
           std::sort(vecmax.begin(),vecmax.end());
           std::cout << "The last value is " << vecmax[vecmax.size()-1] << '\n';

           maximgout.at <uchar>(i,j) = vecmax[vecmax.size()-1];
       }

   cv::namedWindow("Max Filter", 1);
   cv::imshow("Max Filter", maximgout );


}

void ip::minFilter(cv::Mat &imgin, cv::Mat &minimgout, int ksize)
{

    CV_Assert(imgin.depth() == CV_8U);  // accept only uchar images

   int halfwx = floor(ksize/2);
   int halfwy = floor(ksize/2);


   for(int i = halfwx ; i< imgin.rows - halfwx ; i++)
       for(int j = halfwx ; j< imgin.cols - halfwx; j++)
       {
           // Déclaration de la liste vecmedian
           std::vector<int> vecmin(ksize*ksize);
           int count =0;

           for (int wx = -halfwx; wx<=halfwx; wx++ )
               for (int wy = -halfwy; wy<=halfwy; wy++ )
               {
                   int u_wx = i + wx;
                   int v_wy = j + wy;
                    // instanciation de la liste
                    vecmin[count++] = imgin.at<uchar>(u_wx,v_wy);
         }

           //std::nth_element(vecmedian.begin(), vecmedian.begin() + vecmedian.size()/2, vecmedian.end());
           std::sort(vecmin.begin(),vecmin.end());
           std::cout << "The first value is " << vecmin[vecmin.size()-vecmin.size()] << '\n';

           minimgout.at <uchar>(i,j) = vecmin[vecmin.size()-vecmin.size()];
       }

   cv::namedWindow("Min Filter", 1);
   cv::imshow("Min Filter", minimgout );


}

void ip::boxFilter(cv::Mat &imgin, cv::Mat &boximgout, int ksize)
{

    CV_Assert(imgin.depth() == CV_8U);  // accept only uchar images

    cv::Mat mask = (cv::Mat_<float>(3,3) << 1, 1, 1,
                                         1, 1, 1,
                                         1, 1, 1);
    mask=mask/9;

    //cout<<mask<<endl;

     // notre fonction à nous
     convolve(imgin, boximgout, mask);
    cv::namedWindow("Box Filter", 1);
    cv::imshow("Box Filter", boximgout );
}

void ip::add_Salt_Pepper_Noise(cv::Mat &srcArr, float pa, float pb )

{   cv::RNG rng; // rand number generate
    int amount1 = srcArr.rows * srcArr.cols * pa;
    int amount2 = srcArr.rows * srcArr.cols * pb;
    for(int counter=0; counter<amount1; ++counter)
    {
      // rng.uniform( 0,srcArr.rows), rng.uniform(0, srcArr.cols)
        srcArr.at<uchar>(rng.uniform( 0,srcArr.rows), rng.uniform(0, srcArr.cols)) =0;

    }
     for (int counter=0; counter<amount2; ++counter)
     {
        srcArr.at<uchar>(rng.uniform(0,srcArr.rows), rng.uniform(0,srcArr.cols)) = 255;
     }

     cv::namedWindow("Salt and Pepper", 1);
     cv::imshow("Salt and Pepper", srcArr );
}

void ip::add_Gaussian_Noise(Mat &srcArr,double mean,double sigma)
{
    cv::Mat NoiseArr = srcArr.clone();
    cv::RNG rng;
    rng.fill(NoiseArr, RNG::NORMAL, mean,sigma);

     //randn(dstArr,mean,sigma);
    cv::add(srcArr, NoiseArr, srcArr);

    cv::namedWindow("Gaussian noise", 1);
    cv::imshow("Gaussian noise", srcArr );
}

void ip::robertsFilter(cv::Mat &imgin)
{
      cv::Mat mask = (cv::Mat_<float>(3,3) << 0, 0, 0,
                                           0, 1, 0,
                                           0, 0, -1);

      cv::Mat imgout(imgin.rows, imgin.cols, CV_8UC1, Scalar(0));


      //cout<<mask<<endl;

       // notre fonction à nous
       convolve(imgin, imgout, mask);
       cv::namedWindow("roberts Filter", 1);
       cv::imshow("roberts Filter", imgout );
}

void ip::convert2gray(cv::Mat &imgin, cv::Mat &imgout )
{
    if (imgin.type()==CV_8UC3)
    {
        for (int i = 0; i<imgin.rows; i++)
            for (int j=0; j<imgin.cols; j++)
            {
                imgout.at<uchar>(i,j) = (imgin.at<cv::Vec3b>(i,j)[0] +
                imgin.at<cv::Vec3b>(i,j)[1] +
                imgin.at<cv::Vec3b>(i,j)[2]) / 3 ;
            }
    }
}

void ip::colourslice(cv::Mat &imgin, cv::Mat &imgblue ,cv::Mat &imggreen, cv::Mat &imgred)
{

     if (imgin.type()==CV_8UC3)
     {
         for ( int i=0 ; i< imgin.rows ; i++)
             for ( int j=0 ; j< imgin.cols ; j++)
             {
                 // Extracting blue channel
                 imgblue.at<cv::Vec3b>(i,j)[0] = imgin.at<cv::Vec3b>(i,j)[0] ;
                 imgblue.at<cv::Vec3b>(i,j)[1] = 0 ;
                 imgblue.at<cv::Vec3b>(i,j)[2] = 0 ;

                 // Extracting green channel
                 imggreen.at<cv::Vec3b>(i,j)[0] = 0 ;
                 imggreen.at<cv::Vec3b>(i,j)[1] = imgin.at<cv::Vec3b>(i,j)[1] ;
                 imggreen.at<cv::Vec3b>(i,j) [ 2 ] = 0 ;

                 // Extracting red channel
                 imgred.at<cv::Vec3b>(i,j)[0] = 0 ;
                 imgred.at<cv::Vec3b>(i,j)[1] = 0 ;
                 imgred.at<cv::Vec3b>(i,j)[2] = imgin.at<cv::Vec3b>(i,j)[2] ;

            }
     }
}

void ip::convert2HSI (cv::Mat &imgin, cv::Mat &imghsi)
 {
     float blue, green, red, saturation, hue, intensity, theta, frac1, frac2 ;
     float eps = 1*exp(-6); // epsilon

     //initialisation du pointeur l’image
     if (imgin.type() == CV_8UC3)
     {
         for ( int i = 0 ; i< imgin.rows ; i++)
         {
             for ( int j = 0 ; j< imgin.cols ; j++)
             {
              //extraction du bleu , vert et rouge et on normalize par 255
             blue = imgin.at<cv::Vec3b>(i,j)[0] / 255.0 ;
             green = imgin.at<cv::Vec3b>(i,j)[1] / 255.0 ;
             red = imgin.at<cv::Vec3b>(i,j)[2] / 255.0 ;

             frac1 = (float) ((red - green) + (red - blue)) * 0.5 ;
             frac2 = (float) sqrt(pow((red - green),2) + ((red - blue) * (green - blue)) + eps ) ;

             //The denominator cannot be ZERO! ca s e o f unde f ined 0/0
             if (frac2 == 0)
             {
                hue = 0 ;
             }
             else
             {
                 theta = acos(frac1 / frac2) ;
                 if ( blue <= green )
                 hue = theta/(M_PI*2); // on normalise par 2*PI
                 else
                 hue = (2*M_PI-theta)/ (2 *M_PI ) ; //on normalise par 2*PI
             }

              //calcule du min (R,V,B)
             float minRGB = min(min(red, green), blue ) ;

             // somme de (R,V,B)
             float den = red+green+blue ;

              //The denominator cannot be ZERO!
             if ( den == 0)
             saturation = 0 ;
             else
             saturation = 1 - 3*minRGB/den ; // calcule de la saturation

             // calcule de l’intensite
             intensity = den / 3.0 ;

             // concatenation des matrices dans la structure cv::Vec3b
             imghsi.at<cv::Vec3b>(i,j)[0] = hue*255 ;
             imghsi.at<cv::Vec3b>(i,j)[1] = saturation*255 ;
             imghsi.at<cv::Vec3b>(i,j)[2] = intensity*255 ;

             }
         }
     }
}

void ip::erosion(cv::Mat &imgin, cv::Mat &imgout, int ksize)
{

    CV_Assert(imgin.depth() == CV_8U);  // accept only uchar images

   int halfwx = floor(ksize/2);
   int halfwy = floor(ksize/2);


   for(int i = halfwx ; i< imgin.rows - halfwx ; i++)
       for(int j = halfwx ; j< imgin.cols - halfwx; j++)
       {
           // Déclaration de la liste vecmedian
           std::vector<int> vecmin(ksize*ksize);
           int count =0;

           for (int wx = -halfwx; wx<=halfwx; wx++ )
               for (int wy = -halfwy; wy<=halfwy; wy++ )
               {
                   int u_wx = i + wx;
                   int v_wy = j + wy;
                    // instanciation de la liste
                    vecmin[count++] = imgin.at<uchar>(u_wx,v_wy);
         }

           //std::nth_element(vecmedian.begin(), vecmedian.begin() + vecmedian.size()/2, vecmedian.end());
           std::sort(vecmin.begin(),vecmin.end());
           std::cout << "The first value is " << vecmin[vecmin.size()-vecmin.size()] << '\n';

           imgout.at <uchar>(i,j) = vecmin[vecmin.size()-vecmin.size()];
       }

   cv::namedWindow("Image erosion", 1);
   cv::imshow("Image erosion", imgout );


}

void ip::dilation(cv::Mat &imgin, cv::Mat &imgout, int ksize)
{

    CV_Assert(imgin.depth() == CV_8U);  // accept only uchar images

   int halfwx = floor(ksize/2);
   int halfwy = floor(ksize/2);


   for(int i = halfwx ; i< imgin.rows - halfwx ; i++)
       for(int j = halfwx ; j< imgin.cols - halfwx; j++)
       {
           // Déclaration de la liste vecmedian
           std::vector<int> vecmax(ksize*ksize);
           int count =0;

           for (int wx = -halfwx; wx<=halfwx; wx++ )
               for (int wy = -halfwy; wy<=halfwy; wy++ )
               {
                   int u_wx = i + wx;
                   int v_wy = j + wy;
                    // instanciation de la liste
                    vecmax[count++] = imgin.at<uchar>(u_wx,v_wy);
         }

           //std::nth_element(vecmedian.begin(), vecmedian.begin() + vecmedian.size()/2, vecmedian.end());
           std::sort(vecmax.begin(),vecmax.end());
           std::cout << "The last value is " << vecmax[vecmax.size()-1] << '\n';

           imgout.at <uchar>(i,j) = vecmax[vecmax.size()-1];
       }

   cv::namedWindow("Image dilation", 1);
   cv::imshow("Image dilation", imgout );


}

void ip::opening(cv::Mat &imgin, cv::Mat &imgout, int ksize)
{

    Mat imageTemp(imgin.rows, imgin.cols, CV_8UC1, Scalar(0));
    erosion(imgin, imageTemp,ksize);
    dilation(imageTemp,imgout,ksize);

   cv::namedWindow("Image opening", 1);
   cv::imshow("Image opening", imgout );


}

void ip::closing(cv::Mat &imgin, cv::Mat &imgout, int ksize)
{

    Mat imageTemp(imgin.rows, imgin.cols, CV_8UC1, Scalar(0));
    dilation(imgin, imageTemp,ksize);
    erosion(imageTemp,imgout,ksize);

   cv::namedWindow("Image closing", 1);
   cv::imshow("Image closing", imgout );


}

void ip::thresholdingOTSU(cv::Mat &imgin){
    int isize= imgin.rows * imgin.cols;
    float w0, w1, mu0, mu1;
    float var=0.0, max=0.0, kstar=0.0;

    //initialisation du vecteur
    std::vector<float> hist(256,0);
     std::vector<float> pr(256,0);

    // initialisation du pointeur de l’image
    uchar *ptr = imgin.ptr<uchar>(0);

    // calcul d'histgramme
    for (int i=0;i<isize;i++)
        ++hist[ptr[i]]; //on incrémente l'index de l'histogramme

    // normalisation de l'histogramme

    for (int i=0;i<256;i++){
        pr[i]=hist[i]/isize;
        cout << pr[i] <<endl;
    }

    for (int k=1; k<256;k++) {

        w0=0.0; w1=0.0;
       // calcul de w0 et w1

        for (int i=1; i<=k;i++)
            w0+=pr[i];

        w1=1-w0;

        // calcul mu0 et mu1

        mu0=0.0; mu1=0.0;

        for (int i=1;i<=k;i++)
            mu0+=(i*pr[i]/w0);

        for (int i=k+1;i<256;i++)
            mu1=mu1+(i*pr[i]/w1);

        var=w0*w1*pow((mu1-mu0),2);
       // cout << var <<endl;

        if(var>max){
            max=var;
            kstar=k;


        }

    }


    cout << kstar <<endl;
    cv::Mat imgout(imgin.rows, imgin.cols, CV_8UC1, Scalar(0));

    for (int i=0;i<imgin.rows;i++)
        for (int j=0;j<imgin.cols; j++) {

            int val= imgin.at<uchar>(i,j);


            if(val<kstar){
                imgout.at<uchar>(i,j)= 0;
            }
            else {
                imgout.at<uchar>(i,j)= 255;
            }
        }

    cv::namedWindow("image thresholdingOTSU",1);
    cv::imshow("image thresholdingOTSU", imgout);

}



