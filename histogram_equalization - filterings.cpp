// ****************************
// Name = Aras Umut
// Surname = Erarslan
// Student ID = 2005627
// Task = LAB 2
// ****************************

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include <iostream>

using namespace cv;
using namespace std;

// function that calculates the histogram of a given image and showing them. It also takes "name" field
// to use it as the name of the windows. The basic version can be reach from https://docs.opencv.org/3.4/d8/dbc/tutorial_histogram_calculation.html
// I added some lines to show b, g, r in different windows and turned it into a function to use in 3 different places
void CalculateHistogram(Mat& image, String name)
{
    // Vector Mat that hold b, g, r planes
    vector<Mat> bgr_planes;

    // split image into b, g, r planes
    split(image, bgr_planes);

    // number of bins of hist
    int histSize = 256;

    // range for b, g, r. Total 255
    float range[] = { 0, 256 }; //the upper boundary is exclusive
    const float* histRange = { range };

    // parameters to be used in calcHist function
    bool uniform = true, accumulate = false;

    // Mat for b, g, r planes
    Mat b_hist, g_hist, r_hist;

    // Calculation of histograms. Each of takes b g r hists above and calculates histograms.
    /* 
    -- Parameters --
    &bgr_planes[0]: The source array(s)
    1 : The number of source arrays(in this case we are using 1. We can enter here also a list of arrays)
    0 : The channel(dim) to be measured.In this case it is just the intensity(each array is single - channel) so we just write 0.
    Mat(): A mask to be used on the source array(zeros indicating pixels to be ignored).If not defined it is not used
    b_hist : The Mat object where the histogram will be stored
    1 : The histogram dimensionality.
    histSize : The number of bins per each used dimension
    histRange : The range of values to be measured per each dimension
    uniform and accumulate : The bin sizes are the same and the histogram is cleared at the beginning.
    */

    calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate); 

    // Features for histogram windows
    int hist_w = 300, hist_h = 200;
    int bin_w = cvRound((double)hist_w / histSize);

    // Creating Mat for the histogram of blue
    Mat blue(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
    // Normalize the histogram before drawing it so values will be in the range of the parameters that is entered
    normalize(b_hist, b_hist, 0, blue.rows, NORM_MINMAX, -1, Mat());

    // Creating Mat for the histogram of green
    Mat green(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
    // Normalize the histogram before drawing it so values will be in the range of the parameters that is entered
    normalize(g_hist, g_hist, 0, green.rows, NORM_MINMAX, -1, Mat());

    // Creating Mat for the histogram of red
    Mat red(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
    // Normalize the histogram before drawing it so values will be in the range of the parameters that is entered
    normalize(r_hist, r_hist, 0, red.rows, NORM_MINMAX, -1, Mat());


    // draws b g r
    /*
    --Parameters
    b_hist: Input array
    b_hist: Output normalized array (can be the same)
    0 and histImage.rows: For this example, they are the lower and upper limits to normalize the values of r_hist
    NORM_MINMAX: Argument that indicates the type of normalization (as described above, it adjusts the values between the two limits set before)
    -1: Implies that the output normalized array will be the same type as the input
    Mat(): Optional mask
    */
    for (int i = 1; i < histSize; i++)
    {
        line(blue, Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
            Scalar(255, 0, 0), 2, 8, 0);
        line(green, Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))),
            Scalar(0, 255, 0), 2, 8, 0);
        line(red, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
            Scalar(0, 0, 255), 2, 8, 0);
    }

    // Histogram window names
    String blueHistogramName = name + " Blue";
    String greenHistogramName = name + " Green";
    String redHistogramName = name + " Red";

    // Showing the histogram and image windows
    imshow(name, image);
    imshow(blueHistogramName, blue);
    imshow(greenHistogramName, green);
    imshow(redHistogramName, red);

    // Console texts
    cout << "\nThis is the = " << name << ".\n";
    cout << "Histogram Images has the name of the Image they belongs to prevent window confusions.\n";
    cout << "Press any key on the Image window to go to the next part...\n\n\n";
    waitKey();
}

//main
int main(int argc, char** argv)
{
    // Menu Index
    int menuIndex = 3;

    // do while to keep program running. 
    do {

        // do while to check menu index valute
        do {
            cout << "To which part do you want to go?\n( 1 ) Histogram Equalization\n( 2 ) Image Filtering\n( 0 ) Exit\nType 1, 2 or 0: ";
            cin >> menuIndex;
        } while (menuIndex != 1 && menuIndex != 2 && menuIndex != 0);

        // inside of option 1 in menu ( EQUALIZATION )
        if (menuIndex == 1)
        {

            // read image
            Mat image = imread("data/barbecue.png", IMREAD_COLOR);

            // Call CalculateHistogram function
            CalculateHistogram(image, "Source Image");

            // ****************************
            // EQUALIZATION PART
            // ****************************

            // Creating Mat for equalized image
            Mat hist_equalized_image;

            // Cloning source image
            hist_equalized_image = image.clone();

            // Split the image into 3 channels; B, G, R channels respectively and store it in a std::vector
            vector<Mat> vec_channels;
            split(hist_equalized_image, vec_channels);

            // Equalize the histogram of b, g, r channels
            equalizeHist(vec_channels[0], vec_channels[0]);
            equalizeHist(vec_channels[1], vec_channels[1]);
            equalizeHist(vec_channels[2], vec_channels[2]);

            // Merge 3 channels in the vector to form the color image in BGR color space.
            merge(vec_channels, hist_equalized_image);

            // Call CalculateHistogram function
            CalculateHistogram(hist_equalized_image, "BGR Equalized Image");

            // ****************************
            // EQUALIZATION PART OPTIONAL
            // ****************************

            // Creating Mat for equalized image optimal task
            Mat hist_equalized_image_optional;

            //Convert the image from BGR to LAB color space
            cvtColor(image, hist_equalized_image_optional, COLOR_BGR2Lab);

            // Split the image into 3 channels; L, a, b channels respectively and store it in a std::vector
            vector<Mat> vec_channels_optional;
            split(hist_equalized_image_optional, vec_channels_optional);

            // Equalize the histogram of only the L channel 
            equalizeHist(vec_channels_optional[0], vec_channels_optional[0]);

            // Merge 3 channels in the vector to form the color image in Lab color space.
            merge(vec_channels_optional, hist_equalized_image_optional);

            // Convert the histogram equalized image from Lab to BGR color space again
            cvtColor(hist_equalized_image_optional, hist_equalized_image_optional, COLOR_Lab2BGR);

            // Call CalculateHistogram function
            CalculateHistogram(hist_equalized_image_optional, " COLOR_BGR2Lab Only L Equalized Image");

            // Destroy all windows and go back to the start
            destroyAllWindows();
        }

        // inside of option 2 in menu ( EQUALIZATION )
        else if (menuIndex == 2)
        {

            // read image and clone it
            Mat imagePart2 = imread("data/barbecue.png", IMREAD_COLOR);
            Mat imagePart2Clone = imagePart2.clone();

            // ****************************
            // MEDIAN
            // ****************************

            // kernel size to be used in filters
            int kernel_size = 1;

            // create a window for median filters
            namedWindow("Median", 1);

            // text to be displayed in console
            cout << "\nPress ESC to go Gaussian filter...\n";

            // create a trackbar for kernel size on median window
            createTrackbar("Kernel\nSize", "Median", &kernel_size, 51);

            // shows original image
            imshow("Original Image", imagePart2);

            
            while (true)
            {
                // checks the kernel size and corrects it to be an odd value. If it is even, adds 1 to it
                if (kernel_size % 2 == 0)
                    kernel_size = kernel_size + 1; // kernel size needs to be odd, add it 1 if it is even

                // medianblur function
                medianBlur(imagePart2, imagePart2Clone, kernel_size);

                // shows the median blur applied image
                imshow("Median", imagePart2Clone);

                // wait for esc key to go to the next part
                int iKey = waitKey(50);
                if (iKey == 27)
                {
                    break;
                }
            }

            // destroys median window
            destroyWindow("Median");

            // ****************************
            // GAUSSIAN
            // ****************************

            // resets the value of kernel size
            kernel_size = 1;

            // a new window for gaussian filter
            namedWindow("Gaussian", 1);

            // text to be displayed in console
            cout << "\nPress ESC to go Bilateral filter...\n";

            // create track bar for kernel size parameter
            createTrackbar("Kernel\nSize", "Gaussian", &kernel_size, 101);

            // create a track bar for sigma parameter. trackbar does not take double value so I took int and divided it into 100
            // the sigma value is the /100 of the value in trackbar
            int sigma_trackbar = 100; // 100 = 1
            createTrackbar("Sigma\n/100", "Gaussian", &sigma_trackbar, 1000);

            while (true)
            {
                // checks the kernel size and corrects it to be an odd value. If it is even, adds 1 to it
                if (kernel_size % 2 == 0)
                    kernel_size = kernel_size + 1; // kernel size needs to be odd, add it 1 if it is even

                // GaussianBlur function with the values from trackbars
                GaussianBlur(imagePart2, imagePart2Clone, Size(kernel_size, kernel_size), sigma_trackbar / 100, sigma_trackbar / 100);

                // shows gaussian applied image
                imshow("Gaussian", imagePart2Clone);

                // wait for esc key
                int iKey = waitKey(50);
                if (iKey == 27)
                {
                    break;
                }
            }

            // destroy gaussian window
            destroyWindow("Gaussian");

            // ****************************
            // BILATERAL
            // ****************************


            // fixed kernel size
            kernel_size = 9;

            // new window for bilateral
            namedWindow("Bilateral", 1);

            // text to be displayed in console
            cout << "\nKernel Size = 9 FIXED for Bilateral\nPress ESC to RESTART the program...\n";

            // create trackbars for sigma_range and sigma_space. Like in the gaussian, trackbar values are divided into 100 in function
            int sigma_range_trackbar = 1;
            createTrackbar("Sigma\nRange\n/100", "Bilateral", &sigma_range_trackbar, 100000); // /100
            int sigma_space_trackbar = 1;
            createTrackbar("Sigma\nSpace\n/100", "Bilateral", &sigma_space_trackbar, 100000); // /100

            while (true)
            {
                // bilateral filter with the values from trackbar. sigma_range and sigma_space are divided into 100 before applied
                bilateralFilter(imagePart2, imagePart2Clone, kernel_size, sigma_range_trackbar / 100, sigma_space_trackbar / 100);

                // shows the bilateral filter applied image
                imshow("Bilateral", imagePart2Clone);

                // wait for esc key
                int iKey = waitKey(50);
                if (iKey == 27)
                {
                    break;
                }
            }

            //destroy all windows
            destroyAllWindows();
        }
    } while (menuIndex != 0);
    return EXIT_SUCCESS;
}