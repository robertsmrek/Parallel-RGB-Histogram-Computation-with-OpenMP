#include <iostream>
#include <vector>
#include <cstring>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <omp.h>

using namespace std;
using namespace cv;

// Function to display a single-channel histogram as a bar graph (for 256 bins).
void displayHistogram(const int hist[256], const string &winName, Scalar color)
{
    // Dimensions for the histogram image
    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double)hist_w / 256);

    // Create an image to display the histogram
    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

    // Find the maximum count to scale the histogram appropriately
    int maxVal = 0;
    for (int i = 0; i < 256; i++)
    {
        if (hist[i] > maxVal)
            maxVal = hist[i];
    }
    double scale = (maxVal > 0) ? ((double)hist_h / maxVal) : 1.0;

    // Draw the histogram as vertical lines
    for (int i = 0; i < 256; i++)
    {
        line(histImage,
             Point(bin_w * i, hist_h),
             Point(bin_w * i, hist_h - cvRound(hist[i] * scale)),
             color, 2, 8, 0);
    }
    imshow(winName, histImage);
}

// Sequential histogram computation for an RGB image.
void computeHistogramSequential(const Mat &img, int hist_r[256], int hist_g[256], int hist_b[256])
{
    memset(hist_r, 0, 256 * sizeof(int));
    memset(hist_g, 0, 256 * sizeof(int));
    memset(hist_b, 0, 256 * sizeof(int));

    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            Vec3b intensity = img.at<Vec3b>(i, j);
            // OpenCV stores images in BGR order.
            hist_b[intensity[0]]++;
            hist_g[intensity[1]]++;
            hist_r[intensity[2]]++;
        }
    }
}

// Parallel histogram computation WITHOUT synchronization (may produce incorrect results).
void computeHistogramParallelNoSync(const Mat &img, int hist_r[256], int hist_g[256], int hist_b[256])
{
    memset(hist_r, 0, 256 * sizeof(int));
    memset(hist_g, 0, 256 * sizeof(int));
    memset(hist_b, 0, 256 * sizeof(int));

#pragma omp parallel for collapse(2)
    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            Vec3b intensity = img.at<Vec3b>(i, j);
            // Race condition: concurrent updates without synchronization.
            hist_b[intensity[0]]++;
            hist_g[intensity[1]]++;
            hist_r[intensity[2]]++;
        }
    }
}

// Parallel histogram computation with critical sections for synchronization.
void computeHistogramParallelCritical(const Mat &img, int hist_r[256], int hist_g[256], int hist_b[256])
{
    memset(hist_r, 0, 256 * sizeof(int));
    memset(hist_g, 0, 256 * sizeof(int));
    memset(hist_b, 0, 256 * sizeof(int));

#pragma omp parallel for collapse(2)
    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            Vec3b intensity = img.at<Vec3b>(i, j);
#pragma omp critical
            {
                hist_b[intensity[0]]++;
                hist_g[intensity[1]]++;
                hist_r[intensity[2]]++;
            }
        }
    }
}

// Parallel histogram computation using local histograms per thread and merging.
void computeHistogramParallelLocal(const Mat &img, int hist_r[256], int hist_g[256], int hist_b[256])
{
    memset(hist_r, 0, 256 * sizeof(int));
    memset(hist_g, 0, 256 * sizeof(int));
    memset(hist_b, 0, 256 * sizeof(int));

    int num_threads = omp_get_max_threads();
    vector<vector<int>> local_hist_r(num_threads, vector<int>(256, 0));
    vector<vector<int>> local_hist_g(num_threads, vector<int>(256, 0));
    vector<vector<int>> local_hist_b(num_threads, vector<int>(256, 0));

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
#pragma omp for collapse(2)
        for (int i = 0; i < img.rows; i++)
        {
            for (int j = 0; j < img.cols; j++)
            {
                Vec3b intensity = img.at<Vec3b>(i, j);
                local_hist_b[tid][intensity[0]]++;
                local_hist_g[tid][intensity[1]]++;
                local_hist_r[tid][intensity[2]]++;
            }
        }
    }

    // Merge local histograms into the global arrays.
    for (int t = 0; t < num_threads; t++)
    {
        for (int i = 0; i < 256; i++)
        {
            hist_r[i] += local_hist_r[t][i];
            hist_g[i] += local_hist_g[t][i];
            hist_b[i] += local_hist_b[t][i];
        }
    }
}

// Displays a hue histogram where each bin is colored according to its hue.
// hueHist: 1D histogram of hue values (0..179) in OpenCV format (CV_32F).
// windowName: name of the window to display.
void displayHueHistogram(const cv::Mat &hueHist, const std::string &windowName)
{
    int histSize = hueHist.rows; // should be 180 for hue
    int hist_w = 512;
    int hist_h = 400;
    cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

    // Find max histogram value for normalization
    double maxVal;
    cv::minMaxLoc(hueHist, 0, &maxVal);

    // The width of each bin
    int bin_w = cvRound((double)hist_w / histSize);

    for (int i = 0; i < histSize; i++)
    {
        float binVal = hueHist.at<float>(i);
        int intensity = cvRound(binVal * hist_h / maxVal);

        // ---------------------------------------------------------
        // Generate the actual color for this hue:
        // - Create a 1x1 image in HSV with hue=i, saturation=255, value=255
        // - Convert it to BGR to get the final color.
        // ---------------------------------------------------------
        cv::Mat hsvColor(1, 1, CV_8UC3, cv::Scalar(i, 255, 255)); // H= i, S=255, V=255
        cv::Mat bgrColor;
        cv::cvtColor(hsvColor, bgrColor, cv::COLOR_HSV2BGR);
        cv::Vec3b colorVal = bgrColor.at<cv::Vec3b>(0, 0);

        // Draw a rectangle (bar) for this bin using the hue-based color.
        cv::rectangle(histImage,
                      cv::Point(i * bin_w, hist_h),
                      cv::Point((i + 1) * bin_w, hist_h - intensity),
                      cv::Scalar(colorVal[0], colorVal[1], colorVal[2]),
                      cv::FILLED);
    }

    cv::imshow(windowName, histImage);
}

// Function to display a combined RGB histogram for one version.
void displayCombinedHistogram(const int hist_r[256],
                              const int hist_g[256],
                              const int hist_b[256],
                              const std::string &winName)
{
    // Dimensions for the histogram image.
    int hist_w = 768; // 256 bins * 3 pixels per bin (for slight separation)
    int hist_h = 400;
    cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

    // Find the maximum value among all channels to normalize the drawing.
    int maxVal = 0;
    for (int i = 0; i < 256; i++)
    {
        maxVal = std::max(maxVal, hist_r[i]);
        maxVal = std::max(maxVal, hist_g[i]);
        maxVal = std::max(maxVal, hist_b[i]);
    }
    double scale = (maxVal > 0) ? ((double)hist_h / maxVal) : 1.0;

    // Draw each bin. We use a width of 3 pixels per intensity value:
    // pixel i in red is drawn at x = i*3,
    // green at x = i*3 + 1,
    // blue at x = i*3 + 2.
    for (int i = 0; i < 256; i++)
    {
        int redHeight = cvRound(hist_r[i] * scale);
        int greenHeight = cvRound(hist_g[i] * scale);
        int blueHeight = cvRound(hist_b[i] * scale);
        int baseX = i * 3;

        // Draw red channel (in red)
        cv::line(histImage, cv::Point(baseX, hist_h),
                 cv::Point(baseX, hist_h - redHeight),
                 cv::Scalar(0, 0, 255), 1);
        // Draw green channel (in green)
        cv::line(histImage, cv::Point(baseX + 1, hist_h),
                 cv::Point(baseX + 1, hist_h - greenHeight),
                 cv::Scalar(0, 255, 0), 1);
        // Draw blue channel (in blue)
        cv::line(histImage, cv::Point(baseX + 2, hist_h),
                 cv::Point(baseX + 2, hist_h - blueHeight),
                 cv::Scalar(255, 0, 0), 1);
    }

    cv::imshow(winName, histImage);
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        cout << "Usage: " << argv[0] << " <image_file>" << endl;
        return -1;
    }

    // Set locale for formatting with thousands separators
    std::locale loc("");
    std::cout.imbue(loc);

    // Load the image using the command-line argument.
    Mat img = imread(argv[1], IMREAD_COLOR);
    if (img.empty())
    {
        cout << "Error: Could not load image (" << argv[1] << ")" << endl;
        return -1;
    }
    cout << "Image loaded: " << argv[1] << " (" << img.cols << "x" << img.rows << ") \n"
         << endl;

    // ----------------------------------------------------------------------------
    int hist_r_seq[256], hist_g_seq[256], hist_b_seq[256];
    int hist_r_unsync[256], hist_g_unsync[256], hist_b_unsync[256];
    int hist_r_critical[256], hist_g_critical[256], hist_b_critical[256];
    int hist_r_local[256], hist_g_local[256], hist_b_local[256];

    // Sequential version.
    auto start = chrono::high_resolution_clock::now();
    computeHistogramSequential(img, hist_r_seq, hist_g_seq, hist_b_seq);
    auto end = chrono::high_resolution_clock::now();
    double time_seq = chrono::duration<double, milli>(end - start).count();
    cout << "Sequential execution time: " << time_seq << " ms" << endl;

    // Parallel version without synchronization.
    start = chrono::high_resolution_clock::now();
    computeHistogramParallelNoSync(img, hist_r_unsync, hist_g_unsync, hist_b_unsync);
    end = chrono::high_resolution_clock::now();
    double time_unsync = chrono::duration<double, milli>(end - start).count();
    cout << "Parallel (no sync) execution time: " << time_unsync << " ms" << endl;

    // Parallel version with critical sections.
    start = chrono::high_resolution_clock::now();
    computeHistogramParallelCritical(img, hist_r_critical, hist_g_critical, hist_b_critical);
    end = chrono::high_resolution_clock::now();
    double time_critical = chrono::duration<double, milli>(end - start).count();
    cout << "Parallel (critical) execution time: " << time_critical << " ms" << endl;

    // Parallel version with local histograms.
    start = chrono::high_resolution_clock::now();
    computeHistogramParallelLocal(img, hist_r_local, hist_g_local, hist_b_local);
    end = chrono::high_resolution_clock::now();
    double time_local = chrono::duration<double, milli>(end - start).count();
    cout << "Parallel (local histograms) execution time: " << time_local << " ms \n"
         << endl;

    // Validate that the synchronized versions match the sequential result.
    bool matchCritical = true;
    bool matchLocal = true;
    bool matchUnsync = true;
    for (int i = 0; i < 256; i++)
    {
        if (hist_r_seq[i] != hist_r_critical[i] ||
            hist_g_seq[i] != hist_g_critical[i] ||
            hist_b_seq[i] != hist_b_critical[i])
        {
            matchCritical = false;
        }
        if (hist_r_seq[i] != hist_r_local[i] ||
            hist_g_seq[i] != hist_g_local[i] ||
            hist_b_seq[i] != hist_b_local[i])
        {
            matchLocal = false;
        }
        if (hist_r_seq[i] != hist_r_unsync[i] ||
            hist_g_seq[i] != hist_g_unsync[i] ||
            hist_b_seq[i] != hist_b_unsync[i])
        {
            matchUnsync = false;
        }
    }
    cout << "Critical synchronization result " << (matchCritical ? "matches" : "does not match")
         << " the sequential version." << endl;
    cout << "Local histograms merging result " << (matchLocal ? "matches" : "does not match")
         << " the sequential version." << endl;
    cout << "Unsynchronized version result " << (matchUnsync ? "matches" : "does not match")
         << " the sequential version. \n"
         << endl;

    // Display the sequential histograms for R, G, B
    displayHistogram(hist_r_seq, "Red Histogram", Scalar(0, 0, 255));
    displayHistogram(hist_g_seq, "Green Histogram", Scalar(0, 255, 0));
    displayHistogram(hist_b_seq, "Blue Histogram", Scalar(255, 0, 0));

    // ----------------------------------------------------------------------------
    // Compute & Display a Single HUE Histogram
    // ----------------------------------------------------------------------------
    // Convert to HSV color space
    Mat hsv;
    cvtColor(img, hsv, COLOR_BGR2HSV);

    // Define parameters for calcHist (we want channel 0 = HUE, which is [0..179])
    int histSize = 180;          // number of bins for hue
    float hueRange[] = {0, 180}; // hue ranges from 0..179 in OpenCV
    const float *ranges[] = {hueRange};
    int channels[] = {0}; // channel 0 is Hue in HSV

    // Calculate the hue histogram
    Mat hueHist;
    calcHist(&hsv, 1, channels, Mat(), hueHist, 1, &histSize, ranges, true, false);

    // Display the hue histogram
    displayHueHistogram(hueHist, "Hue Histogram");

    // Display combined histogram for Sequential version
    displayCombinedHistogram(hist_r_seq, hist_g_seq, hist_b_seq, "Sequential Histogram");

    // Display combined histogram for Parallel Critical version
    displayCombinedHistogram(hist_r_critical, hist_g_critical, hist_b_critical, "Parallel Critical Histogram");

    // Display combined histogram for Parallel Local Histograms version
    displayCombinedHistogram(hist_r_local, hist_g_local, hist_b_local, "Parallel Local Histogram");

    // Display combined histogram for Unsynchronized version
    displayCombinedHistogram(hist_r_unsync, hist_g_unsync, hist_b_unsync, "Parallel Unsync Histogram");

    int sumSeqR = 0, sumSeqG = 0, sumSeqB = 0;
    int sumUnsyncR = 0, sumUnsyncG = 0, sumUnsyncB = 0;
    int sumCriticalR = 0, sumCriticalG = 0, sumCriticalB = 0;
    int sumLocalR = 0, sumLocalG = 0, sumLocalB = 0;
    for (int i = 0; i < 256; i++)
    {
        sumSeqR += hist_r_seq[i];
        sumSeqG += hist_g_seq[i];
        sumSeqB += hist_b_seq[i];

        sumUnsyncR += hist_r_unsync[i];
        sumUnsyncG += hist_g_unsync[i];
        sumUnsyncB += hist_b_unsync[i];

        sumCriticalR += hist_r_critical[i];
        sumCriticalG += hist_g_critical[i];
        sumCriticalB += hist_b_critical[i];

        sumLocalR += hist_r_local[i];
        sumLocalG += hist_g_local[i];
        sumLocalB += hist_b_local[i];
    }

    cout << "Total counts (Sequential) - R: " << sumSeqR
         << " G: " << sumSeqG
         << " B: " << sumSeqB << endl;
    cout << "Total counts (Unsynchronized) - R: " << sumUnsyncR
         << " G: " << sumUnsyncG
         << " B: " << sumUnsyncB << endl;
    cout << "Total counts (Critical) - R: " << sumCriticalR
         << " G: " << sumCriticalG
         << " B: " << sumCriticalB << endl;
    cout << "Total counts (Local) - R: " << sumLocalR
         << " G: " << sumLocalG
         << " B: " << sumLocalB << endl;

    // Compare unsynchronized results against the sequential (synchronized) histogram.
    int diffR = 0, diffG = 0, diffB = 0;
    for (int i = 0; i < 256; i++)
    {
        diffR += abs(hist_r_seq[i] - hist_r_unsync[i]);
        diffG += abs(hist_g_seq[i] - hist_g_unsync[i]);
        diffB += abs(hist_b_seq[i] - hist_b_unsync[i]);
    }
    cout << "\nTotal absolute differences (Sequential vs Unsynchronized):" << endl;
    cout << "Red: " << diffR << " Green: " << diffG << " Blue: " << diffB << endl;

    // Show windows until user presses escape
    while (true)
    {
        int key = waitKey(10);
        if (key == 27) // ESC key code
            break;
    }
    return 0;
}
