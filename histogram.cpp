#include <iostream>
#include <vector>
#include <cstring>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <omp.h>

// --------------------------------------------------------------------------------
// 1. Utility Functions for Histogram Arrays
// --------------------------------------------------------------------------------

inline void initHist3(int hist_r[256], int hist_g[256], int hist_b[256])
{
    std::memset(hist_r, 0, 256 * sizeof(int));
    std::memset(hist_g, 0, 256 * sizeof(int));
    std::memset(hist_b, 0, 256 * sizeof(int));
}

// Sums all bins in one channel's histogram.
inline int sumHistogram(const int hist[256])
{
    int s = 0;
    for (int i = 0; i < 256; i++)
        s += hist[i];
    return s;
}

// Compares two histograms (R, G, B). Returns true if they match exactly.
inline bool compareHist3(const int ref_r[256], const int ref_g[256], const int ref_b[256],
                         const int test_r[256], const int test_g[256], const int test_b[256])
{
    for (int i = 0; i < 256; i++)
    {
        if (ref_r[i] != test_r[i] || ref_g[i] != test_g[i] || ref_b[i] != test_b[i])
            return false;
    }
    return true;
}

// Computes the total absolute difference between two histograms (R, G, B).
// e.g., how many counts are off across all bins.
inline void computeTotalAbsDiff(const int ref_r[256], const int ref_g[256], const int ref_b[256],
                                const int test_r[256], const int test_g[256], const int test_b[256],
                                int &diffR, int &diffG, int &diffB)
{
    diffR = diffG = diffB = 0;
    for (int i = 0; i < 256; i++)
    {
        diffR += std::abs(ref_r[i] - test_r[i]);
        diffG += std::abs(ref_g[i] - test_g[i]);
        diffB += std::abs(ref_b[i] - test_b[i]);
    }
}

// --------------------------------------------------------------------------------
// 2. Histogram Display Functions
// --------------------------------------------------------------------------------

// A. Single-channel histogram (256 bins).
void displayHistogram(const int hist[256], const std::string &winName, cv::Scalar color)
{
    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double)hist_w / 256);

    cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

    // Find the maximum count
    int maxVal = 0;
    for (int i = 0; i < 256; i++)
        maxVal = std::max(maxVal, hist[i]);

    double scale = (maxVal > 0) ? ((double)hist_h / maxVal) : 1.0;

    // Draw the histogram as vertical lines
    for (int i = 0; i < 256; i++)
    {
        int height = cvRound(hist[i] * scale);
        cv::line(histImage,
                 cv::Point(bin_w * i, hist_h),
                 cv::Point(bin_w * i, hist_h - height),
                 color, 2);
    }
    cv::imshow(winName, histImage);
}

// B. Combined R/G/B histogram on one chart.
void displayCombinedHistogram(const int hist_r[256],
                              const int hist_g[256],
                              const int hist_b[256],
                              const std::string &winName)
{
    int hist_w = 768; // 256 bins * 3 pixels per bin
    int hist_h = 400;
    cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

    // Find the maximum among all channels
    int maxVal = 0;
    for (int i = 0; i < 256; i++)
    {
        maxVal = std::max({maxVal, hist_r[i], hist_g[i], hist_b[i]});
    }
    double scale = (maxVal > 0) ? ((double)hist_h / maxVal) : 1.0;

    // Draw each bin: red at x, green at x+1, blue at x+2
    for (int i = 0; i < 256; i++)
    {
        int baseX = i * 3;
        int redHeight = cvRound(hist_r[i] * scale);
        int greenHeight = cvRound(hist_g[i] * scale);
        int blueHeight = cvRound(hist_b[i] * scale);

        // Red
        cv::line(histImage, cv::Point(baseX, hist_h),
                 cv::Point(baseX, hist_h - redHeight),
                 cv::Scalar(0, 0, 255), 1);
        // Green
        cv::line(histImage, cv::Point(baseX + 1, hist_h),
                 cv::Point(baseX + 1, hist_h - greenHeight),
                 cv::Scalar(0, 255, 0), 1);
        // Blue
        cv::line(histImage, cv::Point(baseX + 2, hist_h),
                 cv::Point(baseX + 2, hist_h - blueHeight),
                 cv::Scalar(255, 0, 0), 1);
    }

    cv::imshow(winName, histImage);
}

// C. Hue histogram (0..179), colored by hue.
void displayHueHistogram(const cv::Mat &hueHist, const std::string &windowName)
{
    int histSize = hueHist.rows; // typically 180
    int hist_w = 512;
    int hist_h = 400;
    cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

    double maxVal;
    cv::minMaxLoc(hueHist, 0, &maxVal);

    int bin_w = cvRound((double)hist_w / histSize);

    for (int i = 0; i < histSize; i++)
    {
        float binVal = hueHist.at<float>(i);
        int intensity = cvRound(binVal * hist_h / maxVal);

        // Create a 1x1 HSV image for hue = i
        cv::Mat hsvColor(1, 1, CV_8UC3, cv::Scalar(i, 255, 255));
        cv::Mat bgrColor;
        cv::cvtColor(hsvColor, bgrColor, cv::COLOR_HSV2BGR);
        cv::Vec3b colorVal = bgrColor.at<cv::Vec3b>(0, 0);

        cv::rectangle(histImage,
                      cv::Point(i * bin_w, hist_h),
                      cv::Point((i + 1) * bin_w, hist_h - intensity),
                      cv::Scalar(colorVal[0], colorVal[1], colorVal[2]),
                      cv::FILLED);
    }
    cv::imshow(windowName, histImage);
}

// --------------------------------------------------------------------------------
// 3. Histogram Computation Functions
// --------------------------------------------------------------------------------

// A. Sequential
void computeHistogramSequential(const cv::Mat &img, int hist_r[256], int hist_g[256], int hist_b[256])
{
    initHist3(hist_r, hist_g, hist_b);
    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            cv::Vec3b intensity = img.at<cv::Vec3b>(i, j);
            // OpenCV stores images in BGR order.
            hist_b[intensity[0]]++;
            hist_g[intensity[1]]++;
            hist_r[intensity[2]]++;
        }
    }
}

// B. Parallel No Sync
void computeHistogramParallelNoSync(const cv::Mat &img, int hist_r[256], int hist_g[256], int hist_b[256])
{
    initHist3(hist_r, hist_g, hist_b);

#pragma omp parallel for collapse(2)
    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            cv::Vec3b intensity = img.at<cv::Vec3b>(i, j);
            // Race condition if multiple threads increment the same bin simultaneously.
            hist_b[intensity[0]]++;
            hist_g[intensity[1]]++;
            hist_r[intensity[2]]++;
        }
    }
}

// C. Parallel with Critical
void computeHistogramParallelCritical(const cv::Mat &img, int hist_r[256], int hist_g[256], int hist_b[256])
{
    initHist3(hist_r, hist_g, hist_b);

#pragma omp parallel for collapse(2)
    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            cv::Vec3b intensity = img.at<cv::Vec3b>(i, j);
#pragma omp critical
            {
                hist_b[intensity[0]]++;
                hist_g[intensity[1]]++;
                hist_r[intensity[2]]++;
            }
        }
    }
}

// D. Parallel with Local Histograms
void computeHistogramParallelLocal(const cv::Mat &img, int hist_r[256], int hist_g[256], int hist_b[256])
{
    initHist3(hist_r, hist_g, hist_b);

    int num_threads = omp_get_max_threads();
    std::vector<std::vector<int>> local_r(num_threads, std::vector<int>(256, 0));
    std::vector<std::vector<int>> local_g(num_threads, std::vector<int>(256, 0));
    std::vector<std::vector<int>> local_b(num_threads, std::vector<int>(256, 0));

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
#pragma omp for collapse(2)
        for (int i = 0; i < img.rows; i++)
        {
            for (int j = 0; j < img.cols; j++)
            {
                cv::Vec3b intensity = img.at<cv::Vec3b>(i, j);
                local_b[tid][intensity[0]]++;
                local_g[tid][intensity[1]]++;
                local_r[tid][intensity[2]]++;
            }
        }
    }

    // Merge step
    for (int t = 0; t < num_threads; t++)
    {
        for (int i = 0; i < 256; i++)
        {
            hist_r[i] += local_r[t][i];
            hist_g[i] += local_g[t][i];
            hist_b[i] += local_b[t][i];
        }
    }
}

// --------------------------------------------------------------------------------
// 4. Main
// --------------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " <image_file>" << std::endl;
        return -1;
    }

    // Load the image
    cv::Mat img = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (img.empty())
    {
        std::cerr << "Error: Could not load image (" << argv[1] << ")" << std::endl;
        return -1;
    }
    std::cout << "Image loaded: " << argv[1] << " (" << img.cols << "x" << img.rows << ")\n"
              << std::endl;

    // Histograms for each version
    int hist_r_seq[256], hist_g_seq[256], hist_b_seq[256];
    int hist_r_unsync[256], hist_g_unsync[256], hist_b_unsync[256];
    int hist_r_crit[256], hist_g_crit[256], hist_b_crit[256];
    int hist_r_local[256], hist_g_local[256], hist_b_local[256];

    // 1) Sequential
    auto start = std::chrono::high_resolution_clock::now();
    computeHistogramSequential(img, hist_r_seq, hist_g_seq, hist_b_seq);
    auto end = std::chrono::high_resolution_clock::now();
    double time_seq = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Sequential execution time: " << time_seq << " ms" << std::endl;

    // 2) Parallel No Sync
    start = std::chrono::high_resolution_clock::now();
    computeHistogramParallelNoSync(img, hist_r_unsync, hist_g_unsync, hist_b_unsync);
    end = std::chrono::high_resolution_clock::now();
    double time_unsync = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Parallel (no sync) execution time: " << time_unsync << " ms" << std::endl;

    // 3) Parallel Critical
    start = std::chrono::high_resolution_clock::now();
    computeHistogramParallelCritical(img, hist_r_crit, hist_g_crit, hist_b_crit);
    end = std::chrono::high_resolution_clock::now();
    double time_crit = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Parallel (critical) execution time: " << time_crit << " ms" << std::endl;

    // 4) Parallel Local
    start = std::chrono::high_resolution_clock::now();
    computeHistogramParallelLocal(img, hist_r_local, hist_g_local, hist_b_local);
    end = std::chrono::high_resolution_clock::now();
    double time_local = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Parallel (local histograms) execution time: " << time_local << " ms\n"
              << std::endl;

    // Compare results to sequential
    bool matchCritical = compareHist3(hist_r_seq, hist_g_seq, hist_b_seq, hist_r_crit, hist_g_crit, hist_b_crit);
    bool matchLocal = compareHist3(hist_r_seq, hist_g_seq, hist_b_seq, hist_r_local, hist_g_local, hist_b_local);
    bool matchUnsync = compareHist3(hist_r_seq, hist_g_seq, hist_b_seq, hist_r_unsync, hist_g_unsync, hist_b_unsync);

    std::cout << "Critical synchronization result "
              << (matchCritical ? "matches" : "does not match")
              << " the sequential version." << std::endl;
    std::cout << "Local histograms merging result "
              << (matchLocal ? "matches" : "does not match")
              << " the sequential version." << std::endl;
    std::cout << "Unsynchronized version result "
              << (matchUnsync ? "matches" : "does not match")
              << " the sequential version.\n"
              << std::endl;

    // Display single-channel histograms (Sequential version)
    // displayHistogram(hist_r_seq, "Red Histogram", cv::Scalar(0, 0, 255));
    // displayHistogram(hist_g_seq, "Green Histogram", cv::Scalar(0, 255, 0));
    // displayHistogram(hist_b_seq, "Blue Histogram", cv::Scalar(255, 0, 0));

    // Compute and display Hue histogram
    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
    int histSize = 180;
    float hueRange[] = {0, 180};
    const float *ranges[] = {hueRange};
    int channels[] = {0}; // Hue channel
    cv::Mat hueHist;
    cv::calcHist(&hsv, 1, channels, cv::Mat(), hueHist, 1, &histSize, ranges, true, false);
    displayHueHistogram(hueHist, "Hue Histogram");

    // Display combined histograms for each approach
    displayCombinedHistogram(hist_r_seq, hist_g_seq, hist_b_seq, "Sequential Histogram");
    displayCombinedHistogram(hist_r_crit, hist_g_crit, hist_b_crit, "Parallel Critical Histogram");
    displayCombinedHistogram(hist_r_local, hist_g_local, hist_b_local, "Parallel Local Histogram");
    displayCombinedHistogram(hist_r_unsync, hist_g_unsync, hist_b_unsync, "Parallel Unsync Histogram");

    // Print total counts for each version
    int seqR = sumHistogram(hist_r_seq), seqG = sumHistogram(hist_g_seq), seqB = sumHistogram(hist_b_seq);
    int unsyncR = sumHistogram(hist_r_unsync), unsyncG = sumHistogram(hist_g_unsync), unsyncB = sumHistogram(hist_b_unsync);
    int critR = sumHistogram(hist_r_crit), critG = sumHistogram(hist_g_crit), critB = sumHistogram(hist_b_crit);
    int locR = sumHistogram(hist_r_local), locG = sumHistogram(hist_g_local), locB = sumHistogram(hist_b_local);

    std::cout << "Total counts (Sequential) - R: " << seqR
              << " G: " << seqG << " B: " << seqB << std::endl;
    std::cout << "Total counts (Unsynchronized) - R: " << unsyncR
              << " G: " << unsyncG << " B: " << unsyncB << std::endl;
    std::cout << "Total counts (Critical) - R: " << critR
              << " G: " << critG << " B: " << critB << std::endl;
    std::cout << "Total counts (Local) - R: " << locR
              << " G: " << locG << " B: " << locB << std::endl;

    // Compare unsynchronized with sequential
    int diffR = 0, diffG = 0, diffB = 0;
    computeTotalAbsDiff(hist_r_seq, hist_g_seq, hist_b_seq,
                        hist_r_unsync, hist_g_unsync, hist_b_unsync,
                        diffR, diffG, diffB);
    std::cout << "\nTotal absolute differences (Sequential vs Unsynchronized):\n";
    std::cout << "Red: " << diffR << "  Green: " << diffG << "  Blue: " << diffB << std::endl;

    // Keep windows open until ESC
    while (true)
    {
        int key = cv::waitKey(10);
        if (key == 27) // ESC
            break;
    }
    return 0;
}
