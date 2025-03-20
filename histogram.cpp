#include <iostream>
#include <vector>
#include <cstring>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <omp.h>

// --------------------------------------------------------------------------------
// Utility Functions for Histogram Arrays
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
// Histogram Display Functions
// --------------------------------------------------------------------------------

// Single-channel histogram (256 bins).
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

// Combined R/G/B histogram on one chart.
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

// Hue histogram (0..179), colored by hue.
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
// Histogram Computation Functions
// --------------------------------------------------------------------------------

/**
 * Sequential histogram computation
 *
 * This is the baseline implementation that processes each pixel sequentially.
 * - Simple to understand and debug
 * - Guaranteed correct results
 * - No synchronization overhead
 * - But uses only a single CPU core/thread
 * - For small images, this can be faster than parallel methods due to lack of thread overhead
 *
 * Implementation details:
 * 1. Initialize histogram arrays to zeros
 * 2. Iterate through each pixel in row-major order (good cache locality)
 * 3. Extract BGR color components from each pixel
 * 4. Increment the corresponding histogram bin for each color channel
 * 5. Entire processing happens in a single thread with no contention
 * 6. Memory access pattern is sequential but with random writes to histogram bins
 */
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

/**
 * Parallel histogram computation without synchronization
 *
 * This version processes pixels in parallel without any thread synchronization.
 * - Fastest parallel execution time
 * - However, produces incorrect results due to race conditions
 * - Multiple threads may update the same bin simultaneously, causing lost updates
 * - Demonstrates the need for proper synchronization in parallel code
 * - Useful as a reference for maximum theoretical parallel performance
 *
 * Implementation details:
 * 1. Initialize histogram arrays to zeros
 * 2. OpenMP divides the image into chunks processed by different threads
 * 3. The 'collapse(2)' directive creates a single iteration space from the nested loops
 * 4. Each thread processes its assigned pixels independently
 * 5. PROBLEM: When two threads try to increment the same histogram bin:
 *    a. Both read the current value (e.g., 5)
 *    b. Both increment their local copy to 6
 *    c. Both write back 6 to memory
 *    d. The final value is 6 instead of the correct 7
 * 6. The errors increase with more threads and larger images
 * 7. With 1 thread, this method produces correct results (no race condition possible)
 */
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

/**
 * Parallel histogram computation with critical sections
 *
 * Uses OpenMP critical sections to synchronize histogram updates.
 * - Guarantees correct results with proper synchronization
 * - Usually the slowest parallel method due to coarse-grained locking
 * - Only one thread can be in the critical section at a time
 * - Creates a severe bottleneck as threads queue up to enter the critical section
 * - Simple implementation but poor scalability
 *
 * Implementation details:
 * 1. Initialize histogram arrays to zeros
 * 2. OpenMP divides the pixels among threads (collapse(2) combines the nested loops)
 * 3. When a thread needs to update histogram bins:
 *    a. It requests exclusive access to the critical section
 *    b. Thread waits if another thread is currently in the critical section
 *    c. Once inside, thread updates all three histogram channels
 *    d. Thread exits critical section, allowing another thread to enter
 * 4. With many threads, most time is spent waiting to enter the critical section
 * 5. Performance gets worse as thread count increases due to contention
 * 6. The critical section is too coarse-grained (locks all histograms at once)
 */
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

/**
 * Parallel histogram computation with atomic operations
 *
 * Uses OpenMP atomic operations for fine-grained synchronization.
 * - Guarantees correct results
 * - Better performance than critical sections
 * - Fine-grained synchronization (locks only the exact memory being updated)
 * - Multiple threads can update different bins concurrently
 * - Still has synchronization overhead for every histogram update
 * - Good balance of simplicity and performance
 *
 * Implementation details:
 * 1. Initialize histogram arrays to zeros
 * 2. OpenMP divides the pixels among threads
 * 3. For each histogram update operation:
 *    a. The atomic directive ensures the read-modify-write is one indivisible operation
 *    b. Updates to the same bin are serialized (one thread at a time)
 *    c. Different threads can update different bins simultaneously
 * 4. Implementation typically uses hardware atomic instructions (e.g., LOCK prefix on x86)
 * 5. Each color channel's update is independent and atomic
 * 6. Trade-off: More parallel than critical sections but still has overhead for every update
 * 7. Performance scales better with thread count than critical sections
 */
void computeHistogramParallelAtomic(const cv::Mat &img, int hist_r[256], int hist_g[256], int hist_b[256])
{
    initHist3(hist_r, hist_g, hist_b);

#pragma omp parallel for collapse(2)
    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            const cv::Vec3b &intensity = img.at<cv::Vec3b>(i, j);

#pragma omp atomic
            hist_b[intensity[0]]++;

#pragma omp atomic
            hist_g[intensity[1]]++;

#pragma omp atomic
            hist_r[intensity[2]]++;
        }
    }
}

/**
 * Thread-local approach for parallel histogram computation
 *
 * Each thread maintains private histograms, combining results at the end.
 * - Usually the fastest correct parallel method
 * - Perfect scalability during computation phase
 * - Minimizes synchronization to just once per thread at the end
 * - Uses more memory (each thread has its own histograms)
 * - Ideal for large images with many pixels
 * - Shows the pattern: "parallelize + privatize + reduce" for best performance
 *
 * Implementation details:
 * 1. Initialize the final histogram arrays to zeros
 * 2. Create a parallel region with OpenMP
 * 3. Within that region, each thread:
 *    a. Allocates its own private histogram arrays (768 integers total)
 *    b. Processes its assigned chunk of the image without any synchronization
 *    c. Updates only its private histogram arrays (no contention with other threads)
 * 4. After all pixels are processed:
 *    a. Each thread enters a critical section once
 *    b. Adds its private histogram counts to the shared final histograms
 *    c. This reduction step combines all thread-local results
 * 5. Synchronization is minimized to just N critical sections (where N = thread count)
 * 6. Memory usage increases with thread count but is typically negligible
 * 7. Nearly linear scaling with thread count for the computation phase
 */
void computeHistogramParallelLocal(const cv::Mat &img, int hist_r[256], int hist_g[256], int hist_b[256])
{
    initHist3(hist_r, hist_g, hist_b);

#pragma omp parallel
    {
        // Private histograms for each thread
        int local_r[256] = {0};
        int local_g[256] = {0};
        int local_b[256] = {0};

#pragma omp for collapse(2)
        for (int i = 0; i < img.rows; i++)
        {
            for (int j = 0; j < img.cols; j++)
            {
                const cv::Vec3b &intensity = img.at<cv::Vec3b>(i, j);
                local_b[intensity[0]]++;
                local_g[intensity[1]]++;
                local_r[intensity[2]]++;
            }
        }

// Merge local histograms at the end
#pragma omp critical
        {
            for (int i = 0; i < 256; i++)
            {
                hist_r[i] += local_r[i];
                hist_g[i] += local_g[i];
                hist_b[i] += local_b[i];
            }
        }
    }
}

// --------------------------------------------------------------------------------
// Main
// --------------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    if (argc < 2 || argc > 3)
    {
        std::cout << "Usage: " << argv[0] << " <image_file> [num_threads]" << std::endl;
        std::cout << "  <image_file> : Path to the image file to process" << std::endl;
        std::cout << "  [num_threads]: Optional - Number of OpenMP threads to use (default: system maximum)" << std::endl;
        return -1;
    }

    // Set number of threads if specified
    if (argc == 3)
    {
        int num_threads = std::atoi(argv[2]);
        if (num_threads <= 0)
        {
            std::cerr << "Error: Number of threads must be positive" << std::endl;
            return -1;
        }
        omp_set_num_threads(num_threads);
        std::cout << "Using " << num_threads << " threads" << std::endl;
    }
    else
    {
        std::cout << "Using default number of threads: " << omp_get_max_threads() << std::endl;
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
    int hist_r_atomic[256], hist_g_atomic[256], hist_b_atomic[256];
    int hist_r_local[256], hist_g_local[256], hist_b_local[256];

    // Sequential
    auto start = std::chrono::high_resolution_clock::now();
    computeHistogramSequential(img, hist_r_seq, hist_g_seq, hist_b_seq);
    auto end = std::chrono::high_resolution_clock::now();
    double time_seq = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Sequential execution time: " << time_seq << " ms" << std::endl;

    // Parallel No Sync
    start = std::chrono::high_resolution_clock::now();
    computeHistogramParallelNoSync(img, hist_r_unsync, hist_g_unsync, hist_b_unsync);
    end = std::chrono::high_resolution_clock::now();
    double time_unsync = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Parallel (no sync) execution time: " << time_unsync << " ms" << std::endl;

    // Parallel Critical
    start = std::chrono::high_resolution_clock::now();
    computeHistogramParallelCritical(img, hist_r_crit, hist_g_crit, hist_b_crit);
    end = std::chrono::high_resolution_clock::now();
    double time_crit = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Parallel (critical) execution time: " << time_crit << " ms" << std::endl;

    // Parallel Atomic
    start = std::chrono::high_resolution_clock::now();
    computeHistogramParallelAtomic(img, hist_r_atomic, hist_g_atomic, hist_b_atomic);
    end = std::chrono::high_resolution_clock::now();
    double time_atomic = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Parallel (atomic) execution time: " << time_atomic << " ms" << std::endl;

    // Parallel Local
    start = std::chrono::high_resolution_clock::now();
    computeHistogramParallelLocal(img, hist_r_local, hist_g_local, hist_b_local);
    end = std::chrono::high_resolution_clock::now();
    double time_local = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Parallel (local histograms) execution time: " << time_local << " ms" << std::endl;

    // Compare results against sequential version
    bool matchUnsync = compareHist3(hist_r_seq, hist_g_seq, hist_b_seq, hist_r_unsync, hist_g_unsync, hist_b_unsync);
    bool matchCrit = compareHist3(hist_r_seq, hist_g_seq, hist_b_seq, hist_r_crit, hist_g_crit, hist_b_crit);
    bool matchAtomic = compareHist3(hist_r_seq, hist_g_seq, hist_b_seq, hist_r_atomic, hist_g_atomic, hist_b_atomic);
    bool matchLocal = compareHist3(hist_r_seq, hist_g_seq, hist_b_seq, hist_r_local, hist_g_local, hist_b_local);

    std::cout << "\nComparison with sequential version:" << std::endl;
    std::cout << "No sync:  " << (matchUnsync ? "MATCH ✓" : "NO MATCH ✗") << std::endl;
    std::cout << "Critical: " << (matchCrit ? "MATCH ✓" : "NO MATCH ✗") << std::endl;
    std::cout << "Atomic:   " << (matchAtomic ? "MATCH ✓" : "NO MATCH ✗") << std::endl;
    std::cout << "Local:    " << (matchLocal ? "MATCH ✓" : "NO MATCH ✗") << std::endl;

    // Calculate pixel counts
    int seqR = sumHistogram(hist_r_seq), seqG = sumHistogram(hist_g_seq), seqB = sumHistogram(hist_b_seq);
    int unsyncR = sumHistogram(hist_r_unsync), unsyncG = sumHistogram(hist_g_unsync), unsyncB = sumHistogram(hist_b_unsync);
    int critR = sumHistogram(hist_r_crit), critG = sumHistogram(hist_g_crit), critB = sumHistogram(hist_b_crit);
    int atomicR = sumHistogram(hist_r_atomic), atomicG = sumHistogram(hist_g_atomic), atomicB = sumHistogram(hist_b_atomic);
    int localR = sumHistogram(hist_r_local), localG = sumHistogram(hist_g_local), localB = sumHistogram(hist_b_local);

    std::cout << "\nTotal pixel counts per channel:" << std::endl;
    std::cout << "Sequential - R: " << seqR << " G: " << seqG << " B: " << seqB << std::endl;
    std::cout << "No sync    - R: " << unsyncR << " G: " << unsyncG << " B: " << unsyncB << std::endl;
    std::cout << "Critical   - R: " << critR << " G: " << critG << " B: " << critB << std::endl;
    std::cout << "Atomic     - R: " << atomicR << " G: " << atomicG << " B: " << atomicB << std::endl;
    std::cout << "Local      - R: " << localR << " G: " << localG << " B: " << localB << std::endl;

    // Calculate differences if histograms don't match
    if (!matchUnsync)
    {
        int diffR, diffG, diffB;
        computeTotalAbsDiff(hist_r_seq, hist_g_seq, hist_b_seq, hist_r_unsync, hist_g_unsync, hist_b_unsync, diffR, diffG, diffB);
        std::cout << "\nNo sync absolute differences - R: " << diffR << " G: " << diffG << " B: " << diffB << std::endl;
    }

    // Display histograms
    displayCombinedHistogram(hist_r_seq, hist_g_seq, hist_b_seq, "Sequential Histogram");
    displayCombinedHistogram(hist_r_unsync, hist_g_unsync, hist_b_unsync, "Parallel No Sync Histogram");
    displayCombinedHistogram(hist_r_crit, hist_g_crit, hist_b_crit, "Parallel Critical Histogram");
    displayCombinedHistogram(hist_r_atomic, hist_g_atomic, hist_b_atomic, "Parallel Atomic Histogram");
    displayCombinedHistogram(hist_r_local, hist_g_local, hist_b_local, "Parallel Local Histogram");

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

    std::cout << "\nPress ESC to exit..." << std::endl;
    while (true)
    {
        char key = cv::waitKey(20);
        if (key == 27)
            break;
    }
}