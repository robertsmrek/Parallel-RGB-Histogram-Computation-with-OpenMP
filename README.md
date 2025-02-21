# Parallel RGB Histogram Computation with OpenMP

This project demonstrates how to compute histograms for an RGB image using various parallelization strategies with OpenMP. It highlights the effects of race conditions when synchronization is omitted, and shows different approaches to ensure correct results.

The project includes:

-   **Sequential Histogram Computation:** A baseline single-threaded approach.
-   **Parallel (No Synchronization):** A parallel approach without synchronization (leading to race conditions).
-   **Parallel with Critical Sections:** A parallel method that uses `#pragma omp critical` to protect shared data.
-   **Parallel with Local Histograms:** Each thread computes its own histogram and then the results are merged (usually the best performance/accuracy trade-off).

Additionally, the project displays:

-   Single-channel histograms (Red, Green, and Blue) for the sequential method.
-   A combined RGB histogram for each approach.
-   A hue histogram (using HSV conversion) to illustrate the color distribution.

## Table of Contents

-   [Overview](#overview)
-   [Requirements](#requirements)
-   [Build Instructions](#build-instructions)
-   [Usage](#usage)
-   [Project Structure](#project-structure)
-   [How It Works](#how-it-works)
    -   [Histogram Computation Approaches](#histogram-computation-approaches)
    -   [Synchronization Techniques](#synchronization-techniques)
-   [Output Explanation](#output-explanation)
-   [Notes and Extensions](#notes-and-extensions)
-   [References](#references)

## Overview

The goal of this project is to efficiently compute the histogram of a color image in parallel using OpenMP, and to illustrate the importance of proper synchronization. The code computes histograms for the three channels (R, G, B) and compares several methods:

1. **Sequential:** No parallelization; used as the correctness baseline.
2. **Parallel (No Sync):** Uses OpenMP without any synchronization, demonstrating race conditions.
3. **Parallel (Critical):** Uses critical sections to protect shared histogram updates.
4. **Parallel (Local Histograms):** Each thread computes a local histogram, which is then merged.

The program also displays various histograms so you can visually compare the output of each approach.

## Requirements

-   **C++ Compiler** with OpenMP support (e.g., GCC or Clang)
-   **OpenCV** (version 4 or later recommended)
-   **pkg-config** (for retrieving OpenCV flags)

## Build Instructions

A `Makefile` is provided. To build the project:

```bash
make
```

This compiles hist.cpp with OpenMP and OpenCV flags and produces an executable named histogram.

To clean up the build files:

```bash
make clean
```

## Usage

The program requires an input image file as an argument. For example:

```bash
./histogram image.jpg
```

This will compute the histograms for the image and display the results.

## Project Structure

The project consists of the following files:

-   **hist.cpp:** The main source file that computes the histograms using OpenMP.
-   **Makefile:** A simple makefile for building the project.
-   **README.md:** This file, providing an overview of the project.
-   **LICENSE:** The license file for the project.

## How It Works

### Histogram Computation Approaches

1. **Sequential:** Loops over each pixel and updates the R, G, and B histogram arrays directly. No race conditions since it runs on a single thread. This method serves as the correctness baseline.

2. **Parallel (No Sync):** Uses OpenMP to parallelize the nested loops. All threads update the same shared histogram arrays simultaneously without protection, leading to race conditions and incorrect results.

3. **Parallel (Critical Sections):** Wraps each histogram update within a `#pragma omp critical` block, ensuring that only one thread updates the shared histogram at any given time. This prevents race conditions but may slow down the computation.

4. **Parallel (Local Histograms):** Each thread maintains its own local histogram arrays (private copies), avoiding race conditions during the pixel processing. After the parallel loop, these local histograms are merged into the global arrays. This method is typically the fastest correct approach.

### Synchronization Techniques

-   **No Synchronization:** Fast but produces incorrect results due to concurrent, unsynchronized updates. This approach demonstrates the importance of proper synchronization.

-   **Critical Sections:** Ensures correctness by serializing access to the histogram arrays, at the cost of potential performance loss due to contention.

-   **Local Histograms:** Minimizes synchronization overhead by deferring the merging of results until after the parallel processing is complete. Each thread computes its own local histograms, which are then combined into the final histograms.

## Output Explanation

The program displays the following histograms:

1. **Single-Channel Histograms:** The histograms for the Red, Green, and Blue channels are displayed for the sequential method.

2. **Combined RGB Histograms:** The combined RGB histogram is displayed for each approach (sequential, parallel without synchronization, parallel with critical sections, and parallel with local histograms).

3. **Hue Histogram:** The hue histogram is displayed to illustrate the color distribution in the image.

The histograms are displayed using OpenCV's `cv::plot::plotBars` function, which provides a visual representation of the histogram data.

## Notes and Extensions

-   **Performance:** The parallel (local histograms) approach is usually the best performance/accuracy trade-off. It minimizes contention by having each thread compute its own local histograms, which are then merged into the final histograms. This approach can be extended to other parallel programming models (e.g., MPI, CUDA).
