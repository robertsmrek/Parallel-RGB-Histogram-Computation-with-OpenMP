# Compiler
CXX = g++

# Retrieve OpenCV flags using pkg-config
OPENCV_CFLAGS = $(shell pkg-config --cflags opencv4)
OPENCV_LIBS = $(shell pkg-config --libs opencv4)

# Compiler Flags
CXXFLAGS = $(OPENCV_CFLAGS) -Wall -O2 -fopenmp

# Output Executable Name
TARGET = histogram

# Source Files
SRC = histogram.cpp

# Compilation Rule
all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(SRC) -o $(TARGET) $(CXXFLAGS) $(OPENCV_LIBS)

# Clean Rule
clean:
	rm -f $(TARGET)
