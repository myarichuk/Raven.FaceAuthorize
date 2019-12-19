// Raven.FaceAuthorize.cpp : Defines the entry point for the application.
//

#include "Raven.FaceAuthorize.h"
#include <opencv2/opencv.hpp>

using namespace std;

int main()
{
	const auto img = cv::imread("C:\\Users\\Michael.HRHINOS\\Pictures\\tempsnip.png");
	cout << "Hello CMake. " <<  img.cols << "x" << img.rows << endl;
	return 0;
}
