// Raven.FaceAuthorize.cpp : Defines the entry point for the application.
//

#include "Raven.FaceComparer.h"
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp> 
#include <opencv2/imgproc.hpp> 

using namespace std;
using namespace cv;

int main()
{
	const auto img = cv::imread("C:\\Users\\Michael.HRHINOS\\Pictures\\tempsnip.png");
	cout << "Hello CMake. " <<  img.cols << "x" << img.rows << endl;
	return 0;
}
