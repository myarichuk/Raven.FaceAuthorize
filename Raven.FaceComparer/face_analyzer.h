/*
 * Credit: heavily adapted code from http://dlib.net/dnn_face_recognition_ex.cpp.html
 * You can fetch the relevant DNN data from
 * 1) http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2
 * 2) http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2
 */

#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp> 
#include <opencv2/imgproc.hpp> 
#include <opencv2/core/types_c.h>
#include <dlib/opencv.h>
#include <dlib/dnn.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>

using namespace dlib;
using namespace std;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET> 
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                            input_rgb_image_sized<150>
                            >>>>>>>>>>>>;

class face_analyzer
{
private:
    frontal_face_detector detector;
    shape_predictor sp;
    anet_type net;

    //the image to analyze
	matrix<rgb_pixel> img;
public:
	face_analyzer(string filename);

    void detect_faces(std::vector<matrix<rgb_pixel>> &faces);
    void get_face_descriptors(const std::vector<matrix<rgb_pixel>> &faces, std::vector<matrix<float,0,1>> &face_descriptors);

    bool are_similar(const matrix<float, 0, 1>& faceA, const matrix<float, 0, 1>& faceB) const;

    //note: dlib image can be converted to OpenCV image like this: cv::Mat matImage = dlib::toMat<matrix<rgb_pixel>>(img);
    void draw_rectangle_over_face(cv::Mat& matImg, matrix<rgb_pixel> &face) const
    {
	   /* const std::vector<rectangle>::value_type face_rect = face;
	    cv::rectangle(matImg, 
	        cv::Point(face_rect.tl_corner().x(), face_rect.tl_corner().y()),
	        cv::Point(face_rect.br_corner().x(), face_rect.br_corner().y()), 
	        cv::Scalar(0,255,0));*/
    }
};

