#include "face_analyzer.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp> 
#include <dlib/opencv.h>
#include <dlib/dnn.h>
#include <dlib/image_io.h>

#include <iostream>

using namespace std;

int main(int argc, char** argv)
{
	try
	{
        matrix<rgb_pixel> me_img;
		load_image(me_img, "me.jpg");
        matrix<rgb_pixel> not_me;
        load_image(not_me, "not_me.jpg");
        matrix<rgb_pixel> me_with_others;
        load_image(me_with_others, "me_with_others.jpg");        

        cout << "loaded images" <<endl;
	    face_analyzer fa;

		std::vector<matrix<rgb_pixel>> faces;
     	std::vector<matrix<rgb_pixel>> faces2;

		fa.detect_faces(me_img,faces);
	    fa.detect_faces(me_with_others,faces2);

        cout << "finished with face detection" << endl;

        std::vector<matrix<float,0,1>> face_descriptors;
        fa.get_face_descriptors(faces, face_descriptors);

		std::vector<matrix<float,0,1>> face_descriptors2;
        fa.get_face_descriptors(faces2, face_descriptors2);

        cout << "finished detecting facial features" << endl;

        for(size_t i = 0; i < face_descriptors2.size(); i++)
        {
            cout << i << ") length(face_descriptor2 - face_descriptors[0]=" << length(face_descriptors2[i] - face_descriptors[0]) <<endl;
	        if(length(face_descriptors2[i] - face_descriptors[0]) < 0.6)
	        {
                cout << "found me on the second image!" << endl;
		        cv::Mat me_on_other_image = dlib::toMat<matrix<rgb_pixel>>(faces2[i]);
                cv::imshow("Me in 'me_with_others.jpg'", me_on_other_image);
                cv::waitKey(0);
	        }
        }
        

		return 0;
	}
	catch (std::exception& e)
	{
	    cout << e.what() << endl;
	}
}

/*
    if (argc != 2)
    {
        cout << "Run this example by invoking it like this: " << endl;
        cout << "   ./dnn_face_recognition_ex faces/bald_guys.jpg" << endl;
        cout << endl;
        cout << "You will also need to get the face landmarking model file as well as " << endl;
        cout << "the face recognition model file.  Download and then decompress these files from: " << endl;
        cout << "http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2" << endl;
        cout << "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2" << endl;
        cout << endl;
        return 1;
    }

    // The first thing we are going to do is load all our models.  First, since we need to
    // find faces in the image we will need a face detector:
    frontal_face_detector detector = get_frontal_face_detector();
    // We will also use a face landmarking model to align faces to a standard pose:  (see face_landmark_detection_ex.cpp for an introduction)
    shape_predictor sp;
    deserialize("shape_predictor_5_face_landmarks.dat") >> sp;
    // And finally we load the DNN responsible for face recognition.
    anet_type net;
    deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;

    matrix<rgb_pixel> img;
    load_image(img, argv[1]);

    // Display the raw image on the screen
    //image_window win(img); 
    cv::Mat matImage = dlib::toMat<matrix<rgb_pixel>>(img);
	

    // Run the face detector on the image of our action heroes, and for each face extract a
    // copy that has been normalized to 150x150 pixels in size and appropriately rotated
    // and centered.
    std::vector<matrix<rgb_pixel>> faces;
    for (auto face : detector(img))
    {
        auto shape = sp(img, face);
        matrix<rgb_pixel> face_chip;
        extract_image_chip(img, get_face_chip_details(shape,150,0.25), face_chip);
        faces.push_back(move(face_chip));
        // Also put some boxes on the faces so we can see that the detector is finding
        // them.

        cv::rectangle(matImage, cv::Point(face.tl_corner().x(), face.tl_corner().y()),cv::Point(face.br_corner().x(), face.br_corner().y()), cv::Scalar(0,255,0));
        //win.add_overlay(face);
    }

    cv::imshow("Display", matImage);

    if (faces.empty())
    {
        cout << "No faces found in image!" << endl;
        return 1;
    }

    // This call asks the DNN to convert each face image in faces into a 128D vector.
    // In this 128D vector space, images from the same person will be close to each other
    // but vectors from different people will be far apart.  So we can use these vectors to
    // identify if a pair of images are from the same person or from different people.  
    std::vector<matrix<float,0,1>> face_descriptors = net(faces);


    // In particular, one simple thing we can do is face clustering.  This next bit of code
    // creates a graph of connected faces and then uses the Chinese whispers graph clustering
    // algorithm to identify how many people there are and which faces belong to whom.
    std::vector<sample_pair> edges;
    for (size_t i = 0; i < face_descriptors.size(); ++i)
    {
        for (size_t j = i; j < face_descriptors.size(); ++j)
        {
            // Faces are connected in the graph if they are close enough.  Here we check if
            // the distance between two face descriptors is less than 0.6, which is the
            // decision threshold the network was trained to use.  Although you can
            // certainly use any other threshold you find useful.
            if (length(face_descriptors[i]-face_descriptors[j]) < 0.6)
                edges.push_back(sample_pair(i,j));
        }
    }
    std::vector<unsigned long> labels;
    const auto num_clusters = chinese_whispers(edges, labels);
    // This will correctly indicate that there are 4 people in the image.
    cout << "number of people found in the image: "<< num_clusters << endl;


    // Now let's display the face clustering results on the screen.  You will see that it
    // correctly grouped all the faces. 
    //std::vector<image_window> win_clusters(num_clusters);
    //for (size_t cluster_id = 0; cluster_id < num_clusters; ++cluster_id)
    //{
    //    std::vector<matrix<rgb_pixel>> temp;
    //    for (size_t j = 0; j < labels.size(); ++j)
    //    {
    //        if (cluster_id == labels[j])
    //            temp.push_back(faces[j]);
    //    }
    //    win_clusters[cluster_id].set_title("face cluster " + cast_to_string(cluster_id));
    //    win_clusters[cluster_id].set_image(tile_images(temp));
    //}




    // Finally, let's print one of the face descriptors to the screen.  
    cout << "face descriptor for one face: " << trans(face_descriptors[0]) << endl;

    // It should also be noted that face recognition accuracy can be improved if jittering
    // is used when creating face descriptors.  In particular, to get 99.38% on the LFW
    // benchmark you need to use the jitter_image() routine to compute the descriptors,
    // like so:
    //matrix<float,0,1> face_descriptor = mean(mat(net(jitter_image(faces[0]))));
    //cout << "jittered face descriptor for one face: " << trans(face_descriptor) << endl;
    // If you use the model without jittering, as we did when clustering the bald guys, it
    // gets an accuracy of 99.13% on the LFW benchmark.  So jittering makes the whole
    // procedure a little more accurate but makes face descriptor calculation slower.


    cout << "hit enter to terminate" << endl;
    cv::waitKey(0); 
 
 */


//using namespace std;
//using namespace cv;
//
//void detectAndDraw( Mat& img, CascadeClassifier& cascade, 
//                    CascadeClassifier& nestedCascade, 
//                    double scale);
//int main()
//{
// // VideoCapture class for playing video for which faces to be detected 
//    VideoCapture capture;  
//    Mat frame, image; 
//  
//    // PreDefined trained XML classifiers with facial features 
//    CascadeClassifier cascade, nestedCascade;  
//    double scale=1; 
//  
//    // Load classifiers from "opencv/data/haarcascades" directory  
//    nestedCascade.load( "./haarcascade/haarcascade_eye_tree_eyeglasses.xml" ) ; 
//  
//    // Change path before execution  
//    cascade.load( "./haarcascade/haarcascade_frontalcatface.xml" ) ;  
//  
//    // Start Video..1) 0 for WebCam 2) "Path to Video" for a Local Video 
//    capture.open(0);  
//    if( capture.isOpened() ) 
//    { 
//        // Capture frames from video and detect faces 
//        cout << "Face Detection Started...." << endl; 
//        while(1) 
//        { 
//            capture >> frame; 
//            if( frame.empty() ) 
//                break; 
//            Mat frame1 = frame.clone(); 
//            detectAndDraw( frame1, cascade, nestedCascade, scale );  
//            char c = (char)waitKey(10); 
//          
//            // Press q to exit from window 
//            if( c == 27 || c == 'q' || c == 'Q' )  
//                break; 
//        } 
//    } 
//    else
//        cout<<"Could not Open Camera"; 
//    return 0; 
// //CascadeClassifier cascade, nestedCascade;  
// //   double scale=1;
//
//	//  // Load classifiers from "opencv/data/haarcascades" directory  
// //   nestedCascade.load( "./haarcascade/haarcascade_eye_tree_eyeglasses.xml" ) ; 
// // 
// //   // Change path before execution  
// //   cascade.load( "./haarcascade/haarcascade_frontalcatface.xml" ) ;  
//
// //   auto photo = imread("E:\\me.jpg");
// //   detectAndDraw( photo, cascade, nestedCascade, scale );
// //   waitKey(0);
//	return 0;
//}
//
//void detectAndDraw( Mat& img, CascadeClassifier& cascade, 
//                    CascadeClassifier& nestedCascade, 
//                    double scale) 
//{ 
//    vector<Rect> faces, faces2; 
//    Mat gray, smallImg; 
//  
//    cvtColor( img, gray, COLOR_BGR2GRAY ); // Convert to Gray Scale 
//    double fx = 1 / scale; 
//  
//    // Resize the Grayscale Image  
//    resize( gray, smallImg, Size(), fx, fx, INTER_LINEAR );  
//    equalizeHist( smallImg, smallImg ); 
//  
//    // Detect faces of different sizes using cascade classifier  
//    cascade.detectMultiScale( smallImg, faces, 1.1,  2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) ); 
//
//
//    // Draw circles around the faces 
//    for ( size_t i = 0; i < faces.size(); i++ ) 
//    { 
//        Rect r = faces[i]; 
//		rectangle(img, r,Scalar(0,255,0));
//
//        Mat smallImgROI; 
//        vector<Rect> nestedObjects; 
//        Point center; 
//        Scalar color = Scalar(255, 0, 0); // Color for Drawing tool 
//        int radius;
//
//        const double aspect_ratio = static_cast<double>(r.width)/r.height; 
//        if( 0.75 < aspect_ratio && aspect_ratio < 1.3 ) 
//        { 
//            center.x = cvRound((r.x + r.width*0.5)*scale); 
//            center.y = cvRound((r.y + r.height*0.5)*scale); 
//            radius = cvRound((r.width + r.height)*0.25*scale); 
//            circle( img, center, radius, color, 3, 8, 0 ); 
//        } 
//        else
//            rectangle( img, cvPoint(cvRound(r.x*scale), cvRound(r.y*scale)), 
//                    cvPoint(cvRound((r.x + r.width-1)*scale),  
//                    cvRound((r.y + r.height-1)*scale)), color, 3, 8, 0); 
//        if( nestedCascade.empty() ) 
//            continue; 
//        smallImgROI = smallImg( r ); 
//          
//        // Detection of eyes int the input image 
//        nestedCascade.detectMultiScale( smallImgROI, nestedObjects, 1.1, 2, 
//                                        0|CASCADE_SCALE_IMAGE, Size(30, 30) );  
//          
//        // Draw circles around eyes 
//        for ( size_t j = 0; j < nestedObjects.size(); j++ )  
//        { 
//            Rect nr = nestedObjects[j]; 
//            center.x = cvRound((r.x + nr.x + nr.width*0.5)*scale); 
//            center.y = cvRound((r.y + nr.y + nr.height*0.5)*scale); 
//            radius = cvRound((nr.width + nr.height)*0.25*scale); 
//            circle( img, center, radius, color, 3, 8, 0 ); 
//        } 
//    } 
//  
//    // Show Processed Image with detected faces 
//    imshow( "Face Detection", img );  
//} 
