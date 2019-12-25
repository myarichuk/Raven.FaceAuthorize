#include "face_analyzer.h"


face_analyzer::face_analyzer(string filename)
{
    // The first thing we are going to do is load all our models.  First, since we need to
    // find faces in the image we will need a face detector:
    detector = get_frontal_face_detector();
    // We will also use a face landmarking model to align faces to a standard pose:  (see face_landmark_detection_ex.cpp for an introduction)
    
    deserialize("shape_predictor_5_face_landmarks.dat") >> sp;
    // And finally we load the DNN responsible for face recognition.
    
    deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;
    load_image(img, filename);   
}

void face_analyzer::detect_faces(std::vector<matrix<rgb_pixel>> &faces)
{
     //detect all faces on the image
    for (auto face : detector(img))
    {
        auto shape = sp(img, face);
        matrix<rgb_pixel> face_chip;
        extract_image_chip(img, get_face_chip_details(shape,150,0.25), face_chip);
        faces.push_back(move(face_chip));
    }
}

void face_analyzer::get_face_descriptors(const std::vector<matrix<rgb_pixel>> &faces, std::vector<matrix<float,0,1>> &face_descriptors)
{
    // This call asks the DNN to convert each face image in faces into a 128D vector.
    // In this 128D vector space, images from the same person will be close to each other
    // but vectors from different people will be far apart.  So we can use these vectors to
    // identify if a pair of images are from the same person or from different people.  
    face_descriptors = net(faces);
}

bool face_analyzer::are_similar(const matrix<float, 0, 1>& faceA, const matrix<float, 0, 1>& faceB) const
{
	return length(faceA - faceB) < 0.6;
}
