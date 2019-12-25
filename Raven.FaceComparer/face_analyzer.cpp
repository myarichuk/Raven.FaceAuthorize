#include "face_analyzer.h"


face_analyzer::face_analyzer()
{
    // The first thing we are going to do is load all our models.  First, since we need to
    // find faces in the image we will need a face detector:
    detector = get_frontal_face_detector();
    // We will also use a face landmarking model to align faces to a standard pose:  (see face_landmark_detection_ex.cpp for an introduction)
    
    deserialize("shape_predictor_5_face_landmarks.dat") >> sp;
    // And finally we load the DNN responsible for face recognition.
    
    deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;
}

void face_analyzer::detect_faces(matrix<rgb_pixel> &img, std::vector<matrix<rgb_pixel>> &faces)
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

    // This next bit of code creates a graph of connected faces and then uses the Chinese whispers graph clustering
    // algorithm to identify how many people there are and which faces belong to whom.
    //std::vector<sample_pair> edges;
    //for (size_t i = 0; i < face_descriptors.size(); ++i)
    //{
    //    for (size_t j = i; j < face_descriptors.size(); ++j)
    //    {
    //        // Faces are connected in the graph if they are close enough.  Here we check if
    //        // the distance between two face descriptors is less than 0.6, which is the
    //        // decision threshold the network was trained to use.  Although you can
    //        // certainly use any other threshold you find useful.
    //        if (length(face_descriptors[i]-face_descriptors[j]) < 0.6)
    //            edges.push_back(sample_pair(i,j));
    //    }
    //}
    //std::vector<unsigned long> labels;
    //const auto num_clusters = chinese_whispers(edges, labels);

    //// Now let's display the face clustering results on the screen.  You will see that it
    //// correctly grouped all the faces. 
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
}