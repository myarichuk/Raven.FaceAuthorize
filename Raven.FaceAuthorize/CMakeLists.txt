cmake_minimum_required (VERSION 3.13)

add_executable (Raven.FaceAuthorize "Raven.FaceAuthorize.cpp" "Raven.FaceAuthorize.h")
add_dependencies(Raven.FaceAuthorize OpenCV)
