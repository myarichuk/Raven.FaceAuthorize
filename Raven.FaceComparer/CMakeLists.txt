cmake_minimum_required (VERSION 3.13)
project(Raven.FaceComparer)

add_executable (Raven.FaceComparer "Raven.FaceComparer.cpp")

message("opencv ==> ${EXTERNAL_LIB_DIR}/opencv")
add_custom_command(
			TARGET Raven.FaceComparer POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/haarcascade
			COMMAND ${CMAKE_COMMAND} -E copy_directory ${EXTERNAL_LIB_DIR}/opencv/etc/haarcascades ${CMAKE_CURRENT_BINARY_DIR}/haarcascade
            )
