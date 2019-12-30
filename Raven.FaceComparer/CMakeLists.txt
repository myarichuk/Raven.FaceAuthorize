cmake_minimum_required (VERSION 3.13)
project(Comparer)

file(GLOB SRC *.cpp)
file(GLOB HDR *.h)

message("headers ==> ${HDR}")
message("sources ==> ${SRC}")

add_executable (Raven.FaceComparer ${HDR} ${SRC})

add_custom_command(
			TARGET Raven.FaceComparer POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/dnn
			COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/dnn ${CMAKE_CURRENT_BINARY_DIR}/dnn
            )
