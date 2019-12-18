cmake_minimum_required (VERSION 3.13)

project ("Raven.FaceAuthorize")

set (EXTERNAL_LIB_DIR "${CMAKE_BINARY_DIR}/external")
include(ExternalProject)
ExternalProject_Add(OpenCV
  GIT_REPOSITORY "http://github.com/Itseez/opencv.git"
  GIT_TAG "master"
  SOURCE_DIR opencv
  BINARY_DIR opencv-build
  CMAKE_ARGS
    -DBUILD_DOCS:BOOL=FALSE
    -DBUILD_EXAMPLES:BOOL=FALSE
    -DBUILD_TESTS:BOOL=FALSE
    -DBUILD_SHARED_LIBS:BOOL=FALSE
    -DWITH_CUDA:BOOL=FALSE
    -DWITH_FFMPEG:BOOL=FALSE
    -DBUILD_PERF_TESTS:BOOL=FALSE
    -DBUILD_PNG:BOOL=ON
    -DBUILD_JPEG:BOOL=ON
    -DBUILD_WITH_STATIC_CRT:BOOL=ON
    -DBUILD_FAT_JAVA_LIB=OFF
    -DCMAKE_INSTALL_PREFIX:PATH=${EXTERNAL_LIB_DIR}/opencv
)

set(OPENCV_INCLUDE_DIR ${EXTERNAL_LIB_DIR}/opencv/include/)
if(NOT WIN32)
  set(OPENCV_LIBRARY_DIR ${EXTERNAL_LIB_DIR}/opencv/lib/)
else()
  if("${CMAKE_SIZEOF_VOID_P}" STREQUAL "8")
    set(ARCH "x64")
  else()
    set(ARCH "x86")
  endif()

  if("${MSVC_TOOLSET_VERSION}" EQUAL "142")
    set(VCVER 16)
  elseif("${MSVC_TOOLSET_VERSION}" EQUAL "141")
    set(VCVER 15)
  elseif("${MSVC_TOOLSET_VERSION}" EQUAL "140")
    set(VCVER 14)
  else()
      message(FATAL_ERROR "MSVC Toolset less with ver < 14.0 is not supported")
  endif() 

  set(OPENCV_LIBRARY_DIR ${EXTERNAL_LIB_DIR}/opencv/${ARCH}/vc${VCVER}/staticlib)
endif()

set(OPENCV_LIBRARIES opencv_imgproc opencv_core)
message("OpenCV include folder ==> ${OPENCV_INCLUDE_DIR}")
message("OpenCV libraries folder ==> ${OPENCV_LIBRARY_DIR}")
include_directories(${OPENCV_INCLUDE_DIR})
link_directories(${OPENCV_LIBRARY_DIR})

add_subdirectory ("Raven.FaceAuthorize")
