cmake_minimum_required (VERSION 3.13)

include(cmake/HunterGate.cmake)
HunterGate(
    URL "https://github.com/ruslo/hunter/archive/v0.23.214.tar.gz"
    SHA1 "e14bc153a7f16d6a5eeec845fb0283c8fad8c358"
)
project ("Raven.FaceComparer")

set (EXTERNAL_LIB_DIR "${CMAKE_BINARY_DIR}/external" CACHE STRING "")
include(ExternalProject)

if(MSVC)
	set(CMAKE_CXX_FLAGS_DEBUG "/MTd /Zi /Ob0 /Od /RTC1")
	set(CMAKE_CXX_FLAGS_RELEASE "/MT /O2 /Ob2 /DNDEBUG")
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(">>>Debug build")
elseif(CMAKE_BUILD_TYPE MATCHES Release)
    message(">>>Release build")
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
else()
    message(">>>No build specified, defaulting to debug build")
    
endif()

include(cmake/build_opencv.cmake)
build_opencv(OpenCV http://github.com/Itseez/opencv.git)
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


hunter_add_package(dlib)

include(${OPENCV_LIBRARY_DIR}/OpenCVConfig.cmake)
find_library(OpenCV REQUIRED)
#find_package(OpenCV REQUIRED)

find_package(dlib CONFIG REQUIRED)

message("=>>${OpenCV_INCLUDE_DIRS}")
message("=>>${OpenCV_LIBS}")
message("=>> DLIB Includes ${dlib_INCLUDE_DIRS}")
include_directories(${OpenCV_INCLUDE_DIRS} ${dlib_INCLUDE_DIRS})
link_libraries(${OpenCV_LIBS} dlib::dlib)

add_subdirectory ("Raven.FaceComparer")