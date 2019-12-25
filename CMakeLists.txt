cmake_minimum_required (VERSION 3.14)
project ("Raven.FaceComparer")

set (EXTERNAL_LIB_DIR "${CMAKE_BINARY_DIR}/external" CACHE STRING "")
include(ExternalProject)

#force windows (Visual Studio) builds to use static runtime
if (POLICY CMP0054)
   cmake_policy(SET CMP0054 NEW)
endif()

#[[
if (MSVC OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    foreach(flag_var
        CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_DEBUG_INIT CMAKE_C_FLAGS_RELEASE
        CMAKE_C_FLAGS_MINSIZEREL CMAKE_C_FLAGS_RELWITHDEBINFO CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
        if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
        endif()
    endforeach(flag_var)
endif()
]]

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

include(cmake/build_dlib.cmake)
build_dlib(dlib)

include(${OPENCV_LIBRARY_DIR}/OpenCVConfig.cmake)
find_library(OpenCV REQUIRED)

set (DLIB_LIBRARY_DIR ${EXTERNAL_LIB_DIR}/dlib/lib/cmake/dlib)
include(${DLIB_LIBRARY_DIR}/dlibConfig.cmake)
find_library(dlib CONFIG REQUIRED)

message("=>>${OpenCV_INCLUDE_DIRS}")
message("=>>${OpenCV_LIBS}")
message("=>> DLIB Includes ${dlib_INCLUDE_DIRS}")
message("=>> DLIB Libs ${dlib_LIBRARIES}")
include_directories(${OpenCV_INCLUDE_DIRS} ${dlib_INCLUDE_DIRS})
link_libraries(${OpenCV_LIBS} ${dlib_LIBRARIES})

add_subdirectory ("Raven.FaceComparer")