cmake_minimum_required (VERSION 3.13)
set(TESTING_CONFIG_OPT FILEPATH cmake/config.cmake)
include(cmake/HunterGate.cmake)

HunterGate(
    URL "https://github.com/cpp-pm/hunter/archive/v0.23.240.tar.gz"
    SHA1 "ca19f3769e6c80cfdd19d8b12ba5102c27b074e0"
)

project ("Raven.FaceComparer")

hunter_add_package(OpenSSL)
hunter_add_package(CURL)
hunter_add_package(ZLIB)
hunter_add_package(nlohmann_json)

find_package(nlohmann_json CONFIG REQUIRED)
find_package(ZLIB CONFIG REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(CURL CONFIG REQUIRED)

set (EXTERNAL_LIB_DIR "${CMAKE_BINARY_DIR}/external" CACHE STRING "")
set (XXHASH_SRC "${EXTERNAL_LIB_DIR}/xxhash")
include(FetchContent) 
FetchContent_Declare(
  xxhash_cpp
  GIT_REPOSITORY https://github.com/RedSpah/xxhash_cpp.git
  GIT_TAG        master
  SOURCE_DIR ${XXHASH_SRC}/repository SUBBUILD_DIR ${XXHASH_SRC}/subbuild  BINARY_DIR ${XXHASH_SRC}/binary
  )

FetchContent_GetProperties(xxhash_cpp)
if(NOT xxhash_cpp_POPULATED)  
  FetchContent_Populate(xxhash_cpp)
endif()

include(ExternalProject)

if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(">>>Debug build")
elseif(CMAKE_BUILD_TYPE MATCHES Release)
    message(">>>Release build")
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
  
set(HUNTER_STATUS_DEBUG ON)
include(cmake/build_raven_client.cmake)
build_raven_client(raven_client)

include(${EXTERNAL_LIB_DIR}/raven_client/Raven_CppClient.cmake)


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
include_directories(${OpenCV_INCLUDE_DIRS} ${dlib_INCLUDE_DIRS} ${EXTERNAL_LIB_DIR}/raven_client/include ${xxhash_cpp_SOURCE_DIR}/xxhash)
link_libraries(${OpenCV_LIBS} ${dlib_LIBRARIES} RavenDB::Raven_CppClient)

add_subdirectory ("Raven.FaceComparer")