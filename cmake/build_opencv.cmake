# This function is used to force a build on a dependant project at cmake configuration phase.
# credit: https://stackoverflow.com/a/23570741/320103
function (build_opencv target url)

    set(trigger_build_dir ${CMAKE_BINARY_DIR}/${target})

    #mktemp dir in build tree
    file(MAKE_DIRECTORY ${trigger_build_dir} ${trigger_build_dir}/build)

    #generate false dependency project
    set(CMAKE_LIST_CONTENT "
        cmake_minimum_required(VERSION 3.13)
        project(${target}_project_name)
        
        #[[
        if (MSVC OR \"${CMAKE_CXX_COMPILER_ID}\" STREQUAL \"MSVC\")
              foreach(flag_var
                 CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
                 CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
                 if(${flag_var} MATCHES \"/MD\")
                    string(REGEX REPLACE \"/MD\" \"/MT\" ${flag_var} \"${${flag_var}}\")
                 endif()
              endforeach(flag_var)          
        endif()
        ]]
        include(ExternalProject)
        set(CMAKE_CXX_STANDARD 17)

        ExternalProject_Add(${target}
          GIT_REPOSITORY ${url}
          GIT_TAG \"master\"
          SOURCE_DIR opencv
          BINARY_DIR opencv-build
          CMAKE_ARGS
            -DBUILD_opencv_core=ON
            -DBUILD_opencv_highgui=ON
            -DBUILD_opencv_imgproc=ON
            -DBUILD_opencv_contrib=ON
            -DBUILD_DOCS:BOOL=FALSE
            -DBUILD_EXAMPLES:BOOL=FALSE
            -DBUILD_TESTS:BOOL=FALSE
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DBUILD_SHARED_LIBS:BOOL=FALSE
            -DBUILD_NEW_PYTHON_SUPPORT:BOOL=OFF
            -DBUILD_WITH_DEBUG_INFO=OFF
            -DWITH_CUDA:BOOL=FALSE
            -DWITH_FFMPEG:BOOL=FALSE
            -DWITH_MSMF:BOOL=FALSE
            -DWITH_IPP:BOOL=FALSE
            -DBUILD_PERF_TESTS:BOOL=FALSE
            -DBUILD_PNG:BOOL=ON
            -DBUILD_JPEG:BOOL=ON
            -DBUILD_WITH_STATIC_CRT:BOOL=OFF
            -DBUILD_FAT_JAVA_LIB=OFF
            -DCMAKE_INSTALL_PREFIX:PATH=${EXTERNAL_LIB_DIR}/opencv
        )

        add_custom_target(trigger_${target})
        add_dependencies(trigger_${target} ${target})

    ")
    message("OpenCV CMAKE: ${CMAKE_LIST_CONTENT}")

    file(WRITE ${trigger_build_dir}/CMakeLists.txt "${CMAKE_LIST_CONTENT}")
    execute_process(COMMAND ${CMAKE_COMMAND} -G ${CMAKE_GENERATOR} ${trigger_build_dir} OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr WORKING_DIRECTORY ${trigger_build_dir} )
    message(STATUS "MAKE stdout='${stdout}'")
    message(STATUS "MAKE stderr='${stderr}'")

    execute_process(COMMAND ${CMAKE_COMMAND} --build ${trigger_build_dir} OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr WORKING_DIRECTORY ${trigger_build_dir} )
    message(STATUS "BUILD stdout='${stdout}'")
    message(STATUS "BUILD stderr='${stderr}'")

endfunction()