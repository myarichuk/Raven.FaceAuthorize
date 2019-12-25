# This function is used to force a build on a dependant project at cmake configuration phase.
# credit: https://stackoverflow.com/a/23570741/320103
function (build_dlib target)

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
                 CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_DEBUG_INIT CMAKE_C_FLAGS_RELEASE
                 CMAKE_C_FLAGS_MINSIZEREL CMAKE_C_FLAGS_RELWITHDEBINFO CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
                 CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
                 if(${flag_var} MATCHES \"/MD\")
                    string(REGEX REPLACE \"/MD\" \"/MT\" ${flag_var} \"${${flag_var}}\")
                 endif()
              endforeach(flag_var)
        endif()
        #]]

        include(ExternalProject)
        set(CMAKE_CXX_STANDARD 17)
        set(CMAKE_C_STANDARD 99)
        
        ExternalProject_Add(${target}
            GIT_REPOSITORY  \"https://github.com/davisking/dlib.git\"
            GIT_TAG \"master\"
            SOURCE_DIR dlib
            CMAKE_ARGS
            #-DDLIB_FORCE_MSVC_STATIC_RUNTIME:BOOL=ON
            -DBUILD_SHARED_LIBS:BOOL=OFF
            -DLIB_PNG_SUPPORT:BOOL=OFF
            -DLIB_NO_GUI_SUPPORT:BOOL=ON
            -DCMAKE_INSTALL_PREFIX:PATH=${EXTERNAL_LIB_DIR}/dlib           
        )
        add_custom_target(trigger_${target})
        add_dependencies(trigger_${target} ${target})

    ")
    message("dlib CMAKE: ${CMAKE_LIST_CONTENT}")

    file(WRITE ${trigger_build_dir}/CMakeLists.txt "${CMAKE_LIST_CONTENT}")
    execute_process(COMMAND ${CMAKE_COMMAND} -G ${CMAKE_GENERATOR} ${trigger_build_dir} OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr WORKING_DIRECTORY ${trigger_build_dir} )
    message(STATUS "MAKE stdout='${stdout}'")
    message(STATUS "MAKE stderr='${stderr}'")

    execute_process(COMMAND ${CMAKE_COMMAND} --build ${trigger_build_dir} OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr WORKING_DIRECTORY ${trigger_build_dir} )
    message(STATUS "BUILD stdout='${stdout}'")
    message(STATUS "BUILD stderr='${stderr}'")

endfunction()