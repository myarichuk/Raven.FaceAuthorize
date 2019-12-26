# This function is used to force a build on a dependant project at cmake configuration phase.
# credit: https://stackoverflow.com/a/23570741/320103
function (build_raven_client target)

    set(trigger_build_dir ${CMAKE_BINARY_DIR}/${target})

    #mktemp dir in build tree
    file(MAKE_DIRECTORY ${trigger_build_dir} ${trigger_build_dir}/build)

    #generate false dependency project
    set(CMAKE_LIST_CONTENT "
        cmake_minimum_required(VERSION 3.13)
        project(${target}_project_name)
        
        include(ExternalProject)
        set(CMAKE_CXX_STANDARD 17)

        ExternalProject_Add(${target}
          GIT_REPOSITORY \"https://github.com/myarichuk/ravendb-cpp-client.git\"
          GIT_TAG \"hunter-gate-poc\"
          SOURCE_DIR raven-client
          BINARY_DIR raven-client-build
          CMAKE_ARGS
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_INSTALL_PREFIX:PATH=\"${EXTERNAL_LIB_DIR}/raven-client\"
        )

        add_custom_target(trigger_${target})
        add_dependencies(trigger_${target} ${target})

    ")
    message("OpenCV CMAKE: ${CMAKE_LIST_CONTENT}")

    file(WRITE ${trigger_build_dir}/CMakeLists.txt "${CMAKE_LIST_CONTENT}")
    execute_process(COMMAND ${CMAKE_COMMAND} -G ${CMAKE_GENERATOR} ${trigger_build_dir} OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr WORKING_DIRECTORY ${trigger_build_dir} )
    message(STATUS "MAKE stdout='${stdout}'")
    
    execute_process(COMMAND ${CMAKE_COMMAND} --build ${trigger_build_dir} OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr WORKING_DIRECTORY ${trigger_build_dir} )
    message(STATUS "BUILD stdout='${stdout}'")
    
endfunction()