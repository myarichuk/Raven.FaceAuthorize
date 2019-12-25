cmake_minimum_required (VERSION 3.13)

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

add_executable (Raven.FaceComparer "Raven.FaceComparer.cpp")

add_custom_command(
			TARGET Raven.FaceComparer POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/haarcascade
			COMMAND ${CMAKE_COMMAND} -E copy_directory ${EXTERNAL_LIB_DIR}/opencv/etc/haarcascades ${CMAKE_CURRENT_BINARY_DIR}/haarcascade
            )
