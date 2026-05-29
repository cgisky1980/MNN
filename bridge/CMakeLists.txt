cmake_minimum_required(VERSION 3.14)
project(mnn_dit_bridge CXX)

set(CMAKE_CXX_STANDARD 17)

add_library(mnn_dit_bridge SHARED mnn_dit_bridge.cpp)

target_include_directories(mnn_dit_bridge PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../3rd_party/half
)

if(NOT DEFINED MNN_BUILD_DIR)
    set(MNN_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../build")
endif()

if(WIN32)
    target_link_libraries(mnn_dit_bridge PRIVATE
        ${MNN_BUILD_DIR}/Release/MNN.lib
    )
    target_compile_options(mnn_dit_bridge PRIVATE /MD)
elseif(APPLE)
    target_link_libraries(mnn_dit_bridge PRIVATE
        ${MNN_BUILD_DIR}/libMNN.dylib
    )
else()
    target_link_libraries(mnn_dit_bridge PRIVATE
        ${MNN_BUILD_DIR}/libMNN.so
    )
endif()