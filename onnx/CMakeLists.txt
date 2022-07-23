# project name
PROJECT(onnx)
# requirement of cmake version
cmake_minimum_required(VERSION 3.5)

# set the directory of executable files
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${onnx_SOURCE_DIR}/bin)

# find required opencv
find_package(OpenCV REQUIRED)
# directory of opencv headers
include_directories(${OpenCV_INCLUDE_DIRS})

#onnxruntime
set(ONNXRUNTIME_ROOT_PATH /home/linwis/Code/onnxruntime-linux-x64-1.11.1/)
set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_ROOT_PATH}/include/)
set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT_PATH}lib/libonnxruntime.so)

# name of executable file and path of source file
add_executable(onnx src/imageSeg.cpp)
# directory of opencv library
link_directories(${OpenCV_LIBRARY_DIRS})
# opencv libraries
target_link_libraries(onnx ${OpenCV_LIBS})

include_directories(${ONNXRUNTIME_INCLUDE_DIRS})
target_link_libraries(onnx ${ONNXRUNTIME_LIB})

