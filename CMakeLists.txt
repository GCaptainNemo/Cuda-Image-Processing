cmake_minimum_required(VERSION 2.8)
project(cuda_imape_processing)

find_package(CUDA REQUIRED)
find_package(PythonLibs REQUIRED)

#set(OpenCV_FOUND 1)
#set(OpenCV_DIR C:/opencv/opencv/build)
#list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
#include_directories(C:/opencv/opencv/build/include)

include_directories(D:/pybind11/pybind11/include/)
include_directories(${PYTHON_INCLUDE_DIRS})

file(GLOB_RECURSE DIR_SRCS    ${CMAKE_CURRENT_SOURCE_DIR}/src/*)
file(GLOB_RECURSE DIR_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)

CUDA_ADD_EXECUTABLE(${PROJECT_NAME} 
					${DIR_SRCS}
					${DIR_HEADERS}
				)
target_link_libraries(${PROJECT_NAME}
		${OpenCV_LIBRARIES}
		${PYTHON_LIBRARIES}
)



