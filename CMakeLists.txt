cmake_minimum_required(VERSION 2.8)
project(conv)

find_package(CUDA REQUIRED)
#set(OpenCV_FOUND 1)
#set(OpenCV_DIR C:/opencv/opencv/build)
#list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB_RECURSE DIR_SRCS    ${CMAKE_CURRENT_SOURCE_DIR}/src/*)
file(GLOB_RECURSE DIR_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)

CUDA_ADD_EXECUTABLE(${PROJECT_NAME} 
					${DIR_SRCS}
					${DIR_HEADERS}
				)
target_link_libraries(${PROJECT_NAME}
					  ${OpenCV_LIBRARIES}
)



