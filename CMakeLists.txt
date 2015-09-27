cmake_minimum_required(VERSION 2.8)
project(VideoTrainer)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake/modules)

find_package(Boost REQUIRED)
find_package(FANN REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(train main.cpp train.cpp video.cpp hog.cpp nnet.cpp)
target_link_libraries(train ${OpenCV_LIBS} fann)
