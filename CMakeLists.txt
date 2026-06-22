cmake_minimum_required(VERSION 3.15)

project(FederatedLearningOpenMPProject LANGUAGES C CXX)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# OpenMP
find_package(OpenMP REQUIRED)


# OS detection
if(WIN32)

    message(STATUS "Building for Windows")
    add_compile_definitions(OS_WINDOWS)

elseif(UNIX)

    message(STATUS "Building for Linux")
    add_compile_definitions(OS_LINUX)

endif()



# Compiler flags

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
    OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")

    message(STATUS "Using Clang")
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
    )

elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")

    message(STATUS "Using GCC")
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
    )

elseif(MSVC)

    add_compile_options(/W4)

endif()



add_executable(inferenceModel
    inference_model.cpp
)


add_executable(masterNode
    master.cpp
)


add_executable(workerNode
    worker.cpp
)



# Link OpenMP ke master dan worker

target_link_libraries(
    masterNode
    OpenMP::OpenMP_CXX
)


target_link_libraries(
    workerNode
    OpenMP::OpenMP_CXX
)