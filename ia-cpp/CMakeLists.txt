cmake_minimum_required(VERSION 3.16)
project(ia-cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Create executable first
add_executable(${PROJECT_NAME} src/main.cpp)

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE include)

# Check for ONNX Runtime
if(EXISTS "/opt/onnxruntime")
    message(STATUS "ONNX Runtime found at /opt/onnxruntime")
    target_compile_definitions(${PROJECT_NAME} PRIVATE WITH_ORT)
    
    # Include directories for ONNX Runtime
    target_include_directories(${PROJECT_NAME} PRIVATE /opt/onnxruntime/include)
    
    # Link ONNX Runtime
    target_link_libraries(${PROJECT_NAME} /opt/onnxruntime/lib/libonnxruntime.so)
    
    # Set library path
    set_target_properties(${PROJECT_NAME} PROPERTIES
        INSTALL_RPATH "/opt/onnxruntime/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
else()
    message(STATUS "ONNX Runtime not found, compiling without ORT support")
endif()

# Link pthread on UNIX systems
if(UNIX)
    find_package(Threads REQUIRED)
    target_link_libraries(${PROJECT_NAME} Threads::Threads)
endif()