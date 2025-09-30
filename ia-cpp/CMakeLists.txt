cmake_minimum_required(VERSION 3.15)
project(ia_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Paths for ONNX Runtime (set by Dockerfile build environment)
set(ONNXRUNTIME_ROOT "/opt/onnxruntime" CACHE PATH "ONNX Runtime root path")
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include" CACHE PATH "ONNX Runtime include directory")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/lib" CACHE PATH "ONNX Runtime lib directory")

include_directories(
  ${CMAKE_SOURCE_DIR}/third_party
  ${ONNXRUNTIME_INCLUDE_DIR}
)

link_directories(${ONNXRUNTIME_LIB_DIR})

add_executable(app
  src/main.cpp
)

# Link ONNX Runtime if available at build time; still allow build without it (dummy mode)
find_library(ONNXRUNTIME_LIB onnxruntime HINTS ${ONNXRUNTIME_LIB_DIR})
if(ONNXRUNTIME_LIB)
  target_link_libraries(app PRIVATE ${ONNXRUNTIME_LIB})
  target_compile_definitions(app PRIVATE WITH_ORT=1)
  message(STATUS "Linking with ONNX Runtime: ${ONNXRUNTIME_LIB}")
else()
  message(WARNING "ONNX Runtime not found; building with dummy inference only")
endif()

if(UNIX)
  target_link_libraries(app PRIVATE pthread)
  # Ensure runtime can find libonnxruntime.so
  set_target_properties(app PROPERTIES
    INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}"
    BUILD_WITH_INSTALL_RPATH TRUE
  )
endif()


