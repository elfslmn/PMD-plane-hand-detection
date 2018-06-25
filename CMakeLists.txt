cmake_minimum_required(VERSION 2.8)

project (PMDDetector)
# Please insert your OpenCV path
set(CMAKE_MODULE_PATH "/opt/opencv331/opencv")
set(ROYALE_DIR "/home/user/libroyale-3.11.0.42-LINUX-x86-64Bit")
set(INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include" )
set(SRC_DIR "${PROJECT_SOURCE_DIR}/src" )

include( CheckCXXCompilerFlag )
CHECK_CXX_COMPILER_FLAG( "-std=c++11" COMPILER_SUPPORTS_CXX11 )
CHECK_CXX_COMPILER_FLAG( "-std=c++0x" COMPILER_SUPPORTS_CXX0X )

if( COMPILER_SUPPORTS_CXX11 )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11" )
elseif( COMPILER_SUPPORTS_CXX0X )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x" )
else()
    message( STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler." )
endif()

set( Boost_USE_STATIC_LIBS ON )
set( Boost_USE_STATIC ON )

# This needs at least OpenCV 2.x, but it should also work with OpenCV 3.x
find_package( OpenCV REQUIRED )
if( OpenCV_FOUND )
   message( STATUS "Found OpenCV: ${OpenCV_INCLUDE_DIRS}" )
endif( OpenCV_FOUND )

set(CMAKE_PREFIX_PATH "${ROYALE_DIR}/share")
find_package(royale REQUIRED)
link_directories(${royale_LIB_DIR})

include_directories(
  ${INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(PMDDetector
  ${SRC_DIR}/mainPMD.cpp
  ${SRC_DIR}/Detector.cpp
  ${SRC_DIR}/PlaneDetector.cpp
  ${SRC_DIR}/Util.cpp
  ${SRC_DIR}/FrameObject.cpp
  ${SRC_DIR}/FramePlane.cpp
   )
   
COPY_ROYALE_LIBS(PMDDetector)
target_link_libraries(PMDDetector "${royale_LIBS}" "${OpenCV_LIBRARIES}")
