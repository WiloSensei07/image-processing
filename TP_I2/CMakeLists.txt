PROJECT(IMAGE_PROC_TP)

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
#set the default path for built executables to the "bin" directory
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# Set optimized building:
IF(CMAKE_COMPILER_IS_GNUCXX AND NOT CMAKE_BUILD_TYPE MATCHES "Debug")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -std=c++11 -O3 -mtune=native")
ENDIF(CMAKE_COMPILER_IS_GNUCXX AND NOT CMAKE_BUILD_TYPE MATCHES "Debug")

#Include opencv package
FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS})
LINK_DIRECTORIES(${OpenCV_LIBS_DIR})

# define src directory for library functions
SET(LIBRARY_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src")

INCLUDE_DIRECTORIES(${LIBRARY_SOURCE_FILES})

SET(SRC  ${LIBRARY_SOURCE_FILES}/improcfuncs.h ${LIBRARY_SOURCE_FILES}/improcfuncs.cpp )
	
ADD_LIBRARY(IMAGE_PROC_FNS STATIC ${SRC})

TARGET_LINK_LIBRARIES(IMAGE_PROC_FNS ${OpenCV_LIBS})

ADD_SUBDIRECTORY(apps)
