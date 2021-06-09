

PROJECT(apps)

######################################################################
ADD_EXECUTABLE (camaquire cam_video.cpp )
target_link_libraries(camaquire
                      ${OpenCV_LIBS} )

######################################################################
ADD_EXECUTABLE (display display.cpp )
target_link_libraries(display
                      ${OpenCV_LIBS} )

######################################################################
add_executable(pointproc pointproc.cpp)
target_link_libraries(pointproc
                      IMAGE_PROC_FNS    #la bibliothèque static
                      ${OpenCV_LIBS} )

#######################################################################

add_executable(filtering filtering.cpp)
target_link_libraries(filtering
                      IMAGE_PROC_FNS    #la bibliothèque static
                      ${OpenCV_LIBS} )

#######################################################################

add_executable(morphology morphology.cpp)
target_link_libraries(morphology
                      IMAGE_PROC_FNS    #la bibliothèque static
                      ${OpenCV_LIBS} )

#######################################################################

add_executable(segmentation segmentation.cpp)
target_link_libraries(segmentation
                      IMAGE_PROC_FNS    #la bibliothèque static
                      ${OpenCV_LIBS} )

#######################################################################

add_executable(colourproc colourproc.cpp)
target_link_libraries(colourproc
                      IMAGE_PROC_FNS    #la bibliothèque static
                      ${OpenCV_LIBS} )
