execute_process(COMMAND "/home/ksavevska/dmpbbo/examples/catkin_ws/build/geometry2/tf2_geometry_msgs/catkin_generated/python_distutils_install.sh" RESULT_VARIABLE res)

if(NOT res EQUAL 0)
  message(FATAL_ERROR "execute_process(/home/ksavevska/dmpbbo/examples/catkin_ws/build/geometry2/tf2_geometry_msgs/catkin_generated/python_distutils_install.sh) returned error code ")
endif()
