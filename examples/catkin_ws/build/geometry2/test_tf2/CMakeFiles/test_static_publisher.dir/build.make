# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.19

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ksavevska/dmpbbo/examples/catkin_ws/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ksavevska/dmpbbo/examples/catkin_ws/build

# Include any dependencies generated for this target.
include geometry2/test_tf2/CMakeFiles/test_static_publisher.dir/depend.make

# Include the progress variables for this target.
include geometry2/test_tf2/CMakeFiles/test_static_publisher.dir/progress.make

# Include the compile flags for this target's objects.
include geometry2/test_tf2/CMakeFiles/test_static_publisher.dir/flags.make

geometry2/test_tf2/CMakeFiles/test_static_publisher.dir/test/test_static_publisher.cpp.o: geometry2/test_tf2/CMakeFiles/test_static_publisher.dir/flags.make
geometry2/test_tf2/CMakeFiles/test_static_publisher.dir/test/test_static_publisher.cpp.o: /home/ksavevska/dmpbbo/examples/catkin_ws/src/geometry2/test_tf2/test/test_static_publisher.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ksavevska/dmpbbo/examples/catkin_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object geometry2/test_tf2/CMakeFiles/test_static_publisher.dir/test/test_static_publisher.cpp.o"
	cd /home/ksavevska/dmpbbo/examples/catkin_ws/build/geometry2/test_tf2 && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_static_publisher.dir/test/test_static_publisher.cpp.o -c /home/ksavevska/dmpbbo/examples/catkin_ws/src/geometry2/test_tf2/test/test_static_publisher.cpp

geometry2/test_tf2/CMakeFiles/test_static_publisher.dir/test/test_static_publisher.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_static_publisher.dir/test/test_static_publisher.cpp.i"
	cd /home/ksavevska/dmpbbo/examples/catkin_ws/build/geometry2/test_tf2 && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ksavevska/dmpbbo/examples/catkin_ws/src/geometry2/test_tf2/test/test_static_publisher.cpp > CMakeFiles/test_static_publisher.dir/test/test_static_publisher.cpp.i

geometry2/test_tf2/CMakeFiles/test_static_publisher.dir/test/test_static_publisher.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_static_publisher.dir/test/test_static_publisher.cpp.s"
	cd /home/ksavevska/dmpbbo/examples/catkin_ws/build/geometry2/test_tf2 && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ksavevska/dmpbbo/examples/catkin_ws/src/geometry2/test_tf2/test/test_static_publisher.cpp -o CMakeFiles/test_static_publisher.dir/test/test_static_publisher.cpp.s

# Object files for target test_static_publisher
test_static_publisher_OBJECTS = \
"CMakeFiles/test_static_publisher.dir/test/test_static_publisher.cpp.o"

# External object files for target test_static_publisher
test_static_publisher_EXTERNAL_OBJECTS =

/home/ksavevska/dmpbbo/examples/catkin_ws/devel/lib/test_tf2/test_static_publisher: geometry2/test_tf2/CMakeFiles/test_static_publisher.dir/test/test_static_publisher.cpp.o
/home/ksavevska/dmpbbo/examples/catkin_ws/devel/lib/test_tf2/test_static_publisher: geometry2/test_tf2/CMakeFiles/test_static_publisher.dir/build.make
/home/ksavevska/dmpbbo/examples/catkin_ws/devel/lib/test_tf2/test_static_publisher: /usr/lib/x86_64-linux-gnu/libboost_thread.so
/home/ksavevska/dmpbbo/examples/catkin_ws/devel/lib/test_tf2/test_static_publisher: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
/home/ksavevska/dmpbbo/examples/catkin_ws/devel/lib/test_tf2/test_static_publisher: /usr/lib/x86_64-linux-gnu/libboost_system.so
/home/ksavevska/dmpbbo/examples/catkin_ws/devel/lib/test_tf2/test_static_publisher: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
/home/ksavevska/dmpbbo/examples/catkin_ws/devel/lib/test_tf2/test_static_publisher: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
/home/ksavevska/dmpbbo/examples/catkin_ws/devel/lib/test_tf2/test_static_publisher: /opt/ros/melodic/lib/libtf.so
/home/ksavevska/dmpbbo/examples/catkin_ws/devel/lib/test_tf2/test_static_publisher: /opt/ros/melodic/lib/liborocos-kdl.so
/home/ksavevska/dmpbbo/examples/catkin_ws/devel/lib/test_tf2/test_static_publisher: /home/ksavevska/dmpbbo/examples/catkin_ws/devel/lib/libtf2_ros.so
/home/ksavevska/dmpbbo/examples/catkin_ws/devel/lib/test_tf2/test_static_publisher: /opt/ros/melodic/lib/libactionlib.so
/home/ksavevska/dmpbbo/examples/catkin_ws/devel/lib/test_tf2/test_static_publisher: /opt/ros/melodic/lib/libmessage_filters.so
/home/ksavevska/dmpbbo/examples/catkin_ws/devel/lib/test_tf2/test_static_publisher: /opt/ros/melodic/lib/libroscpp.so
/home/ksavevska/dmpbbo/examples/catkin_ws/devel/lib/test_tf2/test_static_publisher: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
/home/ksavevska/dmpbbo/examples/catkin_ws/devel/lib/test_tf2/test_static_publisher: /opt/ros/melodic/lib/librosconsole.so
/home/ksavevska/dmpbbo/examples/catkin_ws/devel/lib/test_tf2/test_static_publisher: /opt/ros/melodic/lib/librosconsole_log4cxx.so
/home/ksavevska/dmpbbo/examples/catkin_ws/devel/lib/test_tf2/test_static_publisher: /opt/ros/melodic/lib/librosconsole_backend_interface.so
/home/ksavevska/dmpbbo/examples/catkin_ws/devel/lib/test_tf2/test_static_publisher: /usr/lib/x86_64-linux-gnu/liblog4cxx.so
/home/ksavevska/dmpbbo/examples/catkin_ws/devel/lib/test_tf2/test_static_publisher: /usr/lib/x86_64-linux-gnu/libboost_regex.so
/home/ksavevska/dmpbbo/examples/catkin_ws/devel/lib/test_tf2/test_static_publisher: /opt/ros/melodic/lib/libxmlrpcpp.so
/home/ksavevska/dmpbbo/examples/catkin_ws/devel/lib/test_tf2/test_static_publisher: /home/ksavevska/dmpbbo/examples/catkin_ws/devel/lib/libtf2.so
/home/ksavevska/dmpbbo/examples/catkin_ws/devel/lib/test_tf2/test_static_publisher: /opt/ros/melodic/lib/liborocos-kdl.so.1.4.0
/home/ksavevska/dmpbbo/examples/catkin_ws/devel/lib/test_tf2/test_static_publisher: /opt/ros/melodic/lib/libroscpp_serialization.so
/home/ksavevska/dmpbbo/examples/catkin_ws/devel/lib/test_tf2/test_static_publisher: /opt/ros/melodic/lib/librostime.so
/home/ksavevska/dmpbbo/examples/catkin_ws/devel/lib/test_tf2/test_static_publisher: /opt/ros/melodic/lib/libcpp_common.so
/home/ksavevska/dmpbbo/examples/catkin_ws/devel/lib/test_tf2/test_static_publisher: /usr/lib/x86_64-linux-gnu/libboost_system.so
/home/ksavevska/dmpbbo/examples/catkin_ws/devel/lib/test_tf2/test_static_publisher: /usr/lib/x86_64-linux-gnu/libboost_thread.so
/home/ksavevska/dmpbbo/examples/catkin_ws/devel/lib/test_tf2/test_static_publisher: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
/home/ksavevska/dmpbbo/examples/catkin_ws/devel/lib/test_tf2/test_static_publisher: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
/home/ksavevska/dmpbbo/examples/catkin_ws/devel/lib/test_tf2/test_static_publisher: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
/home/ksavevska/dmpbbo/examples/catkin_ws/devel/lib/test_tf2/test_static_publisher: /usr/lib/x86_64-linux-gnu/libpthread.so
/home/ksavevska/dmpbbo/examples/catkin_ws/devel/lib/test_tf2/test_static_publisher: /usr/lib/x86_64-linux-gnu/libconsole_bridge.so.0.4
/home/ksavevska/dmpbbo/examples/catkin_ws/devel/lib/test_tf2/test_static_publisher: gtest/googlemock/gtest/libgtest.so
/home/ksavevska/dmpbbo/examples/catkin_ws/devel/lib/test_tf2/test_static_publisher: /usr/lib/x86_64-linux-gnu/libboost_signals.so
/home/ksavevska/dmpbbo/examples/catkin_ws/devel/lib/test_tf2/test_static_publisher: /usr/lib/x86_64-linux-gnu/libconsole_bridge.so.0.4
/home/ksavevska/dmpbbo/examples/catkin_ws/devel/lib/test_tf2/test_static_publisher: /opt/ros/melodic/lib/libroscpp_serialization.so
/home/ksavevska/dmpbbo/examples/catkin_ws/devel/lib/test_tf2/test_static_publisher: /opt/ros/melodic/lib/librostime.so
/home/ksavevska/dmpbbo/examples/catkin_ws/devel/lib/test_tf2/test_static_publisher: /opt/ros/melodic/lib/libcpp_common.so
/home/ksavevska/dmpbbo/examples/catkin_ws/devel/lib/test_tf2/test_static_publisher: /usr/lib/x86_64-linux-gnu/libboost_system.so
/home/ksavevska/dmpbbo/examples/catkin_ws/devel/lib/test_tf2/test_static_publisher: /usr/lib/x86_64-linux-gnu/libboost_thread.so
/home/ksavevska/dmpbbo/examples/catkin_ws/devel/lib/test_tf2/test_static_publisher: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
/home/ksavevska/dmpbbo/examples/catkin_ws/devel/lib/test_tf2/test_static_publisher: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
/home/ksavevska/dmpbbo/examples/catkin_ws/devel/lib/test_tf2/test_static_publisher: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
/home/ksavevska/dmpbbo/examples/catkin_ws/devel/lib/test_tf2/test_static_publisher: /usr/lib/x86_64-linux-gnu/libpthread.so
/home/ksavevska/dmpbbo/examples/catkin_ws/devel/lib/test_tf2/test_static_publisher: /usr/lib/x86_64-linux-gnu/libconsole_bridge.so.0.4
/home/ksavevska/dmpbbo/examples/catkin_ws/devel/lib/test_tf2/test_static_publisher: geometry2/test_tf2/CMakeFiles/test_static_publisher.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ksavevska/dmpbbo/examples/catkin_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable /home/ksavevska/dmpbbo/examples/catkin_ws/devel/lib/test_tf2/test_static_publisher"
	cd /home/ksavevska/dmpbbo/examples/catkin_ws/build/geometry2/test_tf2 && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_static_publisher.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
geometry2/test_tf2/CMakeFiles/test_static_publisher.dir/build: /home/ksavevska/dmpbbo/examples/catkin_ws/devel/lib/test_tf2/test_static_publisher

.PHONY : geometry2/test_tf2/CMakeFiles/test_static_publisher.dir/build

geometry2/test_tf2/CMakeFiles/test_static_publisher.dir/clean:
	cd /home/ksavevska/dmpbbo/examples/catkin_ws/build/geometry2/test_tf2 && $(CMAKE_COMMAND) -P CMakeFiles/test_static_publisher.dir/cmake_clean.cmake
.PHONY : geometry2/test_tf2/CMakeFiles/test_static_publisher.dir/clean

geometry2/test_tf2/CMakeFiles/test_static_publisher.dir/depend:
	cd /home/ksavevska/dmpbbo/examples/catkin_ws/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ksavevska/dmpbbo/examples/catkin_ws/src /home/ksavevska/dmpbbo/examples/catkin_ws/src/geometry2/test_tf2 /home/ksavevska/dmpbbo/examples/catkin_ws/build /home/ksavevska/dmpbbo/examples/catkin_ws/build/geometry2/test_tf2 /home/ksavevska/dmpbbo/examples/catkin_ws/build/geometry2/test_tf2/CMakeFiles/test_static_publisher.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : geometry2/test_tf2/CMakeFiles/test_static_publisher.dir/depend

