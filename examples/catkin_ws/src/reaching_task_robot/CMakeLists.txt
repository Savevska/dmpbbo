# add_executable(robotExecuteTrajectory robotExecuteTrajectory.cpp runSimulationThrowBall.cpp)
# target_link_libraries(robotExecuteTrajectory dmp functionapproximators dynamicalsystems ${Boost_LIBRARIES})
# install(TARGETS robotExecuteTrajectory DESTINATION bin)

include_directories(/opt/ros/melodic/include /opt/ros/melodic/lib)
include_directories(/opt/pal/ferrum/include /opt/pal/ferrum/lib)
include_directories(${CMAKE_SOURCE_DIR}/examples/reaching_task)


add_executable(robotExecuteDmpReaching robotExecuteDmpReaching.cpp)
target_link_libraries(robotExecuteDmpReaching dmp functionapproximators dynamicalsystems ${Boost_LIBRARIES} 
                      -I/opt/ros/melodic/include -L/opt/ros/melodic/lib
                      -I/opt/pal/ferrum/include -L/opt/pal/ferrum/lib
                      -lroscpp -lrostime -ltf2_ros -lrosconsole -lroscpp_serialization)
install(TARGETS robotExecuteDmpReaching DESTINATION bin)
#install(TARGETS robotExecuteDmp DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})
