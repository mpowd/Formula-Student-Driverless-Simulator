cmake_minimum_required(VERSION 3.0.2)
project(livox_msgs)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  message_runtime
  std_msgs
)

add_message_files(
  FILES
  Custom.msg
  CustomPoint.msg
)

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS 
  message_runtime 
  std_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)