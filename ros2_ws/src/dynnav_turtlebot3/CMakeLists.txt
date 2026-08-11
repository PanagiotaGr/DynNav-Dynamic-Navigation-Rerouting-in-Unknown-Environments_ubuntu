cmake_minimum_required(VERSION 3.8)

project(dynnav_turtlebot3)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch config worlds
  DESTINATION share/${PROJECT_NAME}
)

ament_package()