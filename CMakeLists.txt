cmake_minimum_required(VERSION 3.8)
project(clipseg_ros)

find_package(ament_cmake_auto REQUIRED)

ament_auto_find_build_dependencies()

add_custom_target(${PROJECT_NAME}_venv ALL
  COMMAND UV_PROJECT_ENVIRONMENT=${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}/.venv uv sync
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

install(
  PROGRAMS ${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}/.venv/bin/segmentation_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_auto_package(
  INSTALL_TO_SHARE
    config
)
