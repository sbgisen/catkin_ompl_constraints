cmake_minimum_required(VERSION 2.8.3)
project(ompl_constraints)

add_compile_options(-std=c++17)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(catkin REQUIRED)
find_package(ompl REQUIRED)

catkin_package(
  LIBRARIES
    ${PROJECT_NAME}
  INCLUDE_DIRS
    include
  DEPENDS
    ompl
)

include_directories(include
                    ${catkin_INCLUDE_DIRS}
                    ${OMPL_INCLUDE_DIRS}
                    SYSTEM)

link_directories(${catkin_LIBRARY_DIRS})

# Create the PlannerManager Plugin
add_library(${PROJECT_NAME}
            src/CBiRRT2.cpp
            src/ConstrainedSpaceInformation.cpp
            src/ConstraintInformation.cpp
            )

target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES})

install (TARGETS ${PROJECT_NAME}
         LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
         RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY include/ DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})
