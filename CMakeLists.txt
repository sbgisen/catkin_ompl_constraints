cmake_minimum_required(VERSION 2.8.3)
project(ompl_constraints)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Boost REQUIRED)
find_package(catkin REQUIRED COMPONENTS ompl_source)

catkin_package(
  LIBRARIES
    ${PROJECT_NAME}
  INCLUDE_DIRS
    include
  CATKIN_DEPENDS
    ompl_source
)

include_directories(include
                    ${catkin_INCLUDE_DIRS}
                    ${Boost_INCLUDE_DIRS}
                    SYSTEM)

link_directories(${catkin_LIBRARY_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

# Create the PlannerManager Plugin
add_library(${PROJECT_NAME}
            src/CBiRRT2.cpp
            src/ConstrainedSpaceInformation.cpp
            src/ConstraintInformation.cpp
            )

target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${Boost_LIBRARIES})

install (TARGETS ${PROJECT_NAME}
         LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
         RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY include/ DESTINATION include)

