file(GLOB SOURCES "*.cpp")

include_directories(${CMAKE_SOURCE_DIR}/include)

add_library(lib_fully_connected_nn SHARED ${SOURCES})
set_target_properties(lib_fully_connected_nn PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/lib"
    LIBRARY_OUTPUT_NAME "_fully_connected_nn"
    SUFFIX ".dll")