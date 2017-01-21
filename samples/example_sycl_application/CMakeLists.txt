set(EXAMPLE_NAME "example_sycl_application")
include_directories(${COMPUTECPP_INCLUDE_DIRECTORY})
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/example_sycl_application.cpp ${CMAKE_CURRENT_SOURCE_DIR}/second_source_file.cpp)

add_executable(${EXAMPLE_NAME} ${SOURCE_FILES})
target_compile_options(${EXAMPLE_NAME} PUBLIC ${HOST_COMPILER_OPTIONS})

add_sycl_to_target(${EXAMPLE_NAME} ${CMAKE_CURRENT_BINARY_DIR} ${SOURCE_FILES})

add_test(NAME ${EXAMPLE_NAME} COMMAND ${EXAMPLE_NAME})
