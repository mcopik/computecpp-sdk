set(SOURCE_NAME "opencl_c_interop")

include_directories(${COMPUTECPP_INCLUDE_DIRECTORY})

add_executable(${SOURCE_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_NAME}.cpp  )
target_compile_options(${SOURCE_NAME} PUBLIC ${HOST_COMPILER_OPTIONS})

add_sycl_to_target(${SOURCE_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_NAME}.cpp 
                    ${CMAKE_CURRENT_BINARY_DIR})

add_test(NAME ${SOURCE_NAME} COMMAND ${SOURCE_NAME})
