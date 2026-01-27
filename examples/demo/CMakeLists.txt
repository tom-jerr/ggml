#
# demo

set(TEST_TARGET demo-backend)
add_executable(${TEST_TARGET} demo-backend.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)

set(TEST_TARGET demo-train)
add_executable(${TEST_TARGET} demo-train.cpp model.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)

set(TEST_TARGET demo-eval)
add_executable(${TEST_TARGET} demo-eval.cpp model.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)


if (GGML_CUDA)
    add_compile_definitions(GGML_USE_CUDA)
endif()

if (GGML_METAL)
    add_compile_definitions(GGML_USE_METAL)
endif()
