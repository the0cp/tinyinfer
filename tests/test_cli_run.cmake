file(MAKE_DIRECTORY "${TEST_DIRECTORY}")
set(manifest "${TEST_DIRECTORY}/model.ti")
set(input "${TEST_DIRECTORY}/input.f32")

file(WRITE "${manifest}"
    "TINYINFER_MODEL 1\n"
    "weights weights.bin\n\n"
    "input input 2 1 3\n"
    "output output\n\n"
    "node relu ReLU 1 input output\n\n"
    "end\n"
)
file(WRITE "${TEST_DIRECTORY}/weights.bin" "")

# Three finite little-endian floats encoded without NUL bytes.
string(ASCII 97 98 99 100 101 102 103 104 105 106 107 10 input_bytes)
file(WRITE "${input}" "${input_bytes}")

execute_process(
    COMMAND "${TINYINFER_CLI}" run "${manifest}" "${input}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error
)

if(NOT result EQUAL 0)
    message(FATAL_ERROR "tinyinfer run failed: ${error}")
endif()
if(NOT output MATCHES "output_shape=1x3")
    message(FATAL_ERROR "unexpected output shape: ${output}")
endif()
if(NOT output MATCHES "predicted_index=1")
    message(FATAL_ERROR "unexpected prediction: ${output}")
endif()
