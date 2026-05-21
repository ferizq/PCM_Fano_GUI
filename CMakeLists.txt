cmake_minimum_required(VERSION 3.18)
project(peakfit_core LANGUAGES C)

add_library(peakfit_core SHARED csrc/peakfit_core.c)
target_compile_definitions(peakfit_core PRIVATE PF_CORE_BUILD)
target_include_directories(peakfit_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/csrc)

if(MSVC)
    target_compile_options(peakfit_core PRIVATE /O2 /W4)
else()
    target_compile_options(peakfit_core PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()

set_target_properties(peakfit_core PROPERTIES OUTPUT_NAME "peakfit_core")
