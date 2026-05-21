# wormuse-sim — top-level CMakeLists.txt (Phase 0 stub).
#
# Phase 0: just declares the project, requires the language standard,
#          and reminds the developer how to activate the MK toolchain.
#
# Subsequent phases will add:
#   - find_package(deal.II)  (Phase 5)
#   - find_package(OpenMP)   (Phase 2)
#   - find_package(MPI)      (Phase 5+, optional)
#   - LIS library            (Phase 5+, via the MK module)

cmake_minimum_required(VERSION 3.16)
project(wormuse_sim LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Activation reminder — must be done OUTSIDE cmake, in the shell:
#
#   source /u/sw/etc/profile.d/mk.sh
#   module load gcc-glibc/11.2.0 eigen lis dealii
#
# Then:  mkdir build && cd build && cmake .. && make -j

# Compile options applied to every wormuse-sim target.
add_library(wormuse_compile_opts INTERFACE)
target_compile_options(wormuse_compile_opts INTERFACE
    -Wall -Wextra -Wpedantic
)

# Phase 0 placeholder: just announce we're alive.
add_custom_target(wormuse_phase0_check
    COMMAND ${CMAKE_COMMAND} -E echo "wormuse-sim Phase 0 — scaffolding only, no targets yet"
)

# Test infrastructure (CTest) — set up early so verification CI sees it.
include(CTest)

# Subsequent phases:
#   add_subdirectory(src/ow_bridge)        # Phase 1
#   add_subdirectory(src/piano_sim)        # Phase 2 (string_1d) + Phase 5 (soundboard_fem)
#   add_subdirectory(src/neuron_to_midi)   # Phase 2
#   add_subdirectory(src/orchestrator)     # Phase 6
#   add_subdirectory(apps)                 # Phase 6
#   add_subdirectory(tests)                # Phase 2+
