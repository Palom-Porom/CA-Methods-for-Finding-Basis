cmake_minimum_required(VERSION 3.14)
project(gmres_solvers CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Option: link against BLAS
# ---------------------------------------------------------------------------
option(USE_BLAS "Use CBLAS backend instead of STL" OFF)

# ---------------------------------------------------------------------------
# Executable
# ---------------------------------------------------------------------------
add_executable(gmres_test gmres_test.cpp)

target_include_directories(gmres_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Optimisation flags
if(MSVC)
    target_compile_options(gmres_test PRIVATE /O2 /W4)
else()
    target_compile_options(gmres_test PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()

# ---------------------------------------------------------------------------
# BLAS backend
# ---------------------------------------------------------------------------
if(USE_BLAS)
    target_compile_definitions(gmres_test PRIVATE USE_BLAS)

    # Try to find BLAS via CMake's built-in module first
    find_package(BLAS QUIET)

    if(BLAS_FOUND)
        message(STATUS "Found BLAS: ${BLAS_LIBRARIES}")
        target_link_libraries(gmres_test PRIVATE ${BLAS_LIBRARIES})

        # Locate cblas.h (not guaranteed by FindBLAS)
        find_path(CBLAS_INCLUDE_DIR cblas.h
            PATHS
                /usr/include
                /usr/local/include
                /opt/homebrew/include
                /opt/homebrew/opt/openblas/include
                /usr/include/openblas
                $ENV{MKLROOT}/include
            DOC "Directory containing cblas.h"
        )
        if(CBLAS_INCLUDE_DIR)
            target_include_directories(gmres_test PRIVATE ${CBLAS_INCLUDE_DIR})
        else()
            message(WARNING
                "cblas.h not found automatically. "
                "Set CBLAS_INCLUDE_DIR manually: "
                "cmake -DCBLAS_INCLUDE_DIR=/path/to/blas ...")
        endif()

    else()
        # Manual fallback: user supplies BLAS_LIBRARIES and CBLAS_INCLUDE_DIR
        message(STATUS
            "BLAS not found automatically. "
            "Set BLAS_LIBRARIES and CBLAS_INCLUDE_DIR manually, e.g.:\n"
            "  cmake -DBLAS_LIBRARIES=/usr/lib/libopenblas.so "
            "-DCBLAS_INCLUDE_DIR=/usr/include/openblas ...")

        if(BLAS_LIBRARIES)
            target_link_libraries(gmres_test PRIVATE ${BLAS_LIBRARIES})
        endif()
        if(CBLAS_INCLUDE_DIR)
            target_include_directories(gmres_test PRIVATE ${CBLAS_INCLUDE_DIR})
        endif()
    endif()

    message(STATUS "Building with BLAS backend (USE_BLAS=ON)")
else()
    message(STATUS "Building with STL backend (USE_BLAS=OFF)")
endif()

# ---------------------------------------------------------------------------
# CTest integration (optional: run via `ctest`)
# ---------------------------------------------------------------------------
enable_testing()
add_test(NAME gmres_correctness COMMAND gmres_test)
