cmake_minimum_required(VERSION 3.16)
project(adaptive_cs_native LANGUAGES CXX)

add_library(adaptive_cs_native SHARED native/adaptive_cs.cpp)
target_compile_features(adaptive_cs_native PRIVATE cxx_std_17)
target_compile_definitions(adaptive_cs_native PRIVATE ADAPTIVE_CS_BUILD=1)
install(TARGETS adaptive_cs_native LIBRARY DESTINATION adaptive_cs RUNTIME DESTINATION adaptive_cs)

# The native solver exposes three modes: strict bisection, GSL Brent--Dekker,
# and Boost.Math TOMS 748.  Boost.Math is header-only; GSL supplies the
# bracket-preserving Brent implementation and is linked here as a normal C
# library.  Keeping the dependencies explicit makes the selected algorithms
# reproducible instead of silently falling back to a private implementation.
find_package(Boost REQUIRED)
find_package(GSL REQUIRED)
target_include_directories(adaptive_cs_native PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(adaptive_cs_native PRIVATE GSL::gsl GSL::gslcblas)
target_compile_definitions(
    adaptive_cs_native PRIVATE
    ADAPTIVE_CS_HAS_BOOST=1
    ADAPTIVE_CS_HAS_GSL=1
)

if(MSVC)
    target_compile_options(adaptive_cs_native PRIVATE /O2)
else()
    target_compile_options(adaptive_cs_native PRIVATE -O3 -DNDEBUG)
endif()

# OpenMP is optional.  The code remains correct, and still fast, without it.
option(ADAPTIVE_CS_ENABLE_OPENMP "Parallelize independent class calculations" ON)
if(ADAPTIVE_CS_ENABLE_OPENMP)
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(adaptive_cs_native PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(adaptive_cs_native PRIVATE ADAPTIVE_CS_OPENMP=1)
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            # Some MinGW CMake installations expose the OpenMP compile flag
            # but omit it from the imported target's link interface.
            target_link_options(adaptive_cs_native PRIVATE -fopenmp)
        endif()
    endif()
endif()

set_target_properties(adaptive_cs_native PROPERTIES
    OUTPUT_NAME "adaptive_cs_native"
)

# Keep the package artifact name independent of the Windows compiler.  The
# Python loader intentionally supports only this canonical build directory.
if(WIN32)
    set_target_properties(adaptive_cs_native PROPERTIES PREFIX "")
endif()
