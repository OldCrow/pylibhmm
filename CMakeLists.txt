cmake_minimum_required(VERSION 3.20)
project(pylibhmm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

set(PYLIBHMM_LOCAL_LIBHMM_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/../libhmm"
    CACHE PATH
    "Path to a local libhmm source tree")

set(_pylibhmm_libhmm_target "")
set(_pylibhmm_libhmm_include_dir "")
if(EXISTS "${PYLIBHMM_LOCAL_LIBHMM_SOURCE}/CMakeLists.txt")
    message(STATUS "Using local libhmm source: ${PYLIBHMM_LOCAL_LIBHMM_SOURCE}")
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
    set(ENABLE_STATIC_ANALYSIS OFF CACHE BOOL "" FORCE)
    set(ENABLE_CLANG_TIDY OFF CACHE BOOL "" FORCE)
    set(ENABLE_CPPCHECK OFF CACHE BOOL "" FORCE)
    add_subdirectory("${PYLIBHMM_LOCAL_LIBHMM_SOURCE}"
                     "${CMAKE_BINARY_DIR}/_deps/libhmm-build"
                     EXCLUDE_FROM_ALL)
    set(_pylibhmm_libhmm_target hmm_static)
    set(_pylibhmm_libhmm_include_dir "${PYLIBHMM_LOCAL_LIBHMM_SOURCE}/include")
else()
    message(STATUS "Local libhmm source not found, using FetchContent")
    include(FetchContent)
    FetchContent_Declare(
        libhmm
        GIT_REPOSITORY https://github.com/OldCrow/libhmm.git
        GIT_TAG        v3.3.0
    )
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
    set(ENABLE_STATIC_ANALYSIS OFF CACHE BOOL "" FORCE)
    set(ENABLE_CLANG_TIDY OFF CACHE BOOL "" FORCE)
    set(ENABLE_CPPCHECK OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(libhmm)
    set(_pylibhmm_libhmm_target hmm_static)
    set(_pylibhmm_libhmm_include_dir "${libhmm_SOURCE_DIR}/include")
endif()

if(NOT TARGET ${_pylibhmm_libhmm_target})
    message(FATAL_ERROR "Unable to locate libhmm static target '${_pylibhmm_libhmm_target}'")
endif()

nanobind_add_module(
    _core
    STABLE_ABI
    NB_STATIC
    src/pylibhmm/_core.cpp
)

target_link_libraries(_core PRIVATE ${_pylibhmm_libhmm_target})
target_include_directories(_core PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/pylibhmm"
    "${_pylibhmm_libhmm_include_dir}"
)

if(WIN32)
    target_compile_definitions(_core PRIVATE NOMINMAX _USE_MATH_DEFINES)
endif()

install(TARGETS _core LIBRARY DESTINATION pylibhmm)
