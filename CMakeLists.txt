cmake_minimum_required(VERSION 3.25)
project(pylibhmm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# SKBUILD_SABI_COMPONENT is set to "Development.SABIModule" by scikit-build-core
# when wheel.py-api requests a Stable ABI build, and to "" otherwise — so this is
# inert for ordinary builds and for direct CMake invocation, where it is
# undefined and expands to nothing. It is NOT optional once py-api is set:
# nanobind gates its STABLE_ABI flag on `TARGET Python::SABIModule` and silently
# falls back to a version-specific build if the component was never requested,
# producing an abi3-TAGGED wheel around a version-LOCKED binary. See
# pyproject.toml. nanobind_add_module below has carried STABLE_ABI since this
# repo began, doing nothing at all, because that key was never set.
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT})
find_package(nanobind CONFIG REQUIRED)

set(PYLIBHMM_LOCAL_LIBHMM_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/../libhmm"
    CACHE PATH
    "Path to a local libhmm source tree")

set(_pylibhmm_libhmm_target "")
set(_pylibhmm_libhmm_include_dir "")
if(EXISTS "${PYLIBHMM_LOCAL_LIBHMM_SOURCE}/CMakeLists.txt")
    message(STATUS "Using local libhmm source: ${PYLIBHMM_LOCAL_LIBHMM_SOURCE}")
    # libhmm's LIBHMM_BUILD_EXAMPLES/LIBHMM_BUILD_TESTS/LIBHMM_BUILD_TOOLS default
    # to PROJECT_IS_TOP_LEVEL as of its target-first refactor, so add_subdirectory
    # here already gets library-only by default — no forced-OFF lines needed.
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
        GIT_TAG        v4.4.0
    )
    # The seven forced unprefixed option sets that used to live here are gone,
    # as their own comment instructed once GIT_TAG moved past the rename.
    # v4.3.0 both retired the unprefixed spellings entirely (they are ignored
    # now, so forcing them did nothing) and defaults its component toggles off
    # PROJECT_IS_TOP_LEVEL — which is FALSE under FetchContent, so libhmm's
    # examples, tests and tools are already off without being told. This branch
    # now behaves the same way the local-source branch above always did.
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

if(MSVC)
    # Docstrings in _core.cpp use Unicode math symbols (e.g. γ, −) via
    # universal character names; without /utf-8, MSVC narrows them to the
    # current code page (cp1252) and warns C4566 for symbols outside that set.
    target_compile_options(_core PRIVATE /utf-8)
endif()

install(TARGETS _core LIBRARY DESTINATION pylibhmm)
