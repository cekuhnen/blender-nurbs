# Builds the USD BRep (SMLib) kernel + its Python bindings as ONE extension
# module for Blender's bundled Python 3.13, without OpenUSD. macOS, Windows, Linux.
#
# It needs a CPython 3.13 with its headers -- the version Blender 5.x runs; the
# module then loads into Blender's own 3.13. (Blender's bundled Python has no
# Python.h, only pyconfig.h.) Without -DPython3_EXECUTABLE a python3.13 on the
# PATH is used.
#
#   macOS    cmake -S kernel_build -B kernel_build/build -DCMAKE_BUILD_TYPE=Release \
#                  -DPython3_EXECUTABLE=/opt/homebrew/bin/python3.13          (brew install python@3.13)
#            (universal arm64 + Intel: add -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64")
#   Linux    cmake -S kernel_build -B kernel_build/build -DCMAKE_BUILD_TYPE=Release \
#                  -DPython3_EXECUTABLE=/usr/bin/python3.13                   (python3.13-dev)
#   Windows  cmake -S kernel_build -B kernel_build/build -A x64 -DPython3_EXECUTABLE=<python.org 3.13>\python.exe
#            cmake --build kernel_build/build --config Release
#            (links python313.lib from that install's libs\ folder)
#
#   cmake --build kernel_build/build -j
#
# Output, copied into the add-on's kernel/ folder (Python picks the one for its platform):
#   _nproto_kernel.cpython-313-darwin.so            macOS
#   _nproto_kernel.cp313-win_amd64.pyd              Windows x64
#   _nproto_kernel.cpython-313-x86_64-linux-gnu.so  Linux x64

cmake_minimum_required(VERSION 3.20)
project(blender_nurbs_kernel CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(APPLE AND NOT CMAKE_OSX_ARCHITECTURES)
  set(CMAKE_OSX_ARCHITECTURES arm64)
endif()
if(MSVC)
  # static C runtime: the module needs no Visual C++ redistributable next to
  # Blender's python.exe (the worker process runs that, not blender.exe)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

set(ROOT ${CMAKE_CURRENT_LIST_DIR}/..)
set(USD_BREP ${ROOT}/third_party/usd-brep/source CACHE PATH "usd-brep/source")
set(PYBIND11_INCLUDE ${ROOT}/third_party/pybind11_pkg/pybind11/include CACHE PATH "pybind11 headers")
set(ADDON_KERNEL_DIR ${ROOT}/blender_nurbs/kernel CACHE PATH "Where to copy the built module")

# --- Python 3.13 (Blender 5.x): headers, extension suffix, and on Windows the
#     import library. BLENDER_PYTHON_INCLUDE / PY_SUFFIX set by hand still win
#     (older build trees configured them for Homebrew's Python on macOS).
set(BLENDER_PYTHON_INCLUDE "" CACHE PATH "Python 3.13 include dir (empty: from the Python found)")
set(PY_SUFFIX "" CACHE STRING "Extension suffix (empty: the Python's EXT_SUFFIX)")
set(PY_IMPLIB "" CACHE FILEPATH "Windows: python313.lib (empty: from the Python found)")
if(WIN32 OR NOT BLENDER_PYTHON_INCLUDE OR NOT PY_SUFFIX)
  if(NOT Python3_EXECUTABLE)
    find_package(Python3 3.13 EXACT REQUIRED COMPONENTS Interpreter)
  endif()
  # asked of the interpreter itself: CMake's FindPython does not find the
  # headers of Blender's bundled Python (its build-time config points elsewhere)
  execute_process(COMMAND ${Python3_EXECUTABLE} -c
    "import os, sys, sysconfig; v = sys.version_info; print(sysconfig.get_paths()['include']); print(sysconfig.get_config_var('EXT_SUFFIX')); print(os.path.join(sys.base_prefix, 'libs', 'python%d%d.lib' % v[:2])); print('%d.%d' % v[:2])"
    OUTPUT_VARIABLE _py OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
  string(REPLACE "\\" "/" _py "${_py}")
  string(REPLACE "\n" ";" _py "${_py}")
  list(GET _py 0 _py_inc)
  list(GET _py 1 _py_suffix)
  list(GET _py 2 _py_lib)
  list(GET _py 3 _py_ver)
  if(NOT _py_ver STREQUAL "3.13")
    message(WARNING "Python ${_py_ver} at ${Python3_EXECUTABLE}: Blender 5.x runs Python 3.13, the module would not load there")
  endif()
  if(NOT BLENDER_PYTHON_INCLUDE)
    set(BLENDER_PYTHON_INCLUDE ${_py_inc})
  endif()
  if(NOT PY_SUFFIX)
    set(PY_SUFFIX ${_py_suffix})
  endif()
  if(WIN32 AND NOT PY_IMPLIB)
    set(PY_IMPLIB ${_py_lib})
  endif()
endif()
if(NOT EXISTS ${BLENDER_PYTHON_INCLUDE}/Python.h)
  message(FATAL_ERROR "No Python.h in ${BLENDER_PYTHON_INCLUDE}: point -DPython3_EXECUTABLE at a Python 3.13 with its headers")
endif()
if(WIN32 AND NOT EXISTS ${PY_IMPLIB})
  message(FATAL_ERROR "No ${PY_IMPLIB}: a python.org Python 3.13 install has it (libs\\python313.lib); pass -DPython3_EXECUTABLE=<that python.exe>")
endif()
message(STATUS "Kernel module: _nproto_kernel${PY_SUFFIX}  (Python headers: ${BLENDER_PYTHON_INCLUDE})")

# --- compiler flags (the kernel's own warnings are not ours to fix: silenced)
if(MSVC)
  set(KERNEL_OPTS /O2 /W0 /bigobj /utf-8 /EHsc)
  set(MODULE_OPTS ${KERNEL_OPTS})
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS _SCL_SECURE_NO_WARNINGS)
  set(PYBIND_TAG "_msvc_nproto")
else()
  set(KERNEL_OPTS -O2 -w)
  set(MODULE_OPTS -O2 -w -fvisibility=hidden)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(PYBIND_TAG "_gcc_nproto")
  else()
    set(PYBIND_TAG "_clang_nproto")
  endif()
  if(NOT APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    # as upstream: no fused multiply-add on Linux aarch64, so results match x86-64
    list(APPEND KERNEL_OPTS -ffp-contract=off)
    list(APPEND MODULE_OPTS -ffp-contract=off)
  endif()
endif()

file(GLOB SMLIB_SRC  ${USD_BREP}/SMLib/src/*.cpp)
file(GLOB SMAPI_SRC  ${USD_BREP}/SM_API/src/*.cpp)
file(GLOB SMPY_SRC   ${USD_BREP}/SmPyLib/src/*.cpp)
list(FILTER SMLIB_SRC EXCLUDE REGEX ".*/StdAfx\\.cpp$")
list(FILTER SMAPI_SRC EXCLUDE REGEX ".*/StdAfx\\.cpp$")
list(FILTER SMPY_SRC  EXCLUDE REGEX ".*/SmPyUsd\\.cpp$")   # replaced by our extension

# Kernel as a static lib so the module is self-contained.
# (SMLib and SM_API each have their own StdAfx.h, so include order differs.)
add_library(smlib_core STATIC ${SMLIB_SRC})
target_include_directories(smlib_core PUBLIC ${USD_BREP}/SMLib/inc)
target_compile_definitions(smlib_core PUBLIC NDEBUG NOMINMAX)
target_compile_options(smlib_core PRIVATE ${KERNEL_OPTS})

add_library(smlib_static STATIC ${SMAPI_SRC})
target_include_directories(smlib_static BEFORE PUBLIC ${USD_BREP}/SM_API/inc)
target_compile_definitions(smlib_static PUBLIC SM_API_EXPORTS)
target_compile_options(smlib_static PRIVATE ${KERNEL_OPTS})
target_link_libraries(smlib_static PUBLIC smlib_core)

add_library(_omni_solid MODULE ${SMPY_SRC} ${CMAKE_CURRENT_LIST_DIR}/ext/SmPyBlenderExt.cpp)
target_include_directories(_omni_solid PRIVATE
  ${USD_BREP}/SmPyLib/src ${PYBIND11_INCLUDE} ${BLENDER_PYTHON_INCLUDE})
target_link_libraries(_omni_solid PRIVATE smlib_static)
target_compile_options(_omni_solid PRIVATE ${MODULE_OPTS})
# Own module name + own pybind11 internals, so this kernel can share a Blender
# process with other add-ons that ship their own _omni_solid / smlibpy build.
target_compile_definitions(_omni_solid PRIVATE
  _omni_solid=_nproto_kernel
  PYBIND11_COMPILER_TYPE="${PYBIND_TAG}")
set_target_properties(_omni_solid PROPERTIES OUTPUT_NAME _nproto_kernel PREFIX "" SUFFIX "${PY_SUFFIX}")
if(APPLE)
  target_link_options(_omni_solid PRIVATE -undefined dynamic_lookup)
elseif(WIN32)
  target_link_libraries(_omni_solid PRIVATE ${PY_IMPLIB})          # python313.lib
endif()
# (Linux: Python's symbols come from the running interpreter, nothing to link)

# Copy under a temporary name, then rename over the old module: a running
# Blender keeps the module it loaded (writing into a loaded library in place
# can crash it on macOS); the new one is used after a restart. On Windows a
# loaded .pyd cannot be replaced at all: close Blender before building.
add_custom_command(TARGET _omni_solid POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${ADDON_KERNEL_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:_omni_solid> ${ADDON_KERNEL_DIR}/$<TARGET_FILE_NAME:_omni_solid>.new
  COMMAND ${CMAKE_COMMAND} -E rename ${ADDON_KERNEL_DIR}/$<TARGET_FILE_NAME:_omni_solid>.new ${ADDON_KERNEL_DIR}/$<TARGET_FILE_NAME:_omni_solid>)
