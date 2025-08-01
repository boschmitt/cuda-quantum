macro(cudaq_llvm_set_bundled_cmake_options)

  # Officially, LLVM requires C++17.
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED on)

  set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "")
  set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "")
  set(LLVM_INCLUDE_BENCHMARKS OFF CACHE BOOL "")
  set(LLVM_APPEND_VC_REV OFF CACHE BOOL "")
  set(LLVM_ENABLE_IDE ON CACHE BOOL "")
  set(LLVM_ENABLE_BINDINGS OFF CACHE BOOL "")

  # Force LLVM to avoid dependencies
  set(LLVM_ENABLE_LIBEDIT OFF CACHE BOOL "Default disable")
  set(LLVM_ENABLE_LIBXML2 OFF CACHE BOOL "Default disable")
  set(LLVM_ENABLE_TERMINFO OFF CACHE BOOL "Default disable")
  set(LLVM_ENABLE_ZLIB OFF CACHE BOOL "Default disable")
  set(LLVM_ENABLE_ZSTD OFF CACHE BOOL "Default disable")
  set(LLVM_FORCE_ENABLE_STATS ON CACHE BOOL "Default enable")

  set(LLVM_ENABLE_WARNINGS OFF)

  set(LLVM_TARGETS_TO_BUILD Native CACHE STRING "")
  set(LLVM_ENABLE_PROJECTS "clang;mlir" CACHE STRING "")
  set(LLVM_EXTERNAL_PROJECTS "" CACHE STRING "")

  set(MLIR_ENABLE_EXECUTION_ENGINE ON CACHE BOOL "")
  set(MLIR_ENABLE_BINDINGS_PYTHON OFF CACHE BOOL "")
  set(MLIR_DISABLE_CONFIGURE_PYTHON_DEV_PACKAGES ON CACHE BOOL "" FORCE)

  message(VERBOSE "Building LLVM Targets: ${LLVM_TARGETS_TO_BUILD}")
  message(VERBOSE "Building LLVM Projects: ${LLVM_ENABLE_PROJECTS}")
endmacro()

macro(cudaq_llvm_configure_bundled)
  message(STATUS "Adding bundled LLVM source dependency")
  cudaq_llvm_set_bundled_cmake_options()

  if(CUDAQ_ENABLE_PYTHON)
    set(MLIR_ENABLE_BINDINGS_PYTHON ON)
  endif()

  # Stash cmake build type in case LLVM messes with it.
  set(_CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}")

  set(LLVM_LIBRARY_OUTPUT_INTDIR "${CMAKE_CURRENT_BINARY_DIR}/llvm/lib")
  set(LLVM_RUNTIME_OUTPUT_INTDIR "${CMAKE_CURRENT_BINARY_DIR}/llvm/bin")

  message(STATUS "Configuring tpls/llvm")
  set(_BUNDLED_LLVM_ROOT "${CUDAQ_ROOT_DIR}/tpls/llvm")

  list(APPEND CMAKE_MESSAGE_INDENT "  ")
  add_subdirectory("${_BUNDLED_LLVM_ROOT}/llvm" "llvm" EXCLUDE_FROM_ALL)

  list(POP_BACK CMAKE_MESSAGE_INDENT)

  # Reset CMAKE_BUILD_TYPE to its previous setting.
  set(CMAKE_BUILD_TYPE "${_CMAKE_BUILD_TYPE}" )

  set(LLVM_CMAKE_DIR "${CUDAQ_BINARY_DIR}/llvm/lib/cmake/llvm")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

  set(MLIR_CMAKE_DIR "${CMAKE_BINARY_DIR}/lib/cmake/mlir")
  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

  set(LLVM_INCLUDE_DIRS
    ${_BUNDLED_LLVM_ROOT}/llvm/include
    ${CUDAQ_BINARY_DIR}/llvm/include
  )
  set(CLANG_INCLUDE_DIRS
    ${_BUNDLED_LLVM_ROOT}/clang/include
    ${CUDAQ_BINARY_DIR}/llvm/tools/clang/include
  )
  set(MLIR_INCLUDE_DIRS
    ${_BUNDLED_LLVM_ROOT}/mlir/include
    ${CUDAQ_BINARY_DIR}/llvm/tools/mlir/include
  )
  set(LLD_INCLUDE_DIRS
    ${_BUNDLED_LLVM_ROOT}/lld/include
    ${CUDAQ_BINARY_DIR}/llvm/tools/lld/include
  )

  set(LLVM_BINARY_DIR "${CUDAQ_BINARY_DIR}/llvm")
  set(LLVM_TOOLS_BINARY_DIR "${LLVM_BINARY_DIR}/bin")
  set(LLVM_EXTERNAL_LIT "${_BUNDLED_LLVM_ROOT}/llvm/utils/lit/lit.py")

  # Need this for nanobind, MLIR things uses PyClassMethod_New which is a
  # missing symbol
  if(APPLE)
    # Darwin-specific linker flags for loadable modules.
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -Wl,-flat_namespace -Wl,-undefined -Wl,dynamic_lookup")
  endif()
endmacro()