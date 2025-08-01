# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This file is derived from                                                    #
# https://github.com/llvm/circt/blob/main/cmake/modules/AddCIRCT.cmake         #
# CIRCT is an LLVM incubator project under Apache License 2.0 with LLVM        #
# Exceptions.                                                                  #
# ============================================================================ #

include_guard()

function(add_cudaq_dialect dialect dialect_namespace)
  set(LLVM_TARGET_DEFINITIONS ${dialect}Dialect.td)
  mlir_tablegen(${dialect}Dialect.h.inc -gen-dialect-decls -dialect=${dialect_namespace})
  mlir_tablegen(${dialect}Dialect.cpp.inc -gen-dialect-defs -dialect=${dialect_namespace})
  add_public_tablegen_target(${dialect}DialectIncGen)
  set(LLVM_TARGET_DEFINITIONS ${dialect}Ops.td)
  mlir_tablegen(${dialect}Ops.h.inc -gen-op-decls)
  mlir_tablegen(${dialect}Ops.cpp.inc -gen-op-defs)
  add_public_tablegen_target(${dialect}OpsIncGen)
  set(LLVM_TARGET_DEFINITIONS ${dialect}Types.td)
  mlir_tablegen(${dialect}Types.h.inc -gen-typedef-decls -typedefs-dialect=${dialect_namespace})
  mlir_tablegen(${dialect}Types.cpp.inc -gen-typedef-defs -typedefs-dialect=${dialect_namespace})
  add_public_tablegen_target(${dialect}TypesIncGen)
  add_dependencies(cudaq-headers
    ${dialect}DialectIncGen ${dialect}OpsIncGen ${dialect}TypesIncGen)
endfunction()

function(add_cudaq_interface interface)
  set(LLVM_TARGET_DEFINITIONS ${interface}.td)
  mlir_tablegen(${interface}.h.inc -gen-op-interface-decls)
  mlir_tablegen(${interface}.cpp.inc -gen-op-interface-defs)
  add_public_tablegen_target(${interface}IncGen)
  add_dependencies(cudaq-headers ${interface}IncGen)
endfunction()

function(add_cudaq_doc tablegen_file output_path command)
  set(LLVM_TARGET_DEFINITIONS ${tablegen_file}.td)
  string(MAKE_C_IDENTIFIER ${output_path} output_id)
  tablegen(MLIR ${output_id}.md ${command} ${ARGN})
  set(GEN_DOC_FILE ${CUDAQ_BINARY_DIR}/docs/${output_path}.md)
  add_custom_command(
    OUTPUT ${GEN_DOC_FILE}
    COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_BINARY_DIR}/${output_id}.md
    ${GEN_DOC_FILE}
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${output_id}.md)
  add_custom_target(${output_id}DocGen DEPENDS ${GEN_DOC_FILE})
  add_dependencies(cudaq-doc ${output_id}DocGen)
endfunction()

function(add_cudaq_dialect_doc dialect dialect_namespace)
  add_cudaq_doc(${dialect} Dialects/${dialect} -gen-dialect-doc -dialect ${dialect_namespace})
endfunction()

function(add_cudaq_library name)
  add_mlir_library(${ARGV} DISABLE_INSTALL)
  set_target_properties(${name} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CUDAQ_LIBRARY_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY "${CUDAQ_LIBRARY_DIR}"
  )
  add_cudaq_library_install(${name})
endfunction()

macro(add_cudaq_executable name)
  add_llvm_executable(${name} ${ARGN})
  llvm_update_compile_flags(${name})
  set_target_properties(${name} PROPERTIES FOLDER "cudaq executables")
endmacro()

macro(add_cudaq_tool name)
  if (NOT CUDAQ_BUILD_TOOLS)
    set(EXCLUDE_FROM_ALL ON)
  endif()

  add_cudaq_executable(${name} ${ARGN})
  set_target_properties(${name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CUDAQ_TOOLS_DIR}"
  )
  if (CUDAQ_BUILD_TOOLS)
    get_target_export_arg(${name} CUDAQ export_to_cudaq_targets)
    install(TARGETS ${name}
      ${export_to_cudaq_targets}
      RUNTIME DESTINATION "${CUDAQ_TOOLS_INSTALL_DIR}"
      COMPONENT ${name})

    if(NOT CMAKE_CONFIGURATION_TYPES)
      add_llvm_install_targets(install-${name}
        DEPENDS ${name}
        COMPONENT ${name})
    endif()
    set_property(GLOBAL APPEND PROPERTY CUDAQ_EXPORTS ${name})
  endif()
endmacro()

# Adds a CUDA Quantum dialect library target for installation. This should normally
# only be called from add_cudaq_dialect_library().
function(add_cudaq_library_install name)
  if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
    get_target_export_arg(${name} CUDAQ export_to_cudaq_targets UMBRELLA cudaq-libraries)
    install(TARGETS ${name}
      COMPONENT ${name}
      ${export_to_cudaq_targets}
      LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX}
      ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX}
      RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
      # Note that CMake will create a directory like:
      #   objects-${CMAKE_BUILD_TYPE}/obj.LibName
      # and put object files there.
      OBJECTS DESTINATION lib${LLVM_LIBDIR_SUFFIX}
    )

    if (NOT LLVM_ENABLE_IDE)
      add_llvm_install_targets(install-${name}
                              DEPENDS ${name}
                              COMPONENT ${name})
    endif()
    set_property(GLOBAL APPEND PROPERTY CUDAQ_ALL_LIBS ${name})
  endif()   
  set_property(GLOBAL APPEND PROPERTY CUDAQ_EXPORTS ${name})
endfunction()

function(add_cudaq_dialect_library name)
  set_property(GLOBAL APPEND PROPERTY CUDAQ_DIALECT_LIBS ${name})
  add_cudaq_library(${ARGV} DEPENDS cudaq-headers)
endfunction()

function(add_cudaq_translation_library name)
  set_property(GLOBAL APPEND PROPERTY CUDAQ_TRANSLATION_LIBS ${name})
  add_cudaq_library(${ARGV} DEPENDS cudaq-headers)
endfunction()

function(add_target_config name)
  install(FILES ${name}.yml DESTINATION targets)
  configure_file(${name}.yml ${CMAKE_BINARY_DIR}/targets/${name}.yml COPYONLY)
endfunction()

function(add_target_mapping_arch providerName name)
  install(FILES ${name} DESTINATION targets/mapping/${providerName})
  configure_file(${name} ${CMAKE_BINARY_DIR}/targets/mapping/${providerName}/${name} COPYONLY)
endfunction()
