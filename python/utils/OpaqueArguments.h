/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "PyTypes.h"
#include "common/FmtCore.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/CodeGen/QIRFunctionNames.h"
#include "cudaq/Optimizer/CodeGen/QIROpaqueStructTypes.h"
#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "cudaq/builder/kernel_builder.h"
#include "cudaq/qis/pauli_word.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/LLVMContext.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Target/LLVMIR/TypeToLLVM.h"
#include <chrono>
#include <complex>
#include <functional>
#include <future>
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/complex.h>
#include <vector>

namespace nb = nanobind;
using namespace std::chrono_literals;
using namespace mlir;

namespace cudaq {

/// @brief The OpaqueArguments type wraps a vector
/// of function arguments represented as opaque
/// pointers. For each element in the vector of opaque
/// pointers, we also track the arguments corresponding
/// deletion function - a function invoked upon destruction
/// of this OpaqueArguments to clean up the memory.
class OpaqueArguments {
public:
  using OpaqueArgDeleter = std::function<void(void *)>;

  const std::vector<void *> &getArgs() const { return args; }

private:
  /// @brief The opaque argument pointers
  std::vector<void *> args;

  /// @brief Deletion functions for the arguments.
  std::vector<OpaqueArgDeleter> deleters;

public:
  /// @brief Add an opaque argument and its `deleter` to this OpaqueArguments
  template <typename ArgPointer, typename Deleter>
  void emplace_back(ArgPointer &&pointer, Deleter &&deleter) {
    args.emplace_back(pointer);
    deleters.emplace_back(deleter);
  }

  /// @brief Return the `args` as a pointer to void*.
  void **data() { return args.data(); }

  /// @brief Return the number of arguments
  std::size_t size() { return args.size(); }

  /// Destructor, clean up the memory
  ~OpaqueArguments() {
    for (std::size_t counter = 0; auto &ptr : args)
      deleters[counter++](ptr);

    args.clear();
    deleters.clear();
  }
};

/// @brief This function modifies input arguments to convert them into valid
/// CUDA-Q argument types. Future work should make this function perform more
/// checks, we probably want to take the Kernel MLIR argument Types as input and
/// use that to validate that the passed arguments are good to go.
inline nb::args simplifiedValidateInputArguments(const nb::args &args) {
    std::vector<nb::object> processed;
    processed.reserve(args.size());
    
    for (size_t i = 0; i < args.size(); ++i) {
        nb::object arg = args[i];
        
        if (nb::hasattr(arg, "tolist")) {
            if (!nb::hasattr(arg, "shape"))
                throw std::runtime_error(
                    "Invalid input argument type, could not get shape of array.");
            auto shape = nb::cast<nb::tuple>(arg.attr("shape"));
            if (shape.size() != 1)
                throw std::runtime_error("Cannot pass ndarray with shape != (N,).");
            arg = arg.attr("tolist")();
//        } else if (nb::isinstance<nb::str>(arg)) {
//            arg = nb::cast<std::string>(arg);
        } else if (nb::isinstance<nb::list>(arg)) {
            nb::list arg_list = nb::cast<nb::list>(arg);
            bool all_strings = true;
            for (nb::handle item : arg_list) {
                if (!nb::isinstance<nb::str>(item)) {
                    all_strings = false;
                    break;
                }
            }
            if (all_strings) {
                std::vector<cudaq::pauli_word> pw_list;
                pw_list.reserve(arg_list.size());
                for (nb::handle item : arg_list)
                    pw_list.emplace_back(nb::cast<std::string>(item));
                arg = nb::cast(pw_list);
            }
        }
        
        processed.push_back(std::move(arg));
    }
    
    nb::list temp_list;
    for (const auto& obj : processed) {
        temp_list.append(obj);
    }
    return nb::args(temp_list, nb::detail::steal_t());
}

/// @brief Search the given Module for the function with provided name.
inline mlir::func::FuncOp getKernelFuncOp(MlirModule module,
                                          const std::string &kernelName) {
  using namespace mlir;
  ModuleOp mod = unwrap(module);
  func::FuncOp kernelFunc;
  mod.walk([&](func::FuncOp function) {
    if (function.getName() == ("__nvqpp__mlirgen__" + kernelName)) {
      kernelFunc = function;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (!kernelFunc)
    throw std::runtime_error("Could not find " + kernelName +
                             " function in current module.");

  return kernelFunc;
}

template <typename T>
void checkArgumentType(nb::handle arg, int index) {
  if (!py_ext::isConvertible<T>(arg)) {
    throw std::runtime_error(
        "kernel argument type is '" + std::string(py_ext::typeName<T>()) + "'" +
        " but argument provided is not (argument " + std::to_string(index) +
        ", value=" + nb::cast<std::string>(nb::str(arg)) +
        ", type=" + nb::cast<std::string>(nb::str(nb::cast<nb::object>(arg).type())) + ").");
  }
}

template <typename T>
void checkListElementType(nb::handle arg, int index) {
  if (!py_ext::isConvertible<T>(arg)) {
    throw std::runtime_error(
        "kernel argument's element type is '" +
        std::string(py_ext::typeName<T>()) + "'" +
        " but argument provided is not (argument " + std::to_string(index) +
        ", value=" + nb::cast<std::string>(nb::str(arg)) +
        ", type=" + nb::cast<std::string>(nb::str(nb::cast<nb::object>(arg).type())) + ").");
  }
}

template <typename T>
inline void addArgument(OpaqueArguments &argData, T &&arg) {
  T *allocatedArg = new T(std::move(arg));
  argData.emplace_back(allocatedArg,
                       [](void *ptr) { delete static_cast<T *>(ptr); });
}

template <typename T>
inline void valueArgument(OpaqueArguments &argData, T *arg) {
  argData.emplace_back(static_cast<void *>(arg), [](void *) {});
}

inline std::string mlirTypeToString(mlir::Type ty) {
  std::string msg;
  {
    llvm::raw_string_ostream os(msg);
    ty.print(os);
  }
  return msg;
}

/// @brief Return the size and member variable offsets for the input struct.
inline std::pair<std::size_t, std::vector<std::size_t>>
getTargetLayout(func::FuncOp func, cudaq::cc::StructType structTy) {
  auto mod = func->getParentOfType<ModuleOp>();
  StringRef dataLayoutSpec = "";
  if (auto attr = mod->getAttr(cudaq::opt::factory::targetDataLayoutAttrName))
    dataLayoutSpec = cast<StringAttr>(attr);
  else
    throw std::runtime_error("No data layout attribute is set on the module.");

  auto dataLayout = llvm::DataLayout(dataLayoutSpec);
  // Convert bufferTy to llvm.
  llvm::LLVMContext context;
  LLVMTypeConverter converter(func.getContext());
  cudaq::opt::initializeTypeConversions(converter);
  auto llvmDialectTy = converter.convertType(structTy);
  LLVM::TypeToLLVMIRTranslator translator(context);
  auto *llvmStructTy =
      cast<llvm::StructType>(translator.translateType(llvmDialectTy));
  auto *layout = dataLayout.getStructLayout(llvmStructTy);
  auto strSize = layout->getSizeInBytes();
  std::vector<std::size_t> fieldOffsets;
  for (std::size_t i = 0, I = structTy.getMembers().size(); i != I; ++i)
    fieldOffsets.emplace_back(layout->getElementOffset(i));
  return {strSize, fieldOffsets};
}

/// @brief For the current struct member variable type, insert the
/// value into the dynamically-constructed struct.
inline void handleStructMemberVariable(void *data, std::size_t offset,
                                       Type memberType, nb::object value) {
  auto appendValue = [](void *data, auto &&value, std::size_t offset) {
    std::memcpy(((char *)data) + offset, &value,
                sizeof(std::remove_cvref_t<decltype(value)>));
  };
  llvm::TypeSwitch<Type, void>(memberType)
      .Case([&](IntegerType ty) {
        if (ty.isInteger(1)) {
          appendValue(data, (bool)nb::cast<nb::bool_>(value), offset);
          return;
        }
        appendValue(data, (std::int64_t)nb::cast<nb::int_>(value), offset);
      })
      .Case([&](mlir::Float64Type ty) {
        appendValue(data, (double)nb::cast<nb::float_>(value), offset);
      })
      .Case([&](cudaq::cc::StdvecType ty) {
        auto appendVectorValue = []<typename T>(nb::object value, void *data,
                                                std::size_t offset, T) {
          auto asList = nb::cast<nb::list>(value);
          std::vector<double> *values = new std::vector<double>(asList.size());
          for (std::size_t i = 0; nb::handle v : asList)
            (*values)[i++] = nb::cast<double>(v);

          std::memcpy(((char *)data) + offset, values, 16);
        };

        TypeSwitch<Type, void>(ty.getElementType())
            .Case([&](IntegerType type) {
              if (type.isInteger(1)) {
                appendVectorValue(value, data, offset, bool());
                return;
              }

              appendVectorValue(value, data, offset, std::size_t());
              return;
            })
            .Case([&](FloatType type) {
              if (type.isF32()) {
                appendVectorValue(value, data, offset, float());
                return;
              }

              appendVectorValue(value, data, offset, double());
              return;
            });
      })
      .Default([&](Type ty) {
        ty.dump();
        throw std::runtime_error(
            "Type not supported for custom struct in kernel.");
      });
}

/// @brief For the current vector element type, insert the
/// value into the dynamically-constructed vector.
inline void *handleVectorElements(Type eleTy, nb::list list) {
  auto appendValue = []<typename T>(nb::list list, auto &&converter) -> void * {
    std::vector<T> *values = new std::vector<T>(list.size());
    for (std::size_t i = 0; auto v : list) {
      auto converted = converter(v, i);
      (*values)[i++] = converted;
    }
    return values;
  };

  return llvm::TypeSwitch<Type, void *>(eleTy)
      .Case([&](IntegerType ty) {
        if (ty.getIntOrFloatBitWidth() == 1)
          return appendValue.template operator()<bool>(
              list, [](nb::handle v, std::size_t i) {
                checkListElementType<nb::bool_>(v, i);
                return nb::cast<bool>(v);
              });
        if (ty.getIntOrFloatBitWidth() == 8)
          return appendValue.template operator()<std::int8_t>(
              list, [](nb::handle v, std::size_t i) {
                checkListElementType<py_ext::Int>(v, i);
                return nb::cast<std::int8_t>(v);
              });
        if (ty.getIntOrFloatBitWidth() == 16)
          return appendValue.template operator()<std::int16_t>(
              list, [](nb::handle v, std::size_t i) {
                checkListElementType<py_ext::Int>(v, i);
                return nb::cast<std::int16_t>(v);
              });
        if (ty.getIntOrFloatBitWidth() == 32)
          return appendValue.template operator()<std::int32_t>(
              list, [](nb::handle v, std::size_t i) {
                checkListElementType<py_ext::Int>(v, i);
                return nb::cast<std::int32_t>(v);
              });
        return appendValue.template operator()<std::int64_t>(
            list, [](nb::handle v, std::size_t i) {
              checkListElementType<py_ext::Int>(v, i);
              return nb::cast<std::int64_t>(v);
            });
      })
      .Case([&](mlir::Float32Type ty) {
        return appendValue.template operator()<float>(
            list, [](nb::handle v, std::size_t i) {
              checkListElementType<py_ext::Float>(v, i);
              return nb::cast<float>(v);
            });
      })
      .Case([&](mlir::Float64Type ty) {
        return appendValue.template operator()<double>(
            list, [](nb::handle v, std::size_t i) {
              checkListElementType<py_ext::Float>(v, i);
              return nb::cast<double>(v);
            });
      })
      .Case([&](cudaq::cc::CharspanType type) {
        return appendValue.template operator()<std::string>(
            list, [](nb::handle v, std::size_t i) {
              return nb::cast<cudaq::pauli_word>(v).str();
            });
      })
      .Case([&](ComplexType type) {
        if (isa<Float64Type>(type.getElementType()))
          return appendValue.template operator()<std::complex<double>>(
              list, [](nb::handle v, std::size_t i) {
                checkListElementType<py_ext::Complex>(v, i);
                return nb::cast<std::complex<double>>(v);
              });
        return appendValue.template operator()<std::complex<float>>(
            list, [](nb::handle v, std::size_t i) {
              checkListElementType<py_ext::Complex>(v, i);
              return nb::cast<std::complex<float>>(v);
            });
      })
      .Case([&](cudaq::cc::StdvecType ty) {
        auto appendVectorValue = []<typename T>(Type eleTy,
                                                nb::list list) -> void * {
          auto *values = new std::vector<std::vector<T>>();
          for (std::size_t i = 0; i < list.size(); i++) {
            auto ptr = handleVectorElements(eleTy, list[i]);
            auto *element = static_cast<std::vector<T> *>(ptr);
            values->emplace_back(std::move(*element));
          }
          return values;
        };

        auto eleTy = ty.getElementType();
        if (ty.getElementType().isInteger(1))
          // Special case for a `std::vector<bool>`.
          return appendVectorValue.template operator()<bool>(eleTy, list);

        // All other `std::Vector<T>` types, including nested vectors.
        return appendVectorValue.template operator()<std::size_t>(eleTy, list);
      })
      .Default([&](Type ty) {
        throw std::runtime_error("invalid list element type (" +
                                 mlirTypeToString(ty) + ").");
        return nullptr;
      });
}

inline void packArgs(OpaqueArguments &argData, nb::args args,
                     mlir::func::FuncOp kernelFuncOp,
                     const std::function<bool(OpaqueArguments &argData,
                                              nb::object &arg)> &backupHandler,
                     std::size_t startingArgIdx = 0) {
  if (kernelFuncOp.getNumArguments() != args.size())
    throw std::runtime_error("Invalid runtime arguments - kernel expected " +
                             std::to_string(kernelFuncOp.getNumArguments()) +
                             " but was provided " +
                             std::to_string(args.size()) + " arguments.");

  for (std::size_t i = startingArgIdx; i < args.size(); i++) {
    nb::object arg = args[i];
    auto kernelArgTy = kernelFuncOp.getArgument(i).getType();
    llvm::TypeSwitch<mlir::Type, void>(kernelArgTy)
        .Case([&](mlir::ComplexType ty) {
          checkArgumentType<py_ext::Complex>(arg, i);
          if (isa<Float64Type>(ty.getElementType())) {
            addArgument(argData, nb::cast<std::complex<double>>(arg));
          } else if (isa<Float32Type>(ty.getElementType())) {
            addArgument(argData, nb::cast<std::complex<float>>(arg));
          } else {
            throw std::runtime_error("Invalid complex type argument: " +
                                     nb::cast<std::string>(nb::str(args)) +
                                     " Type: " + mlirTypeToString(ty));
          }
        })
        .Case([&](mlir::Float64Type ty) {
          checkArgumentType<py_ext::Float>(arg, i);
          addArgument(argData, nb::cast<double>(arg));
        })
        .Case([&](mlir::Float32Type ty) {
          checkArgumentType<py_ext::Float>(arg, i);
          addArgument(argData, nb::cast<float>(arg));
        })
        .Case([&](mlir::Float32Type ty) {
          if (!nb::isinstance<nb::float_>(arg))
            throw std::runtime_error("kernel argument type is `float` but "
                                     "argument provided is not (argument " +
                                     std::to_string(i) + ", value=" +
                                     nb::cast<std::string>(nb::str(arg)) + ").");
          float *ourAllocatedArg = new float();
          *ourAllocatedArg = nb::cast<float>(arg);
          argData.emplace_back(ourAllocatedArg, [](void *ptr) {
            delete static_cast<float *>(ptr);
          });
        })
        .Case([&](mlir::IntegerType ty) {
          if (ty.getIntOrFloatBitWidth() == 1) {
            checkArgumentType<nb::bool_>(arg, i);
            addArgument(argData, nb::cast<bool>(arg));
            return;
          }

          checkArgumentType<py_ext::Int>(arg, i);
          addArgument(argData, nb::cast<std::int64_t>(arg));
        })
        .Case([&](cudaq::cc::CharspanType ty) {
          addArgument(argData, nb::cast<cudaq::pauli_word>(arg).str());
        })
        .Case([&](cudaq::cc::PointerType ty) {
          if (isa<quake::StateType>(ty.getElementType())) {
            addArgument(argData, cudaq::state(*nb::cast<cudaq::state *>(arg)));
          } else {
            throw std::runtime_error("Invalid pointer type argument: " +
                                     nb::cast<std::string>(nb::str(arg)) +
                                     " Type: " + mlirTypeToString(ty));
          }
        })
        .Case([&](cudaq::cc::StructType ty) {
          if (ty.getName() == "tuple") {
            auto [size, offsets] = getTargetLayout(kernelFuncOp, ty);
            auto memberTys = ty.getMembers();
            auto allocatedArg = std::malloc(size);
            auto elements = nb::cast<nb::tuple>(arg);
            for (std::size_t i = 0; i < offsets.size(); i++)
              handleStructMemberVariable(allocatedArg, offsets[i], memberTys[i],
                                         elements[i]);

            argData.emplace_back(allocatedArg,
                                 [](void *ptr) { std::free(ptr); });
          } else {
            auto [size, offsets] = getTargetLayout(kernelFuncOp, ty);
            auto memberTys = ty.getMembers();
            auto allocatedArg = std::malloc(size);
            nb::dict attributes = nb::cast<nb::dict>(arg.attr("__annotations__"));
            for (std::size_t i = 0;
                 const auto &[attr_name, unused] : attributes) {
              nb::object attr_value =
                  arg.attr(nb::cast<std::string>(attr_name).c_str());
              handleStructMemberVariable(allocatedArg, offsets[i], memberTys[i],
                                         attr_value);
              i++;
            }

            argData.emplace_back(allocatedArg,
                                 [](void *ptr) { std::free(ptr); });
          }
        })
        .Case([&](cudaq::cc::StdvecType ty) {
          auto appendVectorValue = [&argData]<typename T>(Type eleTy,
                                                          nb::list list) {
            auto allocatedArg = handleVectorElements(eleTy, list);
            argData.emplace_back(allocatedArg, [](void *ptr) {
              delete static_cast<std::vector<T> *>(ptr);
            });
          };

          checkArgumentType<nb::list>(arg, i);
          auto list = nb::cast<nb::list>(arg);
          auto eleTy = ty.getElementType();
          if (eleTy.isInteger(1)) {
            // Special case for a `std::vector<bool>`.
            appendVectorValue.template operator()<bool>(eleTy, list);
            return;
          }
          // All other `std::vector<T>` types, including nested vectors.
          appendVectorValue.template operator()<std::int64_t>(eleTy, list);
        })
        .Default([&](Type ty) {
          // See if we have a backup type handler.
          auto worked = backupHandler(argData, arg);
          if (!worked)
            throw std::runtime_error(
                "Could not pack argument: " + nb::cast<std::string>(nb::str(arg)) +
                " Type: " + mlirTypeToString(ty));
        });
  }
}

/// @brief Return true if the given `nb::args` represents a request for
/// broadcasting sample or observe over all argument sets. `args` types can be
/// `int`, `float`, `list`, so  we should check if `args[i]` is a `list` or
/// `ndarray`.
inline bool isBroadcastRequest(kernel_builder<> &builder, nb::args &args) {
  if (args.empty())
    return false;

  auto arg = args[0];
  // Just need to check the leading argument
  if (nb::isinstance<nb::list>(arg) && !builder.isArgStdVec(0))
    return true;

  if (nb::hasattr(arg, "tolist")) {
    if (!nb::hasattr(arg, "shape"))
      return false;

    auto shape = nb::cast<nb::tuple>(arg.attr("shape"));
    if (shape.size() == 1 && !builder.isArgStdVec(0))
      return true;

    // // If shape is 2, then we know its a list of list
    if (shape.size() == 2)
      return true;
  }

  return false;
}

} // namespace cudaq
