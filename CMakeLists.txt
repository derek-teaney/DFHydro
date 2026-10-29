cmake_minimum_required(VERSION 3.24)

# # Get the Petsc running 
execute_process ( COMMAND pkg-config PETSc --variable=ccompiler COMMAND tr -d '\n' OUTPUT_VARIABLE C_COMPILER)
SET(CMAKE_C_COMPILER ${C_COMPILER})
execute_process ( COMMAND pkg-config PETSc --variable=cxxcompiler COMMAND tr -d '\n' OUTPUT_VARIABLE CXX_COMPILER)
if (CXX_COMPILER)
  SET(CMAKE_CXX_COMPILER ${CXX_COMPILER})
endif (CXX_COMPILER)

# Set the project name
project(DFHydro)

# Set the C++ standard to c++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


find_package(PkgConfig REQUIRED)
pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET PETSc)
# Make an alias for PETSc instead of using PkgConfig::PETSC everywhere
add_library(PETSc ALIAS PkgConfig::PETSC)


option(PETSC_HAS_HDF5 "PETSc built with HDF5 support" OFF)
# Check for HDF5 support in PETSc
if (PETSC_HAS_HDF5)
    message(STATUS "PETSc built with HDF5 support")
    add_compile_definitions(PETSC_HAVE_HDF5)
    pkg_check_modules(HDF5 REQUIRED IMPORTED_TARGET hdf5)
else()
    message(STATUS "PETSc not built with HDF5 support")
endif()

add_subdirectory(DFHydro)
add_subdirectory(example)
if (PETSC_HAS_HDF5)
    add_subdirectory(tests)
endif()



