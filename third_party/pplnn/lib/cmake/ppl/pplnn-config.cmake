cmake_minimum_required(VERSION 3.10)

if(TARGET "pplnn_basic_static")
    return()
endif()

add_library(pplnn_basic_static STATIC IMPORTED)

# --------------------------------------------------------------------------- #

get_filename_component(__PPLNN_PACKAGE_ROOTDIR__ "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

# --------------------------------------------------------------------------- #

# exported definitions

option(PPLNN_USE_X86_64 "" OFF)
option(PPLNN_USE_AARCH64 "" ON)
option(PPLNN_USE_RISCV64 "" OFF)
option(PPLNN_USE_CUDA "" OFF)
option(PPLNN_USE_LLM_CUDA "" OFF)

option(PPLNN_ENABLE_ONNX_MODEL "" ON)

# pmx is integrated in pplnn_basic_static
set(PPLNN_ENABLE_PMX_MODEL OFF)

set(PPLNN_VERSION_MAJOR 0)
set(PPLNN_VERSION_MINOR 0)
set(PPLNN_VERSION_PATCH 0)
set(PPLNN_COMMIT_STR "bb84dc9952809b3a5a0036858a3d9f68af1c83d5-dirty")

set(PPLNN_INCLUDE_DIRS "${__PPLNN_PACKAGE_ROOTDIR__}/include")
set(PPLNN_LINK_DIRS "${__PPLNN_PACKAGE_ROOTDIR__}/lib")
set(PPLNN_LIBRARIES "pplnn_basic_static")

# --------------------------------------------------------------------------- #

if(NOT TARGET "pplcommon_static")
    include(${CMAKE_CURRENT_LIST_DIR}/pplcommon-config.cmake)
endif()

if(PPLNN_USE_X86_64)
    if(NOT TARGET "pplkernelx86_static")
        include(${CMAKE_CURRENT_LIST_DIR}/pplkernelx86-config.cmake)
    endif()

    get_filename_component(__PPLNN_LIB_PATH__ "${__PPLNN_PACKAGE_ROOTDIR__}/lib/libpplnn_x86_static.a" ABSOLUTE)
    add_library(pplnn_x86_static STATIC IMPORTED)
    set_target_properties(pplnn_x86_static PROPERTIES
        INTERFACE_LINK_LIBRARIES "pplnn_basic_static;pplkernelx86_static"
        IMPORTED_LOCATION "${__PPLNN_LIB_PATH__}"
        IMPORTED_LOCATION_DEBUG "${__PPLNN_LIB_PATH__}"
        IMPORTED_LOCATION_RELEASE "${__PPLNN_LIB_PATH__}")
    unset(__PPLNN_LIB_PATH__)

    list(APPEND PPLNN_LIBRARIES pplnn_x86_static)
endif()

if(PPLNN_USE_AARCH64)
    if(NOT TARGET "pplkernelarm_static")
        include(${CMAKE_CURRENT_LIST_DIR}/pplkernelarm-config.cmake)
    endif()

    get_filename_component(__PPLNN_LIB_PATH__ "${__PPLNN_PACKAGE_ROOTDIR__}/lib/libpplnn_arm_static.a" ABSOLUTE)
    add_library(pplnn_arm_static STATIC IMPORTED)
    set_target_properties(pplnn_arm_static PROPERTIES
        INTERFACE_LINK_LIBRARIES "pplnn_basic_static;pplkernelarm_static"
        IMPORTED_LOCATION "${__PPLNN_LIB_PATH__}"
        IMPORTED_LOCATION_DEBUG "${__PPLNN_LIB_PATH__}"
        IMPORTED_LOCATION_RELEASE "${__PPLNN_LIB_PATH__}")
    unset(__PPLNN_LIB_PATH__)

    list(APPEND PPLNN_LIBRARIES pplnn_arm_static)
endif()

if(PPLNN_USE_CUDA)
    if(NOT TARGET "pplkernelcuda_static")
        include(${CMAKE_CURRENT_LIST_DIR}/pplkernelcuda-config.cmake)
    endif()

    get_filename_component(__PPLNN_LIB_PATH__ "${__PPLNN_PACKAGE_ROOTDIR__}/lib/libpplnn_cuda_static.a" ABSOLUTE)
    add_library(pplnn_cuda_static STATIC IMPORTED)
    set_target_properties(pplnn_cuda_static PROPERTIES
        INTERFACE_LINK_LIBRARIES "pplnn_basic_static;pplkernelcuda_static"
        IMPORTED_LOCATION "${__PPLNN_LIB_PATH__}"
        IMPORTED_LOCATION_DEBUG "${__PPLNN_LIB_PATH__}"
        IMPORTED_LOCATION_RELEASE "${__PPLNN_LIB_PATH__}")
    unset(__PPLNN_LIB_PATH__)

    list(APPEND PPLNN_LIBRARIES pplnn_cuda_static)
endif()

if(PPLNN_USE_RISCV64)
    if(NOT TARGET "pplkernelriscv_static")
        include(${CMAKE_CURRENT_LIST_DIR}/pplkernelriscv-config.cmake)
    endif()

    get_filename_component(__PPLNN_LIB_PATH__ "${__PPLNN_PACKAGE_ROOTDIR__}/lib/libpplnn_riscv_static.a" ABSOLUTE)
    add_library(pplnn_riscv_static STATIC IMPORTED)
    set_target_properties(pplnn_riscv_static PROPERTIES
        INTERFACE_LINK_LIBRARIES "pplnn_basic_static;pplkernelriscv_static"
        IMPORTED_LOCATION "${__PPLNN_LIB_PATH__}"
        IMPORTED_LOCATION_DEBUG "${__PPLNN_LIB_PATH__}"
        IMPORTED_LOCATION_RELEASE "${__PPLNN_LIB_PATH__}")
    unset(__PPLNN_LIB_PATH__)

    list(APPEND PPLNN_LIBRARIES pplnn_riscv_static)
endif()

if(PPLNN_USE_LLM_CUDA)
    if(NOT TARGET "pplkernelcuda_static")
        include(${CMAKE_CURRENT_LIST_DIR}/pplkernelcuda-config.cmake)
    endif()

    set(__LINK_LIBS__ pplnn_basic_static pplkernelcuda_static)
    if(PPLNN_CUDA_ENABLE_NCCL)
        list(APPEND __LINK_LIBS__ ${NCCL_LIBRARIES})
    endif()

    get_filename_component(__PPLNN_LIB_PATH__ "${__PPLNN_PACKAGE_ROOTDIR__}/lib/libppl_llm_cuda_static.a" ABSOLUTE)
    add_library(ppl_llm_cuda_static STATIC IMPORTED)
    set_target_properties(ppl_llm_cuda_static PROPERTIES
        INTERFACE_LINK_LIBRARIES "${__LINK_LIBS__}"
        IMPORTED_LOCATION "${__PPLNN_LIB_PATH__}"
        IMPORTED_LOCATION_DEBUG "${__PPLNN_LIB_PATH__}"
        IMPORTED_LOCATION_RELEASE "${__PPLNN_LIB_PATH__}")
    unset(__PPLNN_LIB_PATH__)

    unset(__LINK_LIBS__)

    list(APPEND PPLNN_LIBRARIES ppl_llm_cuda_static)
endif()

get_filename_component(__PPLNN_LIB_PATH__ "${__PPLNN_PACKAGE_ROOTDIR__}/lib/libpplnn_basic_static.a" ABSOLUTE)
set_target_properties(pplnn_basic_static PROPERTIES
    INTERFACE_LINK_LIBRARIES "pplcommon_static"
    IMPORTED_LOCATION "${__PPLNN_LIB_PATH__}"
    IMPORTED_LOCATION_DEBUG "${__PPLNN_LIB_PATH__}"
    IMPORTED_LOCATION_RELEASE "${__PPLNN_LIB_PATH__}")
unset(__PPLNN_LIB_PATH__)

# --------------------------------------------------------------------------- #

if(PPLNN_ENABLE_ONNX_MODEL)
    if(NOT TARGET "protobuf::libprotobuf")
        include(${__PPLNN_PACKAGE_ROOTDIR__}/lib/cmake/protobuf/protobuf-config.cmake)
    endif()

    get_filename_component(__PPLNN_LIB_PATH__ "${__PPLNN_PACKAGE_ROOTDIR__}/lib/libpplnn_onnx_pb_generated_static.a" ABSOLUTE)
    add_library(pplnn_onnx_generated_static STATIC IMPORTED)
    set_target_properties(pplnn_onnx_generated_static PROPERTIES
        INTERFACE_LINK_LIBRARIES "protobuf::libprotobuf"
        IMPORTED_LOCATION "${__PPLNN_LIB_PATH__}"
        IMPORTED_LOCATION_DEBUG "${__PPLNN_LIB_PATH__}"
        IMPORTED_LOCATION_RELEASE "${__PPLNN_LIB_PATH__}")

    get_filename_component(__PPLNN_LIB_PATH__ "${__PPLNN_PACKAGE_ROOTDIR__}/lib/libpplnn_onnx_static.a" ABSOLUTE)
    add_library(pplnn_onnx_static STATIC IMPORTED)
    set_target_properties(pplnn_onnx_static PROPERTIES
        INTERFACE_LINK_LIBRARIES "pplnn_basic_static;pplnn_onnx_generated_static"
        IMPORTED_LOCATION "${__PPLNN_LIB_PATH__}"
        IMPORTED_LOCATION_DEBUG "${__PPLNN_LIB_PATH__}"
        IMPORTED_LOCATION_RELEASE "${__PPLNN_LIB_PATH__}")
    unset(__PPLNN_LIB_PATH__)
    list(APPEND PPLNN_LIBRARIES pplnn_onnx_static)
endif()

# --------------------------------------------------------------------------- #

unset(__PPLNN_PACKAGE_ROOTDIR__)
