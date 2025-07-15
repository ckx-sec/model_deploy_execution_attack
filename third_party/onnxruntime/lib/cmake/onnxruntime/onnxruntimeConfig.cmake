
####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was PROJECT_CONFIG_FILE                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################
include(CMakeFindDependencyMacro)
        find_dependency(absl)
        find_dependency(date)
        find_dependency(Eigen3)
        find_dependency(nlohmann_json)
        find_dependency(ONNX)
        find_dependency(re2)
        find_dependency(flatbuffers)
        find_dependency(cpuinfo)
        find_dependency(protobuf)
        find_dependency(Boost COMPONENTS mp11)
        find_dependency(Microsoft.GSL 4.0)
        if(NOT WIN32 AND NOT CMAKE_SYSTEM_NAME STREQUAL "Android")
        find_dependency(Iconv)
        endif()
        if(WIN32)
        find_dependency(wil)
        endif()
        find_path(safeint_SOURCE_DIR NAMES "SafeInt.hpp" REQUIRED)
        add_library(safeint_interface IMPORTED INTERFACE)
        target_include_directories(safeint_interface INTERFACE /workspace/model_deploy_dataset/third_party_builds/onnxruntime_source/build_static/Release/_deps/safeint-src)
        include("${CMAKE_CURRENT_LIST_DIR}/onnxruntimeTargets.cmake")
