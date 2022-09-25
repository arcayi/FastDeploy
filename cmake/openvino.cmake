# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

include(ExternalProject)

set(OPENVINO_PROJECT "extern_openvino")

set(OPENVINO_VERSION "2022.2.0.dev20220829")
set(OPENVINO_URL_PREFIX "https://bj.bcebos.com/fastdeploy/third_libs/")

set(COMPRESSED_SUFFIX ".tgz")
if(WIN32)
  set(OPENVINO_FILENAME "w_openvino_toolkit_windows_${OPENVINO_VERSION}")
  set(COMPRESSED_SUFFIX ".zip")
  if(NOT CMAKE_CL_64)
    message(FATAL_ERROR "FastDeploy cannot ENABLE_OPENVINO_BACKEND in win32 now.")
  endif()
elseif(APPLE)
  if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "arm64")
    message("Cannot compile with openvino while in osx arm64 platform right now")
  else()
    set(OPENVINO_FILENAME "m_openvino_toolkit_osx_${OPENVINO_VERSION}")
  endif()
else()
  if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
    message("Cannot compile with openvino while in linux-aarch64 platform")
  else()
    set(OPENVINO_FILENAME "l_openvino_toolkit_centos7_${OPENVINO_VERSION}")
  endif()
endif()
set(OPENVINO_URL "${OPENVINO_URL_PREFIX}${OPENVINO_FILENAME}${COMPRESSED_SUFFIX}")


set(OPENVINO_INSTALL_DIR ${THIRD_PARTY_PATH}/install/${OPENVINO_FILENAME}/runtime)
set(OPENVINO_INSTALL_INC_DIR
  "${OPENVINO_INSTALL_DIR}/include"
  "${OPENVINO_INSTALL_DIR}/include/ie"
  CACHE PATH "openvino install include directory." FORCE)
  
set(OPENVINO_LIB_DIR
  "${OPENVINO_INSTALL_DIR}/lib/"
  "${OPENVINO_INSTALL_DIR}/3rdparty/tbb/lib/"
  CACHE PATH "openvino lib directory." FORCE)
set(CMAKE_BUILD_RPATH "${CMAKE_BUILD_RPATH}" "${OPENVINO_LIB_DIR}")

# For OPENVINO code to include internal headers.
include_directories(${OPENVINO_INSTALL_INC_DIR})

download_and_decompress(${OPENVINO_URL}
    ${CMAKE_CURRENT_BINARY_DIR}/${OPENVINO_FILENAME}${COMPRESSED_SUFFIX}
    ${THIRD_PARTY_PATH}/install)

if(WIN32)
  set(OPENVINO_LIB
      "${OPENVINO_INSTALL_DIR}/lib/openvino.lib"
      CACHE FILEPATH "OPENVINO shared library." FORCE)
  file(GLOB_RECURSE OPENVINO_LIB_FILES ${OPENVINO_INSTALL_DIR}/lib/intel64/Release/*)
  file(COPY ${OPENVINO_LIB_FILES} DESTINATION ${OPENVINO_INSTALL_DIR}/lib/)
  file(REMOVE_RECURSE ${OPENVINO_INSTALL_DIR}/lib/intel64)

  file(GLOB_RECURSE OPENVINO_BIN_FILES ${OPENVINO_INSTALL_DIR}/bin/intel64/Release/*)
  file(COPY ${OPENVINO_BIN_FILES} DESTINATION ${OPENVINO_INSTALL_DIR}/bin/)
  file(REMOVE_RECURSE ${OPENVINO_INSTALL_DIR}/bin/intel64)
elseif(APPLE)
  set(OPENVINO_LIB
      "${OPENVINO_INSTALL_DIR}/lib/libopenvino.dylib"
      CACHE FILEPATH "OPENVINO shared library." FORCE)
  file(GLOB_RECURSE OPENVINO_LIB_FILES ${OPENVINO_INSTALL_DIR}/lib/intel64/Release/*)
  file(COPY ${OPENVINO_LIB_FILES} DESTINATION ${OPENVINO_INSTALL_DIR}/lib/)
  file(REMOVE_RECURSE ${OPENVINO_INSTALL_DIR}/lib/intel64)
else()
  set(OPENVINO_LIB
      "${OPENVINO_INSTALL_DIR}/lib/libopenvino.so"
      CACHE FILEPATH "OPENVINO shared library." FORCE)
  file(GLOB_RECURSE OPENVINO_LIB_FILES ${OPENVINO_INSTALL_DIR}/lib/intel64/*)
  file(COPY ${OPENVINO_LIB_FILES} DESTINATION ${OPENVINO_INSTALL_DIR}/lib/)
  file(REMOVE_RECURSE ${OPENVINO_INSTALL_DIR}/lib/intel64)
endif()

file(REMOVE_RECURSE ${THIRD_PARTY_PATH}/install/${OPENVINO_FILENAME}/docs)
file(REMOVE_RECURSE ${THIRD_PARTY_PATH}/install/${OPENVINO_FILENAME}/install_dependencies)
file(REMOVE_RECURSE ${THIRD_PARTY_PATH}/install/${OPENVINO_FILENAME}/samples)
file(REMOVE_RECURSE ${THIRD_PARTY_PATH}/install/${OPENVINO_FILENAME}/setupvars.sh)
file(REMOVE_RECURSE ${THIRD_PARTY_PATH}/install/${OPENVINO_FILENAME}/tools)

add_library(external_openvino STATIC IMPORTED GLOBAL)
set_property(TARGET external_openvino PROPERTY IMPORTED_LOCATION ${OPENVINO_LIB})
set(OPENVINO_LIBS external_openvino)

find_package(TBB PATHS "${OPENVINO_INSTALL_DIR}/3rdparty/tbb")
if (TBB_FOUND)
  list(APPEND OPENVINO_LIBS ${TBB_IMPORTED_TARGETS})
else()
  # TODO(zhoushunjie): Use openvino with tbb on linux in future.
  set(OMP_LIB "${OPENVINO_INSTALL_DIR}/3rdparty/omp/lib/libiomp5.so")
  add_library(omp STATIC IMPORTED GLOBAL)
  set_property(TARGET omp PROPERTY IMPORTED_LOCATION
                                          ${OMP_LIB})
  list(APPEND OPENVINO_LIBS omp)
endif()
message(STATUS "OPENVINO_LIBS = ${OPENVINO_LIBS}")

list(APPEND DEPEND_LIBS ${OPENVINO_LIBS})
