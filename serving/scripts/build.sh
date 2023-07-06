#!/usr/bin/env bash
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARGS=$(getopt -a -o w:n:h:hs:tv:cv:dv -l WITH_GPU:,docker_name:,http_proxy:,https_proxy:,trt_version:,cuda_version:,cudnn_version: -- "$@")

eval set -- "${ARGS}"
echo "parse start"

while true; do
    case "$1" in
    -w | --WITH_GPU)
        WITH_GPU="$2"
        shift
        ;;
    -n | --docker_name)
        docker_name="$2"
        shift
        ;;
    -h | --http_proxy)
        http_proxy="$2"
        shift
        ;;
    -hs | --https_proxy)
        https_proxy="$2"
        shift
        ;;
    -tv | --trt_version)
        trt_version="$2"
        shift
        ;;
    -cv | --cuda_version)
        cuda_version="$2"
        shift
        ;;
    # -dv|--cudnn_version)
    #         cudnn_version="$2"
    #         shift;;
    --)
        shift
        break
        ;;
    esac
    shift
done

if [ -z $WITH_GPU ]; then
    WITH_GPU="ON"
fi

if [ -z $docker_name ]; then
    docker_name="build_fd"
fi

# cd /workspaces/sportai.py/FastDeploy/serving

if [ -z $trt_version ]; then
    # The optional value of trt_version: ["8.4.1.5", "8.5.2.2"]
    trt_version="8.6.1.6"
fi
if [ -z $cuda_version ]; then
    cuda_version="12.0"
fi
# if [ -z $cudnn_version ]; then
#     cudnn_version="8.6"
# fi

# if [ $trt_version == "8.5.2.2" ]
# then
#     cudnn_version="11.8"
#     cudnn_version="8.6"
# else
#     cuda_version="11.6"
#     cudnn_version="8.4"
# fi

echo "start build FD GPU library"

if [ ! -d "./cmake-3.26.4-linux-x86_64/" ]; then
    wget https://github.com/Kitware/CMake/releases/download/v3.26.4/cmake-3.26.4-Linux-x86_64.tar.gz
    tar -zxvf cmake-3.26.4-Linux-x86_64.tar.gz
    rm -rf cmake-3.26.4-Linux-x86_64.tar.gz
fi

if [ ! -d "./TensorRT-${trt_version}/" ]; then
    # wget https://fastdeploy.bj.bcebos.com/resource/TensorRT/TensorRT-${trt_version}.Linux.x86_64-gnu.cuda-${cuda_version}.cudnn${cudnn_version}.tar.gz
    # tar -zxvf TensorRT-${trt_version}.Linux.x86_64-gnu.cuda-${cuda_version}.cudnn${cudnn_version}.tar.gz
    # rm -rf TensorRT-${trt_version}.Linux.x86_64-gnu.cuda-${cuda_version}.cudnn${cudnn_version}.tar.gz
    wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/${trt_version%.*}/tars/TensorRT-${trt_version}.Linux.x86_64-gnu.cuda-${cuda_version}.tar.gz
    tar -zxvf TensorRT-${trt_version}.Linux.x86_64-gnu.cuda-${cuda_version}.tar.gz
    rm -rf TensorRT-${trt_version}.Linux.x86_64-gnu.cuda-${cuda_version}.tar.gz
fi

workspace_path="/data/Git_Repository/Projects_AI/sportai.py/FastDeploy/"

# nvidia-docker run -i --rm --name ${docker_name} \
docker run -i --rm --gpus all --name ${docker_name} \
    -v ${workspace_path}:/workspace/fastdeploy \
    -e "http_proxy=${http_proxy}" \
    -e "https_proxy=${https_proxy}" \
    -e "trt_version=${trt_version}" \
    nvcr.io/nvidia/tritonserver:23.05-py3 \
    bash -c \
    'export https_proxy_tmp=${https_proxy}
        export http_proxy_tmp=${http_proxy}
        sed -i -e "s|http\(s\)*:\(.*\)\/ubuntu|http:\/\/mirrors.ustc.edu.cn\/ubuntu|g" /etc/apt/sources.list
        echo "!!! build fastdeploy python"
        cd /workspace/fastdeploy/python;
        rm -rf .setuptools-cmake-build dist build fastdeploy/libs/third_libs fastdeploy/libs/*.so;
        apt-get update;
        apt-get install -y --no-install-recommends patchelf python3-dev python3-pip rapidjson-dev git;
        unset http_proxy
        unset https_proxy
        git config --global --add safe.directory /workspace/fastdeploy
        ln -s /usr/bin/python3 /usr/bin/python;
        export PATH=/workspace/fastdeploy/serving/cmake-3.26.4-linux-x86_64/bin:$PATH;
        export WITH_GPU=ON;
        export ENABLE_TRT_BACKEND=ON;
        export TRT_DIRECTORY=/workspace/fastdeploy/serving/TensorRT-${trt_version}/;
        export ENABLE_ORT_BACKEND=ON;
        export ENABLE_PADDLE_BACKEND=ON;
        export ENABLE_OPENVINO_BACKEND=OFF;
        export ENABLE_VISION=ON;
        # export ENABLE_VISION=OFF;
        export ENABLE_TEXT=OFF;
        python setup.py build;
        python setup.py bdist_wheel;
        echo "!!! build fastdeploy"
        cd /workspace/fastdeploy;
        rm -rf build; mkdir -p build;cd build;
        cmake .. \
            -D WITH_GPU=ON  \
            -D ENABLE_TRT_BACKEND=ON \
            -D CMAKE_INSTALL_PREFIX=${PWD}/fastdeploy_install \
            -D TRT_DIRECTORY=/workspace/fastdeploy/serving/TensorRT-${trt_version}/ \
            -D ENABLE_ORT_BACKEND=ON \
            -D ENABLE_PADDLE_BACKEND=ON \
            -D ENABLE_OPENVINO_BACKEND=OFF \
            -D ENABLE_VISION=ON \
            -D ENABLE_TEXT=OFF \
            -D BUILD_FASTDEPLOY_PYTHON=OFF \
            -D ENABLE_PADDLE2ONNX=ON \
            -D LIBRARY_NAME=fastdeploy_runtime;
        # cmake .. -DCMAKE_INSTALL_PREFIX=${PWD}/fastdeploy_install -DBUILD_FASTDEPLOY_PYTHON=OFF -DENABLE_PADDLE2ONNX=ON -DLIBRARY_NAME=fastdeploy_runtime;
        make -j`nproc`;
        make install;
        echo "!!! build fastdeploy serving"
        cd /workspace/fastdeploy/serving;
        rm -rf build; mkdir build; cd build;
        export https_proxy=${https_proxy_tmp}
        export http_proxy=${http_proxy_tmp}
        cmake .. -DFASTDEPLOY_DIR=/workspace/fastdeploy/build/fastdeploy_install -DTRITON_COMMON_REPO_TAG=r23.05 -DTRITON_CORE_REPO_TAG=r23.05 -DTRITON_BACKEND_REPO_TAG=r23.05;
        make -j`nproc`'

echo "build FD GPU library done"
