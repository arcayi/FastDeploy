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

# %%
import fastdeploy as fd
from fastdeploy import ModelFormat
import numpy as np
from pathlib import Path

# # 下载模型并解压
# model_url = "https://bj.bcebos.com/fastdeploy/models/mobilenetv2.onnx"
# fd.download(model_url, path=".")

option = fd.RuntimeOption()

model = Path(
    # "/workspaces/sportai.py/ai-models/src/ai_models/bpose/pose_detection/model_float32.onnx"
    "/workspaces/sportai.py/ai-models/src/ai_models/bpose/pose_landmark_full/modified_model_float32.onnx"
)
option.set_model_path(model_path=model.as_posix(), model_format=ModelFormat.ONNX)

# **** CPU 配置 ****
# option.use_cpu()
# option.use_ort_backend()
# option.set_cpu_thread_num(12)
option.use_gpu()
option.use_trt_backend()
option.set_trt_cache_file(f"/workspaces/sportai.py/.tensorrt_cache/{model.name}.trt")
option.set_cpu_thread_num(12)

# **** GPU 配置 ****
# 如需使用GPU，使用如下注释代码
# option.use_gpu(0)

# 初始化构造runtime
runtime = fd.Runtime(option)

# 获取模型输入名
input_name = runtime.get_input_info(0).name
input_shape = runtime.get_input_info(0).shape
input_dtype = runtime.get_input_info(0).dtype

print(f"{input_name=},{input_shape=},{input_dtype=}")

input = np.random.random(size=input_shape).astype("float32")
# %%
%%timeit -r 10 -n 1000
# 构造随机数据进行推理
# results = runtime.infer({input_name: np.random.rand(1, 3, 224, 224).astype("float32")})
results = runtime.infer({input_name: input})

# %%
print(results[0].shape)

# %%
