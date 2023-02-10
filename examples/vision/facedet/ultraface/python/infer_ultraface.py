# %%
import logging

import cv2
import fastdeploy as fd
import numpy as np
from matplotlib import pyplot as plt

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# %%
_model = "version-RFB-320.onnx"
_image = "../../test_lite_face_detector_3.jpg"
_device = "gpu"
_use_trt = True


def build_option(_device, _use_trt):
    option = fd.RuntimeOption()

    if _device.lower() == "gpu":
        option.use_gpu()

    if _use_trt:
        option.use_trt_backend()
        option.set_trt_input_shape("images", [1, 3, 640, 640])
        option.set_trt_cache_file(
            f"/workspaces/sportai.py/.tensorrt_cache/{_model}.trt"
        )
    return option


# %%
print(">> 配置runtime，加载模型")
# 配置runtime，加载模型
runtime_option = build_option(_device, _use_trt)
model = fd.vision.facedet.UltraFace(_model, runtime_option=runtime_option)

# %%
print(">> 加载图片")
# 预测图片检测结果
im = cv2.imread(_image)
plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
plt.show()

# %%
# %%timeit
# print(">> 检测结果")
result = model.predict(im)

# %%
print(result)

# %%
dir(result)
# %%
result.landmarks_per_face
# %%
result.boxes
# %%
result.scores

# %%
print(">> 预测结果可视化")
# 预测结果可视化
vis_im = fd.vision.vis_face_detection(im, result, line_size=5)
plt.imshow(cv2.cvtColor(vis_im, cv2.COLOR_BGR2RGB))
plt.show()

# print("Visualized result save in ./visualized_result.jpg")

# %%
# !pwd
# %%
