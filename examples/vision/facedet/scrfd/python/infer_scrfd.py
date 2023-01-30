# %%
import logging

import cv2
import fastdeploy as fd

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# %%
_model = "scrfd_500m_bnkps_shape640x640.onnx"
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
        option.set_trt_cache_file(f"/workspaces/sportai.py/.tensorrt_cache/{_model}.trt")
    return option


# %%
print(">> 配置runtime，加载模型")
# 配置runtime，加载模型
runtime_option = build_option(_device, _use_trt)
model = fd.vision.facedet.SCRFD(_model, runtime_option=runtime_option)

# %%
print(">> 加载图片")
# 预测图片检测结果
im = cv2.imread(_image)
cv2.imshow("original", im)
key = cv2.pollKey()

# %%
# %%timeit
# print(">> 检测结果")
result = model.predict(im)

# %%
print(result)

# %%
print(">> 预测结果可视化")
# 预测结果可视化
vis_im = fd.vision.vis_face_detection(im, result)
cv2.imshow("visualized_result.jpg", vis_im)

key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()
# print("Visualized result save in ./visualized_result.jpg")

# %%
# !pwd
# %%
