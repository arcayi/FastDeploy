# %%
import fastdeploy as fd
import cv2
import os


def parse_arguments():
    import argparse
    import ast

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Path of yolov8 model.")
    parser.add_argument("--image", default=None, help="Path of test image file.")
    parser.add_argument(
        "--device", type=str, default="cpu", help="Type of inference device, support 'cpu' or 'gpu' or 'kunlunxin'."
    )
    parser.add_argument("--use_trt", type=ast.literal_eval, default=False, help="Wether to use tensorrt.")
    return parser.parse_args()


def build_option(device="gpu", use_trt=True):
    option = fd.RuntimeOption()

    if device.lower() == "gpu":
        option.use_gpu()

    if device.lower() == "ascend":
        option.use_ascend()

    if use_trt:
        option.use_trt_backend()
        option.set_trt_input_shape("images", [1, 3, 640, 640])
    return option


# %%
# args = parse_arguments()
model = "/workspaces/sportai.py/FastDeploy/examples/vision/detection/yolov8/yolov8s.onnx"
image = "/workspaces/sportai.py/FastDeploy/benchmark/000000014439.jpg"

# %%
# Configure runtime, load model
runtime_option = build_option(device="gpu", use_trt=True)
model = fd.vision.detection.YOLOv8(model, runtime_option=runtime_option)

# Predicting image
if image is None:
    image = fd.utils.get_detection_test_image()
else:
    image = image
im = cv2.imread(image)
# %%
# %%timeit
result = model.predict(im)

# %%
# Visualization
vis_im = fd.vision.vis_detection(im, result)
from matplotlib import pyplot as plt

plt.imshow(cv2.cvtColor(vis_im, cv2.COLOR_BGR2RGB))

# cv2.imwrite("visualized_result.jpg", vis_im)
print("Visualized result save in ./visualized_result.jpg")

# %%
