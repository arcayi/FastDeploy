# %%
from pathlib import Path
import fastdeploy as fd
import cv2
import os


def parse_arguments():
    import argparse
    import ast

    parser = argparse.ArgumentParser()
    parser.add_argument("--tinypose_model_dir", required=True, help="path of paddletinypose model directory")
    parser.add_argument("--image", required=True, help="path of test image file.")
    parser.add_argument(
        "--device", type=str, default="cpu", help="type of inference device, support 'cpu', 'kunlunxin' or 'gpu'."
    )
    parser.add_argument("--use_trt", type=ast.literal_eval, default=False, help="wether to use tensorrt.")
    return parser.parse_args()


def build_tinypose_option(device="gpu", use_trt=True):
    option = fd.RuntimeOption()

    if device.lower() == "gpu":
        option.use_gpu()

    if device.lower() == "kunlunxin":
        option.use_kunlunxin()

    if use_trt:
        option.use_trt_backend()
        option.set_trt_input_shape("image", [1, 3, 256, 192])
    return option


# %%
# args = parse_arguments()

# tinypose_model_file = os.path.join(args.tinypose_model_dir, "model.pdmodel")
# tinypose_params_file = os.path.join(args.tinypose_model_dir, "model.pdiparams")
# tinypose_config_file = os.path.join(args.tinypose_model_dir, "infer_cfg.yml")

tinypose_model_dir = "PP_TinyPose_256x192_infer"

tinypose_model_file = os.path.join(tinypose_model_dir, "model.pdmodel")
tinypose_params_file = os.path.join(tinypose_model_dir, "model.pdiparams")
tinypose_config_file = os.path.join(tinypose_model_dir, "infer_cfg.yml")

image = "../../hrnet_demo.jpg"

device = "gpu"
use_trt = True

# %%
# 配置runtime，加载模型
runtime_option = build_tinypose_option(device, use_trt)
# cache_path = Path.home() / ".cache" if cache_path is None else Path(cache_path)
cache_path = Path.home() / ".cache"
runtime_option.set_trt_cache_file(str(cache_path / f"{Path(tinypose_model_dir).name}.trt"))

tinypose_model = fd.vision.keypointdetection.PPTinyPose(
    tinypose_model_file, tinypose_params_file, tinypose_config_file, runtime_option=runtime_option
)
# 预测图片检测结果
im = cv2.imread(image)

# %%
# %%timeit
tinypose_result = tinypose_model.predict(im)

# %%
print("Paddle TinyPose Result:\n", tinypose_result)

# 预测结果可视化
vis_im = fd.vision.vis_keypoint_detection(im, tinypose_result, conf_threshold=0.5)
cv2.namedWindow("visualized_result.jpg", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.imshow("visualized_result.jpg", vis_im)
cv2.waitKey(0)
cv2.destroyAllWindows()
# print("TinyPose visualized result save in ./visualized_result.jpg")

# %%
