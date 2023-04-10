# %%
from pathlib import Path
import fastdeploy as fd
import numpy as np
import cv2
import os


def parse_arguments():
    import argparse
    import ast

    parser = argparse.ArgumentParser()
    parser.add_argument("--tinypose_model_dir", required=True, help="path of paddletinypose model directory")
    parser.add_argument("--det_model_dir", help="path of paddledetection model directory")
    parser.add_argument("--image", required=True, help="path of test image file.")
    parser.add_argument(
        "--device", type=str, default="cpu", help="type of inference device, support 'cpu', 'kunlunxin' or 'gpu'."
    )
    parser.add_argument("--use_trt", type=ast.literal_eval, default=False, help="wether to use tensorrt.")
    return parser.parse_args()


def build_picodet_option(device="gpu", use_trt=True):
    option = fd.RuntimeOption()

    if device.lower() == "gpu":
        option.use_gpu()

    if use_trt:
        option.use_trt_backend()
        option.set_trt_input_shape("image", [1, 3, 320, 320])
        option.set_trt_input_shape("scale_factor", [1, 2])
        option.enable_trt_fp16()
        option.enable_pinned_memory()
    return option


def build_tinypose_option(device="gpu", use_trt=True):
    option = fd.RuntimeOption()

    if device.lower() == "gpu":
        option.use_gpu()

    if device.lower() == "kunlunxin":
        option.use_kunlunxin()

    if use_trt:
        option.use_trt_backend()
        option.set_trt_input_shape("image", [1, 3, 256, 192])
        option.enable_trt_fp16()
        option.enable_pinned_memory()
    return option


det_model_dir = "PP_PicoDet_V2_S_Pedestrian_320x320_infer"
tinypose_model_dir = "../../tiny_pose/python/PP_TinyPose_256x192_infer"

device = "gpu"
use_trt = True

# %%
# args = parse_arguments()
picodet_model_file = os.path.join(det_model_dir, "model.pdmodel")
picodet_params_file = os.path.join(det_model_dir, "model.pdiparams")
picodet_config_file = os.path.join(det_model_dir, "infer_cfg.yml")

# %%
# 配置runtime，加载PicoDet模型
runtime_option = build_picodet_option(device, use_trt)
# cache_path = Path.home() / ".cache" if cache_path is None else Path(cache_path)
cache_path = Path.home() / ".cache"
runtime_option.set_trt_cache_file(str(cache_path / f"{Path(det_model_dir).name}.trt"))
det_model = fd.vision.detection.PicoDet(
    picodet_model_file, picodet_params_file, picodet_config_file, runtime_option=runtime_option
)

# %%

tinypose_model_file = os.path.join(tinypose_model_dir, "model.pdmodel")
tinypose_params_file = os.path.join(tinypose_model_dir, "model.pdiparams")
tinypose_config_file = os.path.join(tinypose_model_dir, "infer_cfg.yml")

# %%
# 配置runtime，加载PPTinyPose模型
runtime_option = build_tinypose_option(device, use_trt)
# cache_path = Path.home() / ".cache" if cache_path is None else Path(cache_path)
cache_path = Path.home() / ".cache"
runtime_option.set_trt_cache_file(str(cache_path / f"{Path(tinypose_model_dir).name}.trt"))
tinypose_model = fd.vision.keypointdetection.PPTinyPose(
    tinypose_model_file, tinypose_params_file, tinypose_config_file, runtime_option=runtime_option
)

# %%
# 预测图片检测结果
# image = "../../000000018491.jpg"
# image = "/workspaces/data/SportAi/跑道/20230304/12/vlcsnap-2023-03-18-14h20m48s705.png"
image = "/workspaces/data/SportAi/跑道/20230304/12/vlcsnap-2023-03-18-14h27m51s653.png"
im = cv2.imread(image)
pipeline = fd.pipeline.PPTinyPose(det_model, tinypose_model)
pipeline.detection_model_score_threshold = 0.4

# %%
# %%timeit
pipeline_result = pipeline.predict(im)

# %%
print(f"{np.array(pipeline_result.keypoints).shape=}, {pipeline_result.keypoints=}")
print(f"{np.array(pipeline_result.scores).shape=}, {pipeline_result.scores=}")
# print(f"{pipeline_result.scores=}")
print(f"{pipeline_result.num_joints=}")
print("#person:\n", np.array(pipeline_result.scores).shape[0] / pipeline_result.num_joints)
# print("Paddle TinyPose Result:\n", pipeline_result)

# 预测结果可视化
vis_im = fd.vision.vis_keypoint_detection(im, pipeline_result, conf_threshold=0.2)
cv2.namedWindow("visualized_result.jpg", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.imshow("visualized_result.jpg", vis_im)
cv2.waitKey(0)
cv2.destroyAllWindows()
# print("TinyPose visualized result save in ./visualized_result.jpg")

# %%
