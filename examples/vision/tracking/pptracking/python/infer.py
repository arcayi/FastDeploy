# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import fastdeploy as fd
import cv2
import os
from video_grabber import VideoGrabber, cv_show_images, main_multiprocessing


def parse_arguments():
    import argparse
    import ast

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path of model.")
    parser.add_argument("--video", type=str, required=True, help="Path of test video file.")
    parser.add_argument("--device", type=str, default="cpu", help="Type of inference device, support 'cpu' or 'gpu'.")
    parser.add_argument("--use_trt", type=ast.literal_eval, default=False, help="Wether to use tensorrt.")
    return parser.parse_args()


def build_option(args):
    option = fd.RuntimeOption()

    if args.device.lower() == "gpu":
        option.use_gpu()

    if args.use_trt:
        option.use_trt_backend()
    return option


def pollKey(interval):
    if interval < 0:
        key = cv2.pollKey()
    else:
        key = cv2.waitKey(interval)
    # key = cv2.pollKey()
    if key > 0:
        logging.info(f"{key=}")
    if key == 27:  # exit on ESC
        raise RuntimeError("Exit on key")
    elif key == 32:
        # Pause on space bar
        key = cv2.waitKey(0)
        while key != 32:
            key = cv2.waitKey(0)
    # elif key == ord("1"):
    #     self.wakeup()

    elif key > 0:
        key = chr(key)
        # if draw_okv.toggle_by_key(key):
        #     logging.debug(f"{draw_okv.kos[key]=},{draw_okv.value_by_key(key)=}")
        #     if draw_okv.kos[key] == "draw_segmentation":
        #         infer_rt.postprocess_segmentation = draw_okv.value_by_option("draw_segmentation")


def vis_mot(frame, result, x=0.0, r=None):
    pass


def main(_args, _idx=0):

    # 配置runtime，加载模型
    runtime_option = build_option(_args)
    runtime_option.set_trt_cache_file(".cache.temp.trt")
    model_file = os.path.join(_args.model, "model.pdmodel")
    params_file = os.path.join(_args.model, "model.pdiparams")
    config_file = os.path.join(_args.model, "infer_cfg.yml")
    model = fd.vision.tracking.PPTracking(model_file, params_file, config_file, runtime_option=runtime_option)

    # 初始化轨迹记录器
    recorder = fd.vision.tracking.TrailRecorder()
    # 绑定记录器 注意：每次预测时，往trail_recorder里面插入数据，随着预测次数的增加，内存会不断地增长，
    # 可以通过unbind_recorder()方法来解除绑定
    model.bind_recorder(recorder)
    # 预测图片分割结果
    # cap = cv2.VideoCapture(args.video)
    vg = VideoGrabber(video_path=_args.video)

    # count = 0
    # while True:
    #     _, frame = cap.read()
    #     if frame is None:
    #         break
    vg.start()
    while vg.is_alive:
        try:
            frame = vg.get()
        except RuntimeError as e:
            break
        result = model.predict(frame)
        # count += 1
        # if count == 10:
        #     model.unbind_recorder()
        img = fd.vision.vis_mot(frame, result, 0.0, recorder)
        # cv2.imshow("video", img)
        kvs = {"video": img}
        cv_show_images(kvs)
        # if cv2.waitKey(30) == ord("q"):
        try:
            pollKey(-1)
        except:
            break
        vg.statistics()
    vg.statistics()
    model.unbind_recorder()
    # cap.release()
    vg.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(process)d,%(thread)x,%(name)s]%(asctime)s -%(levelname)s- %(message)s",
        level=logging.DEBUG,
    )

    args = parse_arguments()
    main_multiprocessing(main, args, n_workers=2)
    # main(args)
