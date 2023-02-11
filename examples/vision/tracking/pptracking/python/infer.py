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
import os

import cv2
import numpy as np
from video_grabber import VideoGrabber, cv_show_images, main_multiprocessing

import fastdeploy as fd


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


def GetMOTBoxColor(idx: int):
    idx = idx * 3
    color = (37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255
    return color


def vis_mot(img: np.ndarray, results: fd.C.vision.MOTResult, score_threshold: float = 0.0, recorder=None):
    vis_img = img.copy()
    im_h, im_w = img.shape[:2]
    text_scale = max(1, int(im_w / 1600.0))
    text_thickness = 2.0
    line_thickness = max(1, int(im_w / 500.0))

    # for (int i = 0; i < results.boxes.size(); ++i) {
    for i, box in enumerate(results.boxes):
        pass
        if results.scores[i] < score_threshold:
            continue
        obj_id = results.ids[i]
        score = results.scores[i]
        color = GetMOTBoxColor(obj_id)
        if recorder is not None:
            id = results.ids[i]
            # logging.info(f"{recorder.records= }")
            iter = recorder.records.get(id, None)
            # if (iter != recorder->records.end()) {
            if iter is not None:
                # logging.info(f"{iter= }")
                for j, xy in enumerate(iter):
                    center = xy
                    cv2.circle(vis_img, center, int(text_thickness), color)
        pt1 = np.array([results.boxes[i][0], results.boxes[i][1]])
        pt2 = np.array([results.boxes[i][2], results.boxes[i][3]])
        id_pt = np.array([results.boxes[i][0], results.boxes[i][1] + 10])
        score_pt = np.array([results.boxes[i][0], results.boxes[i][1] - 10])
        cv2.rectangle(vis_img, pt1, pt2, color, line_thickness)
        #     std::ostringstream idoss;
        #     idoss << std::setiosflags(std::ios::fixed) << std::setprecision(4);
        #     idoss << obj_id;
        id_text = f"{obj_id}"

        cv2.putText(vis_img, id_text, id_pt, cv2.FONT_HERSHEY_PLAIN, text_scale, color, int(text_thickness))

        #     std::ostringstream soss;
        #     soss << std::setiosflags(std::ios::fixed) << std::setprecision(2);
        #     soss << score;
        score_text = f"{score:.2f}"

        cv2.putText(vis_img, score_text, score_pt, cv2.FONT_HERSHEY_PLAIN, text_scale, color, int(text_thickness))

    return vis_img


def main(_args, _idx=0):

    # 配置runtime，加载模型
    runtime_option = build_option(_args)
    runtime_option.set_trt_cache_file(".cache.temp.trt")
    model_file = os.path.join(_args.model, "model.pdmodel")
    params_file = os.path.join(_args.model, "model.pdiparams")
    config_file = os.path.join(_args.model, "infer_cfg.yml")
    model = fd.vision.tracking.PPTracking(model_file, params_file, config_file, runtime_option=runtime_option)

    # # 初始化轨迹记录器
    # recorder = fd.vision.tracking.TrailRecorder()
    # # 绑定记录器 注意：每次预测时，往trail_recorder里面插入数据，随着预测次数的增加，内存会不断地增长，
    # # 可以通过unbind_recorder()方法来解除绑定
    # model.bind_recorder(recorder)

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
        # img = fd.vision.vis_mot(frame, result, 0.0, recorder)
        # img1 = vis_mot(frame, result, 0.0, recorder)
        img1 = vis_mot(
            frame,
            result,
            0.0,
        )
        # cv2.imshow("video", img)
        # kvs = {"video": img, "vis_mot": img1}
        kvs = {"vis_mot": img1}
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
    # main_multiprocessing(main, args, n_workers=2)
    main(args)
