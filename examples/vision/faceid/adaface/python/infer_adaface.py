# %%
import logging

import cv2
import fastdeploy as fd
import numpy as np
from matplotlib import pyplot as plt

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# %%
_model = "mobilefacenet_adaface/mobilefacenet_adaface.pdmodel"
_params_file = "mobilefacenet_adaface/mobilefacenet_adaface.pdiparams"
_device = "gpu"
_use_trt = True
_face = "../../face_demo/face_0.jpg"
_face_positive = "../../face_demo/face_1.jpg"
_face_negative = "../../face_demo/face_2.jpg"


# 余弦相似度
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    mul_a = np.linalg.norm(a, ord=2)
    mul_b = np.linalg.norm(b, ord=2)
    mul_ab = np.dot(a, b)
    return mul_ab / (np.sqrt(mul_a) * np.sqrt(mul_b))


def build_option(_device, _use_trt):
    option = fd.RuntimeOption()

    if _device.lower() == "gpu":
        option.use_gpu()

    if _device.lower() == "kunlunxin":
        option.use_kunlunxin()

    if _use_trt:
        option.use_trt_backend()
        option.set_trt_input_shape("data", [1, 3, 112, 112])
        option.set_trt_cache_file(
            f"/workspaces/sportai.py/.tensorrt_cache/{_model}.trt"
        )
    return option


# %%
print(">> 配置runtime，加载模型")
# 配置runtime，加载模型
from fastdeploy.runtime import ModelFormat

runtime_option = build_option(_device, _use_trt)
model = fd.vision.faceid.AdaFace(_model, _params_file, runtime_option=runtime_option, model_format=ModelFormat.PADDLE)

# %%
%matplotlib inline

print(">> 加载图片")
# 预测图片检测结果
face0 = cv2.imread(_face)  # 0,1 同一个人
face1 = cv2.imread(_face_positive)
face2 = cv2.imread(_face_negative)  # 0,2 不同的人

# im = cv2.imread(_image)
images = (face0,face1,face2)
for i,ima in enumerate(images):
    plt.subplot(1, len(images), i+1)
    plt.imshow(cv2.cvtColor(ima,cv2.COLOR_BGR2RGB))
plt.show()

# %%
# print(">> 检测结果")
# result = model.predict(im)

model.l2_normalize = True

result0 = model.predict(face0)
result1 = model.predict(face1)
result2 = model.predict(face2)

embedding0 = result0.embedding
embedding1 = result1.embedding
embedding2 = result2.embedding

    # cosine01 = cosine_similarity(embedding0, embedding1)
    # cosine02 = cosine_similarity(embedding0, embedding2)

# %%
for i, result in enumerate([result0,result1,result2]):
    print(f"{i}: {result}", end="")


from itertools import combinations
emb_list = [embedding0,embedding1,embedding2]
for i, j in combinations(range(3),2):
    print(f"cosine_similarity({i}, {j}):  {cosine_similarity(emb_list[i], emb_list[j])}")

print(model.runtime_option)

# %%
%%timeit
result0 = model.predict(face0)

# %%
dir(result0)
# %%
result0.embedding
# %%
np.linalg.norm(result0.embedding, ord=2)

# %%
