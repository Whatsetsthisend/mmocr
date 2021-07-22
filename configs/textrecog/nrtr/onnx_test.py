import onnxruntime as rt
import numpy as np
import cv2
np.set_printoptions(threshold=np.inf)
imgs = cv2.imread('G:/OCR/mmocr-fork/mmocr/demo/images/1.png')
imm = cv2.resize(imgs, (160, 32))

img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
new_img = imm / 255.

std = [0.229, 0.224, 0.225]
mean = [0.485, 0.456, 0.406]

_std = np.array(std).reshape((1, 1, 3))
_mean = np.array(mean).reshape((1, 1, 3))

new_img = (new_img - _mean) / _std
sess = rt.InferenceSession("./nrtr.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

print(input_name, label_name)
print(new_img.shape)

new_img = np.transpose([new_img], (0, 3, 1, 2))
# probability_name = sess.get_outputs()[1].name
pred_onx = sess.run([label_name], {input_name: new_img.astype(np.float32)})

# print info
print('input_name: ' + input_name)
print('label_name: ' + label_name)

np.set_printoptions(threshold=np.inf)
print(np.array(pred_onx[0][0][2]))