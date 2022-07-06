import onnxruntime as ort
import numpy as np
import os
from PIL import Image
import copy

colors = [ (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
                            (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128),
                            (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
                            (128, 64, 12)]

def preprocess_input(image):
    image /= 255.0
    return image

def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[-2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image

def resize_image(image, size):
    iw, ih  = image.size
    w, h    = size

    scale   = min(w/iw, h/ih)
    nw      = int(iw*scale)
    nh      = int(ih*scale)

    image   = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    return new_image, nw, nh

# 读取图片
model_path = 'model_data/MAnet/best_epoch_weights.pth'
onnx_path = os.path.split(model_path)[0] + '/'
infer_path = "test_500size/2/"
VOCdevkit_path = 'VOCdevkit/' + infer_path + '1_21.jpg'
input_shape = (500, 500)

# img = cv2.imread(VOCdevkit_path)
# print(img.shape)

img = Image.open(VOCdevkit_path)
img.resize((500,500), Image.BICUBIC)
orininal_h, orininal_w = img.size
img = cvtColor(img)
old_img   = copy.deepcopy(img)

img  = np.expand_dims(np.transpose(preprocess_input(np.array(img, np.float32)), (2, 0, 1)), 0)

# 加载 onnx
onnx_name = onnx_path + 'torch.onnx'
sess = ort.InferenceSession(onnx_name, providers=['CPUExecutionProvider']) # 'CUDAExecutionProvider'
input_name = sess.get_inputs()[0].name
output_name = [output.name for output in sess.get_outputs()]

# 推理
outputs = sess.run(output_name, {input_name:img})
outputs = np.array(outputs)
seg_img = np.reshape(np.array(colors, np.uint8)[np.reshape(outputs, [-1])], [orininal_h, orininal_w, -1])
image   = Image.fromarray(np.uint8(seg_img))
image   = Image.blend(old_img, image, 0.7)
image.save("./test.tif")
print("1")
# print(outputs)
# print(outputs.shape)



# count = 0
# for i in range(outputs.shape[0]):
#     for j in range(outputs.shape[1]):
#         if outputs[i][j] == 1:
#             count = count + 1
#             outputs[i][j] = 255

# image = Image.fromarray(np.uint8(outputs))
# image.save("./test.png")
# print(count)
# outputs = [torch.Tensor(x) for x in outputs]  # 转换为 tensor