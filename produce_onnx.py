from nets.deeplabv3_version_1.MAnet import MAnet
import torch
import os
from PIL import Image
import numpy as np
import onnx
import onnxruntime

def preprocess_input(image):
    image /= 255.0
    return image

def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[-2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image

# 检查输出
def check_onnx_output(filename, input_data, torch_output):
    print("模型测试")
    session = onnxruntime.InferenceSession(filename)
    input_name = session.get_inputs()[0].name
    result = session.run([], {input_name: input_data.detach().cpu().numpy()})
    for test_result, gold_result in zip(result, torch_output.values()):
        np.testing.assert_almost_equal(
            gold_result.cpu().numpy(), test_result, decimal=3,
        )
    return result
# 检查模型
def check_onnx_model(model, onnx_filename, input_image):
    with torch.no_grad():
        torch_out = {"output": model(input_image)}
    check_onnx_output(onnx_filename, input_image, torch_out)
    print("模型输出一致")
    onnx_model = onnx.load(onnx_filename)
    onnx.checker.check_model(onnx_model)
    print("模型测试成功")
    return onnx_model


if __name__ == '__main__':
    model_path = 'model_data/MAnet/last_epoch_weights.pth'
    onnx_path = os.path.split(model_path)[0] + '/'
    device = 'cpu'
    infer_path = "test_500size/1/"
    VOCdevkit_path = 'VOCdevkit/' + infer_path + '1_40.jpg'

    img = Image.open(VOCdevkit_path)
    img = cvtColor(img)
    img  = np.expand_dims(np.transpose(preprocess_input(np.array(img, np.float32)), (2, 0, 1)), 0)
    img = torch.from_numpy(img)

    net = MAnet()
    net.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    net = net.eval()
    out = net(img)
    print(out)

    # torch.onnx.export(net, img, onnx_path + "torch.onnx", verbose=True ,input_names=["input"], output_names=["output"], opset_version=11)

    # traced_cpu = torch.jit.trace(net, img)
    # torch.jit.save(traced_cpu, onnx_path + "cpu1.pt")

    # 检测导出的onnx模型是否完整
    onnx_name = onnx_path + "torch.onnx"
    onnx_model = check_onnx_model(net, onnx_name, img)