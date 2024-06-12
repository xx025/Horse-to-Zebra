import os
import subprocess

import torch.onnx

from model import ResNetGenerator


def Convert_ONNX(model, dummy_input=None, output_path=None):
    model.eval()
    torch.onnx.export(
        model,  # model being run
        dummy_input,  # model input (or a tuple for multiple inputs)
        f=output_path,  # where to save the model
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=10,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=['modelInput'],  # the model's input names
        output_names=['modelOutput'],  # the model's output names
        dynamic_axes={
            'modelInput': {
                0: 'batch_size',
                2: 'height',
                3: 'width'
            },  # 可变长度轴
            'modelOutput': {
                0: 'batch_size',
                2: 'height',
                3: 'width'
            }
        }  # 可变长度轴)
    )
    print(" ")
    print('Model has been converted to ONNX')


# os.environ["HTTP_PROXY"] = "socks5://127.0.0.1:7898"
# os.environ["HTTPS_PROXY"] = "socks5://127.0.0.1:7898"

if __name__ == "__main__":
    netG = ResNetGenerator()

    model_path = "data/horse2zebra_0.4.0.pth"  # 加载模型权重
    url = "https://raw.githubusercontent.com/deep-learning-with-pytorch/dlwpt-code/master/data/p1ch2/horse2zebra_0.4.0.pth"    # 下载 URL

    if not os.path.exists(model_path):
        print(f"model weights not found, downloading from {url}...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        subprocess.run(["curl", "-o", model_path, url], check=True)

    output_path = "data/horse2zebra_0.4.0.onnx"
    model_data = torch.load(model_path)
    netG.load_state_dict(model_data)
    torch_model = netG
    dummy_input = torch.randn([1, 3, 250, 250], requires_grad=False)
    Convert_ONNX(model=netG, dummy_input=dummy_input, output_path=output_path)
