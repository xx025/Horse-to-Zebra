import gradio as gr

import numpy as np
import onnxruntime
from torchvision import transforms

onxx_path = r"data\horse2zebra_0.4.0.onnx"

# Load the ONNX model
ort_session = onnxruntime.InferenceSession(
    onxx_path,
    providers=[
        'CUDAExecutionProvider',  # 需要CuDNN
        'CPUExecutionProvider'
    ]
)


def preprocess(img, resize_h, resize_w):
    preprocess_transform = transforms.Compose([
        transforms.Resize((resize_h, resize_w)),
        # transforms.ToTensor(),
    ])
    return preprocess_transform(img)


# 创建一个推理函数

def inference(image, resize_h=250, resize_w=250):
    img = image

    img_t = preprocess(img, resize_h, resize_w)
    # 使用numpy array
    img_t = np.array(img_t).astype(np.float32)  # H,W,C
    img_t = img_t.transpose(2, 0, 1)  # H,W,C -> C,H,W
    batch_t = np.expand_dims(img_t, axis=0)  # B,C,H,W
    batch_t = batch_t / 255.0  # 归一化

    # Run the model on the input image
    ort_inputs = {ort_session.get_inputs()[0].name: batch_t}

    ort_outs = ort_session.run(None, ort_inputs)

    # Postprocess the output
    img_out_t2 = ort_outs[0][0]  # [(B,C,H,W),....]  -> C,H,W
    img_out_t = np.transpose(img_out_t2, (1, 2, 0))  # C,H,W -> H,W,C
    out_t = (img_out_t.squeeze() + 1.0) / 2.0  # [-1,1] -> [0,1]

    # Display the result
    out_img = transforms.ToPILImage()(out_t)
    return out_img


# 创建 Gradio 接口
demo = gr.Interface(
    title="Horse to Zebra",
    fn=inference,
    inputs=[
        gr.Image(type="pil", label="上传图片"),
        gr.Number(value=250, label="宽度"),
        gr.Number(value=250, label="高度")
    ],
    outputs=gr.Image(type="pil", label="输出图片", width=250, height=250),
    examples=[
        ["data/horse.jpg"],
        ["data/horse2.png"]
    ]
)

# 启动 Gradio 接口
demo.launch()
