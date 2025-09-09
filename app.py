# Cell
import gradio as gr
import onnxruntime as ort
import torchvision.transforms as transforms

# Cell
# Load ONNX model
session = ort.InferenceSession("fastai_model.onnx")

# Define categories manually (since ONNX doesn't store vocab)
categories = ['basset', 'beagle', 'bulldog']  # Replace with your actual classes

# Preprocessing pipeline (must match training)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                         std=[0.229, 0.224, 0.225])
])

# Cell
def classify_image(img):
    img = preprocess(img).unsqueeze(0).numpy()  # Add batch dimension
    ort_inputs = {session.get_inputs()[0].name: img}
    ort_outs = session.run(None, ort_inputs)
    probs = ort_outs[0][0]
    return dict(zip(categories, map(float, probs)))

# Cell
image = gr.Image(type='pil')
label = gr.Label()
examples = ['basset.jpg']

# Cell
intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch()
