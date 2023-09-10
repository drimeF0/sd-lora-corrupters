import modules.scripts as scripts
import gradio as gr
import os
import torch
import safetensors
from safetensors.torch import save_file
import numpy as np
import random

from modules import script_callbacks


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as ui_component:
        gr.Interface(fn=corrupt_all, inputs=["text","text"] outputs="text",title="corrupt all")

        return [(ui_component, "Corrupter", "corrupter_tab")]

def corrupt_all(input_path,output_path):
    tensors = {}
    with safetensors.safe_open(input_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    for key in tensors:
        param = tensors[key]
        noise = torch.from_numpy(np.random.normal(-0.01, 0.01, size=param.shape)).float()
        param += noise
    save_file(tensors, output_path)
    return "Done"

script_callbacks.on_ui_tabs(on_ui_tabs)