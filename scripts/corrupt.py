import modules.scripts as scripts
import gradio as gr
import os
import torch
import safetensors
from safetensors.torch import save_file
import numpy as np
import random

from modules import script_callbacks

noise_multiplayer = 0.01
def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as corrupters:
        gr.Interface(fn=corrupt_all, inputs=["text","text"], outputs="text",title="corrupt all")
        gr.Interface(fn=get_layers, inputs=["text"], outputs="text",title="get layers")
        gr.Interface(fn=corrupt_by_name, inputs=["text", "text", "text"], outputs="text",title="corrupt by name")
        gr.Interface(fn=corrupt_only_one_tensor, inputs=["text", "text"], outputs="text",title="corrupt only one tensor")
        gr.Interface(fn=corrupt_only_n_tensors, inputs=["text", "number", "text"], outputs="text",title="corrupt only n tensors")
        gr.Interface(fn=set_noise_multiplayer, inputs=["number"], outputs="text",title="set noise multiplayer")


        return [(corrupters, "Corrupter", "corrupter_tab")]

def corrupt_all(input_path,output_path):
    tensors = {}
    with safetensors.safe_open(input_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    for key in tensors:
        param = tensors[key]
        noise = torch.from_numpy(np.random.normal(-noise_multiplayer, noise_multiplayer, size=param.shape)).float()
        param += noise
    save_file(tensors, output_path)
    return "Done"

def get_layers(input_path):
    tensors = {}
    keys = ""
    with safetensors.safe_open(input_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    for key in tensors.keys():
        keys += key + "\n"
    return keys

def corrupt_by_name(input_path, name, output_path):
    name = name.split("\n")
    tensors = {}

    with safetensors.safe_open(input_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    
    for key in tensors:
        if key in name:
            param = tensors[key]
            noise = torch.from_numpy(np.random.normal(-noise_multiplayer, noise_multiplayer, size=param.shape)).float()
            param += noise
    save_file(tensors,output_path)

    return "Done."

def corrupt_only_one_tensor(input_path, output_path):
    tensors = {}
    with safetensors.safe_open(input_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    keys = list(tensors.keys())
    random_key = random.choice(keys)
    param = tensors[random_key]
    noise = torch.from_numpy(np.random.normal(-noise_multiplayer, noise_multiplayer, size=param.shape)).float()
    param += noise
    save_file(tensors,output_path)

    return "Done."

def corrupt_only_n_tensors(input_path, n, output_path):
    tensors = {}
    with safetensors.safe_open(input_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    keys = list(tensors.keys())
    for i in range(n):
        random_key = random.choice(keys)
        param = tensors[random_key]
        noise = torch.from_numpy(np.random.normal(-noise_multiplayer, noise_multiplayer, size=param.shape)).float()
        param += noise
    save_file(tensors,output_path)  

    return "Done."

def set_noise_multiplayer(new_noise_multiplayer):
    global noise_multiplayer
    noise_multiplayer = new_noise_multiplayer
    return f"Noise multiplayer set to {noise_multiplayer}"

  

script_callbacks.on_ui_tabs(on_ui_tabs)