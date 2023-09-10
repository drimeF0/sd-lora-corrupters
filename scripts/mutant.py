import modules.scripts as scripts
import gradio as gr
import os
import torch
import safetensors
from safetensors.torch import save_file
import numpy as np
import random

from modules import script_callbacksч

def on_ui_tabs():
    
    with gr.Blocks(analytics_enabled=False) as mutators:
        gr.Interface(fn=get_loras, inputs=["text"], outputs="text",title="get loras")
        gr.Interface(fn=get_layers, inputs=["text"], outputs="text",title="get layers")
        gr.Interface(fn=mutate_negative, inputs=["text", "text"], outputs="text",title="to negative")
        gr.Interface(fn=mutate_random_merge, inputs=["text","text","text"], outputs="text",title="random merge")
        gr.Interface(fn=merge_by_name, inputs=["text","text","text","text"], outputs="text",title="merge by name")
        gr.Interface(fn=get_difference, inputs=["text","text","text","number"], outputs="text",title="get difference (only same size)")

        return [(mutators, "Mutator", "mutator_tab")]



def get_loras(self, path):
    if not path:
        path = "/content/something/models/Lora/"
        print("No path given, using default: " + path)
    
    loras = []
    for file in os.listdir(path):
        loras.append(file)
    return "\n".join(loras)

def get_layers(input_path):
    tensors = {}
    keys = ""
    with safetensors.safe_open(input_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    for key in tensors.keys():
        keys += key + "\n"
    return keys


def mutate_negative(input_path, output_path):
    tensors = {}
    with safetensors.safe_open(input_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
            
    for key in tensors:
        param = tensors[key]
        param = torch.neg(param)
        tensors[key] = param
    save_file(tensors, output_path)

    return "Done."

def mutate_random_merge(input_path1,input_path2,output_path):
    tensors1 = {}
    tensors2 = {}
    with safetensors.safe_open(input_path1, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors1[key] = f.get_tensor(key)
    
    with safetensors.safe_open(input_path2, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors2[key] = f.get_tensor(key)
    
    merged = 0
    for key in tensors1:
        random_value = random.choice([True,False])
        if random_value:
            param = tensors2[key]
            tensors1[key] = param
            merged = merged + 1
    save_file(tensors1, output_path)

    return f"Done, merged {merged} tensors."

def merge_by_name(input_path1,input_path2,output_path,names):
    tensors1 = {}
    tensors2 = {}
    names = names.split("\n")
    with safetensors.safe_open(input_path1, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors1[key] = f.get_tensor(key)
    
    with safetensors.safe_open(input_path2, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors2[key] = f.get_tensor(key)
    
    merged = 0
    for key in tensors1:
        if key in names:
            param = tensors2[key]
            tensors1[key] = param
            merged = merged + 1
    save_file(tensors1, output_path)

    return f"Done, merged {merged} tensors."

def get_difference(input_path1,input_path2,output_path,alpha):
    tensors1 = {}
    tensors2 = {}
    with safetensors.safe_open(input_path1, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors1[key] = f.get_tensor(key)
    
    with safetensors.safe_open(input_path2, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors2[key] = f.get_tensor(key)
    
    for key in tensors1:
        param1 = tensors1[key]
        param2 = tensors2[key]
        param = (param1 - param2) * alpha
        tensors1[key] = param
        print(param)
    save_file(tensors1, output_path)

    return f"Done."
           

  

script_callbacks.on_ui_tabs(on_ui_tabs)