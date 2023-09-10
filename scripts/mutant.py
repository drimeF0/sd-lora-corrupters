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
    
    with gr.Blocks(analytics_enabled=False) as mutators:
        gr.Interface(fn=mutate_negative, inputs=["text", "text"], outputs="text",title="mutate negative")
        gr.Interface(fn=mutate_random_merge, inputs=["text","text","text"], outputs="text",title="mutate random merge")
        gr.Interface(fn=mutate_add_diff, inputs=["text","text","text"], outputs="text",title="mutate add diff")

        return [(mutators, "Mutator", "mutator_tab")]



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
    

def mutate_add_diff(input_path1,input_path2,output_path):
    tensors1 = {}
    tensors2 = {}
    with safetensors.safe_open(input_path1, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors1[key] = f.get_tensor(key)
    
    with safetensors.safe_open(input_path2, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors2[key] = f.get_tensor(key)
    
    done = 0
    for key in tensors1:
        param1 = tensors1[key]
        param2 = tensors2[key]
        param = param1 - param2
        tensors1[key] = param
        done = done + 1
    save_file(tensors1, output_path)

    return f"Done, total {done} tensors."
           

  

script_callbacks.on_ui_tabs(on_ui_tabs)