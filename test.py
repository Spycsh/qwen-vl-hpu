# -*- coding: utf-8 -*-
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers.generation import GenerationConfig
import torch
import time
torch.manual_seed(1234)
from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
adapt_transformers_to_gaudi()
# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

# use bf16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
# use cpu only
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="hpu", trust_remote_code=True).eval()
# use cuda device
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()

# Specify hyperparameters for generation (No need to do this if you are using transformers>=4.32.0)
# model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

# 1st dialogue turn
query = tokenizer.from_list_format([
    {'image': 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'},
    #{'image': './tire.png'},
    #{'text': 'What is the serial number in this image?'},
    {'text': '这是什么？'},
])
#response, history = model.chat(tokenizer, 'What is the serial number in this image?', history=history)
#print(response)
#print(time.time() - s)

# 测试无history warmup性能
for i in range(2):
    s=time.time()
    response, history = model.chat(tokenizer, query=query, history=None)
    print(response)
    print(time.time() - s)

# 测试有history warmup性能
for i in range(2):
    s=time.time()
    response, history = model.chat(tokenizer, '请画出击掌位置', history=history)
    print(response)
    print(history)
    print(time.time() - s)

#image = tokenizer.draw_bbox_on_latest_picture(response, history)
#if image:
#  image.save('1.jpg')
#else:
#  print("no box")