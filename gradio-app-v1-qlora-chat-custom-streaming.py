
import os  
#os.environ["CUDA_VISIBLE_DEVICES"] = '4'

import torch
import gradio as gr
import re
import transformers
import peft
import traceback

from queue import Queue
from threading import Thread
import gc

# streaming source: https://github.com/lxe/cerebras-lora-alpaca

from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch
from peft import PeftModel
from transformers import AutoTokenizer
import torch


import gradio as gr
from random import randint
from opencc import OpenCC
from transformers import BloomTokenizerFast, BloomForCausalLM

from transformers import TextStreamer,TextIteratorStreamer
from threading import Thread

import torch
from transformers import GenerationConfig
s2t = OpenCC('s2t')  # convert from Simplified Chinese to Traditional Chinese
t2s = OpenCC('t2s')  # convert from  Traditional Chinese to Simplified Chinese 


# device = 'cuda' if torch.cuda.is_available() else 'cpu'

CUDA_AVAILABLE = torch.cuda.is_available()
device = torch.device("cuda" if CUDA_AVAILABLE else "cpu")





class ModelUtils(object):

    @classmethod
    def load_model(cls, model_name_or_path, load_in_4bit=False, adapter_name_or_path=None):
        # 是否使用4bit量化进行推理
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
        else:
            quantization_config = None

        # 加载base model
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            load_in_4bit=load_in_4bit,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16 if CUDA_AVAILABLE else None,
            device_map='auto',
            quantization_config=quantization_config
        )

        # 加载adapter
        if adapter_name_or_path is not None:
            model = PeftModel.from_pretrained(model, adapter_name_or_path)
            model = model.merge_and_unload()         
        
        return model


# 使用base model和adapter进行推理，无需手动合并权重

# YeungNLP效果稍差一些
model_name_or_path = 'my-pretrained-2epochs\model.safetensors'
adapter_name_or_path = 'my-lora-model-1epoch-YeungNLP'

# Langboat
# model_name_or_path = 'Langboat/bloom-389m-zh'
# adapter_name_or_path = 'my-lora-model-1epoch-langboat'

# 是否使用4bit进行推理，能够节省很多显存，但效果可能会有一定的下降
load_in_4bit = False
# 生成超参配置
max_new_tokens = 500
top_p = 0.9
temperature = 0.35
repetition_penalty = 1.0
#device = 'cuda'

# 加载模型
model = ModelUtils.load_model(
    model_name_or_path,
    load_in_4bit=load_in_4bit,
    adapter_name_or_path=adapter_name_or_path
).eval()
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
    # llama不支持fast
    use_fast=False if model.config.model_type == 'llama' else True
)


#streamer = TextStreamer(tokenizer, skip_prompt=True)
#iter_streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)


def generate_prompt(user_input_prompt):

    # 這裡可以做繁簡轉換
    user_input_prompt = t2s.convert(user_input_prompt)
    # user_input_prompt = t2s.convert(user_input_prompt)
    input_prompt_pattern = 'Human: {}\nAssitant: \n\n' #格式須注意
    # input_prompt_pattern = '<s>{}</s>' #格式須注意 #
    # result_prompt = f"Human: \n{instruction} {input_text}\nAssistant: \n\n{output_text}"
    user_input_prompt = input_prompt_pattern.format(user_input_prompt)
    return user_input_prompt

'''

def generate(input_text):
    prompt = generate_prompt(input_text)
    print(prompt)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=0.1,
        top_p=0.75,
        top_k=2,
        # num_beams=num_beams,
        max_new_tokens=150, # max_length=max_new_tokens+input_sequence
        # min_new_tokens=min_new_tokens, # min_length=min_new_tokens+input_sequence
        repetition_penalty=7.0,
        do_sample=True,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=3,
        output_scores=False,
        return_dict_in_generate=True,
    )
    with torch.no_grad():
        # Run the generation in a separate thread, so that we can fetch the generated text in a non-blocking way.
        generation_kwargs = dict(input_ids=input_ids, streamer=iter_streamer, generation_config=generation_config)
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        # 
        generated_text = ""
        for new_text in iter_streamer:
            generated_text += new_text
            # print(generated_text)
            yield generated_text
        print(generated_text)
        yield generated_text.split('</s>')[0].strip()
        #yield generated_text.split("### Response:")[1].strip()
'''

examples = [
    ["哈利波特是什麼？"],
    ["法國的首都是什麼?"],
    ["英國的首都是什麼?"],
    ["中國的首都是什麼?"],
    ["你能不能詳細介紹一下怎麼做披薩？"],
    ["'下雨天'，請生成一篇文章。"],
    ["請生成一篇關於'下雨天'的文章。"],
    ["請生成一個新聞標題，描述一場正在發生的大型自然災害。"],
    ["為指定的詞彙創建一個關於該詞彙的簡短解釋。\n“人工智慧”"],
    ["編寫一篇簡短的新聞稿。\n\n\n新聞標題：一隻熊闖入市中心並在一棵樹上小憩\n"],
    ["列出3個不同的機器學習演算法，並說明它們的適用範圍。\n\n"],
    ["你是一個資深導遊，你能介紹一下中國的首都嗎"],
    ["生成一篇關於人工智慧的200字文章，簡單介紹人工智慧的起源、應用和發展前景。\n"],
    ["針對給定的文本，生成一個摘要。摘要長度應該在100到200個字元之間。\n\n\n以下是一篇新聞報導的全文：\n北京時間7月23日消息，穀歌母公司Alphabet於當地時間週四發佈了該公司第二季度財報，超過了華爾街分析師的預期，一份財報顯示，Alphabet的二季度收入達到了382.1億美元，三個月淨利潤為72億美元，大幅超過市場預期，這主要歸功於線上廣告業務的快速增長。\n"],
    ]


# Streaming functionality taken from https://github.com/oobabooga/text-generation-webui/blob/master/modules/text_generation.py#L105

class Stream(transformers.StoppingCriteria):
    def __init__(self, callback_func=None):
        self.callback_func = callback_func

    def __call__(self, input_ids, scores) -> bool:
        if self.callback_func is not None:
            self.callback_func(input_ids[0])
        return False

class Iteratorize:
    """
    Transforms a function that takes a callback
    into a lazy iterator (generator).
    """
    def __init__(self, func, kwargs={}, callback=None):
        self.mfunc=func
        self.c_callback=callback
        self.q = Queue()
        self.sentinel = object()
        self.kwargs = kwargs
        self.stop_now = False

        def _callback(val):
            if self.stop_now:
                raise ValueError
            self.q.put(val)

        def gentask():
            try:
                ret = self.mfunc(callback=_callback, **self.kwargs)
            except ValueError:
                traceback.print_exc()
                pass
            except:
                traceback.print_exc()
                pass

            clear_torch_cache()
            self.q.put(self.sentinel)
            if self.c_callback:
                self.c_callback(ret)

        self.thread = Thread(target=gentask)
        self.thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        obj = self.q.get(True,None)
        if obj is self.sentinel:
            raise StopIteration
        else:
            return obj

    def __del__(self):
        clear_torch_cache()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_now = True
        clear_torch_cache()

def clear_torch_cache():
    gc.collect()
    if CUDA_AVAILABLE:
        torch.cuda.empty_cache()
'''
[['法國的首都是什麼?', None]]
--------
 Human: 法國的首都是什麼?

Assistant:
[['法國的首都是什麼?', '法国的首都是巴黎。'], ['英國的首都是什麼?', None]]
--------
 Human: 法國的首都是什麼?

Assistant: 法国的首都是巴黎。

Human: 英國的首都是什麼?

Assistant:
'''

def generate_text(
    history,  
    max_new_tokens, 
    do_sample, 
    temperature, 
    top_p, 
    top_k, 
    repetition_penalty, 
    typical_p, 
    num_beams
):
    # Create a conversation context of the last 4 entries in the history
    # [[ , None]] -> [[ , ], [ , None]] -> [[ , ], [ , ], [ , None]]  history會越串越長
    print("history:\n",history)
    
    # 最多只取4組對話
    inp = ''.join([
        f"Human: {h[0]}\n\nAssistant: {'' if h[1] is None else h[1]}\n\n" for h in history[-4:]
    ]).strip()
    
    #轉成簡體字
    inp = t2s.convert(inp)
    # inp = ''.join([
    #     f"Human: {h[0]}\n\nAssistant: {'' if h[1] is None else h[1]}\n\n" for h in history[-4:]
    # ]).strip()
    
    print("input prompt: --------\n",inp) 
    
    input_ids = tokenizer.encode(
        inp,
        return_tensors='pt', 
        # truncation=True, 
        add_special_tokens=False
    ).to(device) # type: ignore

    generate_params = {
        "input_ids": input_ids,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "typical_p": typical_p,
        "num_beams": num_beams,
        "stopping_criteria": transformers.StoppingCriteriaList(),
        "pad_token_id": tokenizer.pad_token_id,
    }
    
    def generate_with_callback(callback=None, **kwargs):
        kwargs['stopping_criteria'].append(Stream(callback_func=callback))
        clear_torch_cache()
        with torch.no_grad():
            model.generate(**kwargs) # type: ignore

    def generate_with_streaming(**kwargs):
        return Iteratorize(generate_with_callback, kwargs, callback=None)

    with generate_with_streaming(**generate_params) as generator:
        for output in generator:
            new_tokens = len(output) - len(input_ids[0])
            reply = tokenizer.decode(output[-new_tokens:], skip_special_tokens=True)

            # If reply contains '^Human:' or '^Assistant:' 
            # then we have reached the end of the assistant's response
            stop_re = re.compile(r'^(Human|Assistant):', re.MULTILINE)
            if re.search(stop_re, reply):
                reply = ''.join(reply.split('\n')[:-1])
                history[-1][1] = reply.strip()
                yield history
                break

            # if reply contains 'EOS' then we have reached the end of the conversation
            if output[-1] in [tokenizer.eos_token_id]:
                yield history
                break

            history[-1][1] = reply.strip()
            yield history

with gr.Blocks() as demo:
    gr.Markdown("# 自己微調的小型ChatGPT\n微調GPT，讓它可以回答各式各樣的問題，就像人類對話一般。")
    with gr.Row():
        with gr.Column():
            chatbot = gr.Chatbot()

            msg = gr.Textbox(value="深度學習是甚麼?", placeholder="Type a message...")
            with gr.Row():
                clear = gr.Button("Clear")
                submit = gr.Button("Submit")

        with gr.Column():
            some_examples = gr.Examples(examples, inputs=[msg])
        with gr.Column():
            max_new_tokens = gr.Slider(10, 600, 200, step=1, label="max_new_tokens")
            do_sample = gr.Checkbox(True, label="do_sample")
            with gr.Row():
                with gr.Column():
                    temperature = gr.Slider(0, 2, 0.1, step=0.01, label="temperature")
                    top_p = gr.Slider(0, 1, 0.8, step=0.01, label="top_p")
                    top_k = gr.Slider(0, 100, 35, step=1, label="top_k")
                with gr.Column():
                    repetition_penalty = gr.Slider(0, 10, 1.1, step=0.01, label="repetition_penalty")
                    typical_p = gr.Slider(0, 1, 1, step=0.01, label="typical_p")
                    num_beams = gr.Slider(1, 6, 1, step=1, label="num_beams")
            
    def user(user_message, history):
        return "", history + [[user_message, None]]
    
    def fix_history(history):
        update_history = False
        for i, (user, bot) in enumerate(history):
            if bot is None:
                update_history = True
                history[i][1] = "_silence_"
        if update_history:
            chatbot.update(history) 

    submit.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        generate_text, inputs=[
            chatbot,
            max_new_tokens, 
            do_sample, 
            temperature, 
            top_p, 
            top_k, 
            repetition_penalty, 
            typical_p, 
            num_beams
        ], outputs=[chatbot],
    ).then(fix_history, chatbot)
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        generate_text, inputs=[
            chatbot,
            max_new_tokens, 
            do_sample, 
            temperature, 
            top_p, 
            top_k, 
            repetition_penalty, 
            typical_p, 
            num_beams
        ], outputs=[chatbot],
    ).then(fix_history, chatbot)

    clear.click(lambda: None, None, chatbot, queue=False)

demo.queue().launch()
