
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from threading import Thread
from transformers import GenerationConfig

from opencc import OpenCC
s2t = OpenCC('s2t')  # convert from Simplified Chinese to Traditional Chinese
t2s = OpenCC('t2s')  # convert from  Traditional Chinese to Simplified Chinese 

# https://www.gradio.app/guides/creating-a-chatbot-fast
# tokenizer = AutoTokenizer.from_pretrained("togethercomputer/RedPajama-INCITE-Chat-3B-v1")
# model = AutoModelForCausalLM.from_pretrained("togethercomputer/RedPajama-INCITE-Chat-3B-v1", torch_dtype=torch.float16)

# model = model.to('cuda:0')

CUDA_AVAILABLE = torch.cuda.is_available()
device = torch.device("cuda" if CUDA_AVAILABLE else "cpu")

model_name_or_path = 'my-lora-merged-model-steps640-YeungNLP'

# For CPU環境，必須是float32
model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)

# For GPU環境，是float16格式，模型比較小
#model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype='auto').to(device)

tokenizer = AutoTokenizer.from_pretrained(
    'YeungNLP/bloomz-396m-zh',
    trust_remote_code=True,
    # llama不支持fast
    use_fast=True
)


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [tokenizer.eos_token_id]
        #stop_ids = [29, 0]
        # if reply contains 'EOS' then we have reached the end of the conversation
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

def predict(message, history):

    history_transformer_format = history + [[message, ""]]
    stop = StopOnTokens()

    messages = "".join(["".join(["<Human>:"+item[0], "\n\n<Assistant>:"+item[1]])  #curr_system_message +
                for item in history_transformer_format])
    
    messages = t2s.convert(messages)
    # 最多只取4組對話
    # inp = ''.join([
    #     f"Human: {h[0]}\n\nAssistant: {'' if h[1] is None else h[1]}\n\n" for h in history[-4:]
    # ]).strip()
    
    #轉成簡體字
    #inp = t2s.convert(inp)
    
    #print(messages)
    
    model_inputs = tokenizer([messages], return_tensors="pt", add_special_tokens=False).to(device)
    # model_inputs = tokenizer([messages], return_tensors="pt").to(device)
    
    streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True) #似乎沒有差別?
    #streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)
    
    
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens= 250, #1024
        do_sample=True,
        top_p=0.95,
        top_k=200, #1000,
        temperature=1.0,
        # repetition_penalty=7, #會報錯 不知原因
        num_beams=1, #目前只支援 =1
        stopping_criteria=StoppingCriteriaList([stop])
        )
    
    '''
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
    '''
    
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    #-----------------
    partial_message  = ""
    for new_token in streamer:
        
        partial_message += s2t.convert(new_token)
        # print(new_token)
        yield partial_message
        
        # 可在此額外設定跳開字元 似乎有許多<>這樣的符號會在文字中，不知原因
        # if new_token != '<':
        #     partial_message += new_token
        #     yield s2t.convert(partial_message)

    #-----------



examples = [
    ["介紹一下哈利波特是什麼？"],
    ["法國的首都是什麼?"],
    ["英國的首都是什麼?"],
    ["中國的首都是什麼?"],
    ["你能不能詳細介紹一下怎麼做披薩？"],
    ["'下雨天'，請生成一篇文章。"],
    ["請生成一篇關於'下雨天'的文章。"],
    ["請生成一個新聞標題，描述一場正在發生的大型自然災害。"],
    ["為指定的詞彙創建一個關於該詞彙的簡短解釋。\n“人工智慧”"],
    ["編寫一篇簡短的新聞稿。\n新聞標題：一隻熊闖入市中心並在一棵樹上小憩"],
    ["列出3個不同的機器學習演算法，並說明它們的適用範圍。"],
    ["你是一個資深導遊，你能介紹一下中國的首都嗎"],
    ["生成一篇關於人工智慧的200字文章，簡單介紹人工智慧的起源、應用和發展前景。\n"],
    ["針對給定的文本，生成一個摘要。摘要長度應該在100到200個字元之間。\n以下是一篇新聞報導的全文：\n北京時間7月23日消息，穀歌母公司Alphabet於當地時間週四發佈了該公司第二季度財報，超過了華爾街分析師的預期，一份財報顯示，Alphabet的二季度收入達到了382.1億美元，三個月淨利潤為72億美元，大幅超過市場預期，這主要歸功於線上廣告業務的快速增長。"],
    ]


# gr.ChatInterface(predict).queue().launch()
title = "自己微調的小型ChatGPT"
description = "微調GPT，讓它可以回答各式各樣的問題，就像人類對話一般"

            
chatinterface = gr.ChatInterface(fn=predict,
                                    examples=examples,
                                    title=title,
                                    description=description,
                                    # cache_examples=True,
                                    textbox=gr.Textbox(value="請告訴我深度學習是甚麼?", placeholder="Ask me a question", container=False, lines=1, scale=5),
                                    
                                    #submit_btn=gr.Button("確定", scale=3),
                                    #stop_btn=gr.Button("中斷", scale=3),
                                    theme="soft",
                                    retry_btn="再產生一次答案",
                                    undo_btn="刪除最後一次對談",
                                    clear_btn="新的交談")               
chatinterface.queue()
chatinterface.launch()
#chatinterface.launch(server_name='0.0.0.0',server_port=8001)

# https://www.gradio.app/docs/chatinterface

'''
gr.ChatInterface(
    yes_man,
    chatbot=gr.Chatbot(height=300),
    textbox=gr.Textbox(placeholder="Ask me a yes or no question", container=False, scale=7),
    title="Yes Man",
    description="Ask Yes Man any question",
    theme="soft",
    examples=["Hello", "Am I cool?", "Are tomatoes vegetables?"],
    cache_examples=True,
    retry_btn=None,
    undo_btn="Delete Previous",
    clear_btn="Clear",
).launch()
'''