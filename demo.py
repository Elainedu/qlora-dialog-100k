"""
QLoRA Dialog Model - Gradio Demo
使用 QLoRA 微調的中文對話模型 Demo

用法:
    python demo.py --mode adapter  # 使用 LoRA adapter
    python demo.py --mode merged   # 使用合併後的模型
"""
import argparse
import os
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


class QLoRAChatbot:
    def __init__(self, model_path, mode='merged'):
        """
        初始化聊天機器人

        Args:
            model_path: 模型路徑（adapter 或 merged model）
            mode: 'adapter' 或 'merged'
        """
        print(f"📦 載入模型 ({mode} 模式): {model_path}")

        self.mode = mode

        if mode == 'adapter':
            # 載入基礎模型 + LoRA adapter
            base_model = 'YeungNLP/bloom-396m-zh'
            self.tokenizer = AutoTokenizer.from_pretrained(base_model)
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map='auto'
            )
            # 載入 LoRA adapter
            self.model = PeftModel.from_pretrained(self.model, model_path)
            print("✅ LoRA Adapter 載入完成")

        else:  # merged
            # 直接載入合併後的模型
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map='auto'
            )
            print("✅ 合併模型載入完成")

        # 設定 tokenizer
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"💡 使用裝置: {self.model.device}")

    def chat(self, message, history, max_length=512, temperature=0.8, top_p=0.9, top_k=50):
        """
        聊天函數

        Args:
            message: 用戶訊息
            history: 對話歷史 [(user, assistant), ...]
            max_length: 最大生成長度
            temperature: 溫度參數
            top_p: Top-p 採樣
            top_k: Top-k 採樣
        """
        # 構建對話 prompt
        conversation = ""
        for user_msg, assistant_msg in history:
            conversation += f"Human: {user_msg}\n"
            conversation += f"Assistant: {assistant_msg}\n"
        conversation += f"Human: {message}\n"
        conversation += "Assistant:"

        # Tokenize
        inputs = self.tokenizer(
            conversation,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        ).to(self.model.device)

        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2
            )

        # 解碼
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 提取 Assistant 的回應
        if "Assistant:" in full_response:
            response = full_response.split("Assistant:")[-1].strip()
        else:
            response = full_response

        return response


def create_demo(model_path, mode='merged'):
    """建立 Gradio 聊天介面"""
    chatbot = QLoRAChatbot(model_path, mode)

    def chat_fn(message, history, temperature, top_p, top_k):
        """聊天包裝函數"""
        if not message.strip():
            return history, ""

        response = chatbot.chat(message, history, temperature=temperature, top_p=top_p, top_k=top_k)
        history.append((message, response))
        return history, ""

    # 建立介面
    with gr.Blocks(theme=gr.themes.Soft(), title="QLoRA 中文對話") as demo:
        gr.Markdown(f"""
        # 💬 QLoRA Chinese Dialog Chatbot
        ### 參數高效微調對話模型 ({mode.upper()} 模式)

        **模型特色:**
        - 🎯 只訓練 6.7% 的參數達到相近效果
        - ⚡ 使用 4-bit 量化技術 (NF4)
        - 🗣️ 訓練資料: BELLE 100K 中文對話
        - 🧠 基礎模型: BLOOMZ-396M-ZH

        {'**當前模式:** LoRA Adapter (基礎模型 + 97MB adapter)' if mode == 'adapter' else '**當前模式:** Merged Model (完整模型 667MB)'}
        """)

        chatbox = gr.Chatbot(label="對話視窗", height=400)

        with gr.Row():
            msg = gr.Textbox(
                label="訊息",
                placeholder="輸入你的訊息...",
                scale=4
            )
            send_btn = gr.Button("發送", variant="primary", scale=1)

        with gr.Accordion("參數設定", open=False):
            temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.1, label="Temperature")
            top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
            top_k = gr.Slider(10, 100, value=50, step=5, label="Top-k")

        clear_btn = gr.Button("清除對話")

        # 範例對話
        gr.Examples(
            examples=[
                "你好，請介紹一下自己",
                "台灣有哪些著名景點？",
                "如何學習深度學習？",
                "請推薦幾本好書",
            ],
            inputs=msg,
            label="範例問題"
        )

        # 事件綁定
        send_btn.click(
            fn=chat_fn,
            inputs=[msg, chatbox, temperature, top_p, top_k],
            outputs=[chatbox, msg]
        )

        msg.submit(
            fn=chat_fn,
            inputs=[msg, chatbox, temperature, top_p, top_k],
            outputs=[chatbox, msg]
        )

        clear_btn.click(lambda: [], outputs=chatbox)

        gr.Markdown("""
        ---
        **訓練資訊:**
        - 訓練方法: QLoRA (4-bit NF4 quantization)
        - 可訓練參數: 6.7% (~25M / 396M)
        - 訓練資料: BELLE Dialog 100K
        - Final Loss: 0.77
        - LoRA rank: 8, alpha: 32

        **對比全量微調:**
        - 記憶體使用: 降低 ~75%
        - 訓練速度: 提升 ~50%
        - 效果: 相近
        """)

    return demo


def main():
    parser = argparse.ArgumentParser(description='啟動 QLoRA 對話 Demo')
    parser.add_argument('--mode', type=str, default='merged',
                       choices=['adapter', 'merged'],
                       help='模式: adapter (LoRA) 或 merged (合併)')
    parser.add_argument('--model', type=str, default=None,
                       help='模型路徑 (可選，使用預設路徑)')
    parser.add_argument('--share', action='store_true',
                       help='建立公開分享連結')
    parser.add_argument('--port', type=int, default=7860,
                       help='Port 號')

    args = parser.parse_args()

    # 決定模型路徑
    if args.model:
        model_path = args.model
    else:
        if args.mode == 'adapter':
            model_path = 'models/adapter'
        else:
            model_path = 'models/merged'

    # 檢查模型
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型: {model_path}")
        print(f"請先執行訓練: python train_qlora.py")
        return

    # 建立並啟動 demo
    print(f"🚀 啟動 Gradio Demo ({args.mode} 模式)...")
    demo = create_demo(model_path, args.mode)
    demo.launch(
        share=args.share,
        server_port=args.port,
        server_name="0.0.0.0"
    )


if __name__ == '__main__':
    main()
