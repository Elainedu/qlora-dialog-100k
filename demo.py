"""
===============================================================================
專案名稱: QLoRA Chinese Dialog System - 參數高效微調對話系統
===============================================================================

[專案簡介]
這是一個使用 QLoRA (Quantized Low-Rank Adaptation) 技術微調的中文對話系統。
QLoRA 結合了量化技術和 LoRA 參數高效微調方法，只需訓練極少量參數（~0.1%）
就能達到接近全量微調的效果，大幅降低記憶體需求和訓練成本。

[核心技術]
- 微調方法: QLoRA (4-bit Quantization + LoRA)
- 量化技術: NF4 (4-bit NormalFloat) Quantization
- 參數高效: PEFT (Parameter-Efficient Fine-Tuning)
- 基礎模型: BLOOMZ-396M-ZH (中文語言模型)
- 深度學習框架: PyTorch + Transformers + PEFT
- Web 介面: Gradio

[QLoRA 技術優勢]
相比傳統全量微調 (Full Fine-tuning):
1. 記憶體使用: 降低約 75% (4-bit 量化 + 凍結大部分參數)
2. 訓練速度: 提升約 50% (只更新少量參數)
3. 儲存空間: 只需儲存 adapter 權重 (~97MB vs 1.5GB)
4. 模型效果: 相近的對話品質
5. 可訓練參數: 僅 6.7% (約 25M / 396M 參數)

[LoRA 原理]
LoRA (Low-Rank Adaptation) 的核心概念:
- 凍結預訓練模型的所有參數
- 在特定層插入低秩矩陣 (rank-r decomposition)
- 只訓練這些低秩矩陣的參數
- 公式: W' = W + BA，其中 B 和 A 是可訓練的低秩矩陣
- rank=8 表示分解維度，alpha=32 是縮放因子

[訓練資料]
- 資料集: BELLE Dialog 100K（中文對話資料）
- 對話格式: Human-Assistant 多輪對話
- 資料處理: Instruction tuning format
- 訓練樣本: 100,000 對話

[訓練配置]
- LoRA rank: 8
- LoRA alpha: 32
- LoRA dropout: 0.05
- 量化: 4-bit NF4
- 訓練 epochs: 3
- Final Loss: 0.77
- GPU 記憶體: ~8GB (vs ~24GB for full fine-tuning)

[兩種模式]
1. Adapter 模式:
   - 載入基礎模型 + LoRA adapter
   - 檔案大小: 667MB (base) + 97MB (adapter)
   - 適合: 多個 adapter 切換使用

2. Merged 模式:
   - adapter 權重已合併到基礎模型
   - 檔案大小: 667MB (單一檔案)
   - 適合: 部署和分享

[啟動方式]
使用 Adapter 模式（基礎模型 + LoRA adapter）:
    python demo.py --mode adapter

使用 Merged 模式（合併後的完整模型）:
    python demo.py --mode merged

指定自訂模型路徑:
    python demo.py --mode adapter --model path/to/adapter

產生公開分享連結:
    python demo.py --share

指定 Port:
    python demo.py --port 8080

[使用說明]
1. 啟動後開啟 http://127.0.0.1:7860
2. 在訊息框輸入問題或指令
3. 點擊「發送」或按 Enter
4. 系統會根據對話歷史生成回應
5. 可調整參數：
   - Temperature: 控制回應的創造性（0.1-1.5）
   - Top-p: 核採樣參數（0.1-1.0）
   - Top-k: Top-k 採樣參數（10-100）

[訓練模型]
執行 QLoRA 訓練:
    python train_qlora.py

訓練完成後會生成:
- models/adapter/: LoRA adapter 權重
- models/merged/: 合併後的完整模型

[面試展示重點]
1. **QLoRA 原理**: 說明 4-bit 量化 + LoRA 的技術組合
2. **參數效率**: 強調只訓練 6.7% 參數達到相近效果
3. **記憶體優化**: 解釋如何用消費級 GPU 訓練大型模型
4. **實際應用**: 討論在資源受限環境下的模型微調策略
5. **技術權衡**: LoRA rank 的選擇、量化對精度的影響

[對話範例]
輸入: "台灣有哪些著名景點？"
輸出: "台灣有許多著名景點，例如台北101、日月潭、阿里山、墾丁國家公園..."

輸入: "如何學習深度學習？"
輸出: "學習深度學習建議從以下步驟開始：1. 學習 Python 基礎 2. 了解數學基礎..."

[檔案結構]
demo.py                          # 本檔案 - Gradio 對話介面
train_qlora.py                   # QLoRA 訓練程式
models/
    ├── adapter/                 # LoRA adapter 權重 (~97MB)
    │   ├── adapter_config.json
    │   └── adapter_model.bin
    └── merged/                  # 合併後的模型 (~667MB)
        ├── config.json
        ├── pytorch_model.bin
        └── tokenizer files

[技術細節]
QLoRA 的關鍵創新:
1. 4-bit NormalFloat (NF4) 量化 - 特別為正態分佈的權重設計
2. Double Quantization - 對量化常數再次量化，節省更多記憶體
3. Paged Optimizers - 使用 CPU RAM 作為 GPU 記憶體的後備

LoRA 層配置:
- target_modules: ["query_key_value"] (只對注意力層應用 LoRA)
- 其他層保持凍結，大幅減少可訓練參數

[與全微調的比較]
指標              | 全微調    | QLoRA
-----------------|---------|--------
記憶體使用        | ~24GB   | ~8GB
訓練速度          | 基準     | +50%
可訓練參數        | 100%    | 6.7%
模型儲存          | 1.5GB   | 97MB (adapter)
效果              | 基準     | ~95-98%

[開發者]
碩士班課程專案 - 深度學習（進階）
建立日期: 2024
更新日期: 2026-03-10 (修正 emoji 編碼問題)

===============================================================================
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
        print(f"[LOAD] 載入模型 ({mode} 模式): {model_path}")

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
            print("[OK] LoRA Adapter 載入完成")

        else:  # merged
            # 直接載入合併後的模型
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map='auto'
            )
            print("[OK] 合併模型載入完成")

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
        - [TARGET] 只訓練 6.7% 的參數達到相近效果
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
        print(f"[ERROR] 找不到模型: {model_path}")
        print(f"請先執行訓練: python train_qlora.py")
        return

    # 建立並啟動 demo
    print(f"[START] 啟動 Gradio Demo ({args.mode} 模式)...")
    demo = create_demo(model_path, args.mode)
    demo.launch(
        share=args.share,
        server_port=args.port,
        server_name="0.0.0.0"
    )


if __name__ == '__main__':
    main()
