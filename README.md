# QLoRA 微調 - YeungNLP BLOOM (100K 資料集)

使用 QLoRA (Quantized Low-Rank Adaptation) 方法對 YeungNLP BLOOM 模型進行參數高效微調。

## 📋 專案概述

本專案採用 QLoRA 微調技術，使用 100K Human-Assistant 對話資料集，在有限的 GPU 資源下高效訓練中文對話模型。

## 🎯 技術亮點

- **QLoRA**: 4-bit 量化 + LoRA，大幅降低顯存需求
- **基礎模型**: YeungNLP/bloom-1b1-zh (中文優化版)
- **資料規模**: 100K 對話樣本
- **硬體支援**: T4 GPU / DGX-1

## 📊 資料集

### 訓練資料
- **資料來源**: BELLE Dialog 100K (YeungNLP 版本)
- **訓練集**: `train_dataset_belle_100k_YeungNLP/data_train/` (351MB)
- **驗證集**: `train_dataset_belle_100k_YeungNLP/data_val/` (7MB)

### 資料集結構
```
train_dataset_belle_100k_YeungNLP/
├── data_train/
│   ├── data-00000-of-00001.arrow (351MB)
│   ├── dataset_info.json
│   └── state.json
└── data_val/
    ├── data-00000-of-00001.arrow (7MB)
    ├── dataset_info.json
    └── state.json
```

## 📓 核心檔案

### 1. QLoRA 微調訓練
**10-10-qlora-finetune bloomz-YeungNLP-T4-支援DGX1.ipynb** (57KB)
- QLoRA 配置與初始化
- 4-bit 量化設定
- LoRA 適配器訓練
- 支援 T4 和 DGX-1 GPU

### 2. 模型合併
**10-15-合併lora與基礎模型.ipynb** (2.6KB)
- LoRA 權重與基礎模型合併
- 生成完整推理模型
- 模型導出與保存

### 3. Gradio 應用

#### v1: QLoRA 聊天模式
**gradio-app-v1-qlora-chatbot對話模式-自訂streaming-可調整多個參數.py** (14KB)
- 即時串流輸出
- 可調整生成參數 (temperature, top_p, top_k)
- 自訂對話歷史管理

#### v2: 合併模型推理
**gradio-app-v2-for-merged-model-gr對話式與TransformersTextIteratorStreamer.py** (7.2KB)
- 使用合併後的完整模型
- TextIteratorStreamer 實現
- 優化的對話介面

## 🤖 訓練好的模型

### LoRA 適配器
`my-lora-model-1epoch-YeungNLP/`
- **adapter_model.bin**: 97MB (LoRA 權重)
- **adapter_config.json**: LoRA 配置

### 合併後的完整模型
`my-lora-merged-model-steps640-YeungNLP/`
- **pytorch_model.bin**: 667MB (完整模型)
- **config.json**: 模型配置
- **tokenizer.json**: 2.1MB (分詞器)
- **generation_config.json**: 生成配置

## ⚙️ QLoRA 配置

```python
# LoRA 配置
lora_config = {
    "r": 8,                    # LoRA rank
    "lora_alpha": 32,          # LoRA alpha
    "target_modules": ["query_key_value"],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM"
}

# 4-bit 量化配置
bnb_config = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": "float16",
    "bnb_4bit_use_double_quant": True
}
```

## 🚀 使用方式

### 1. QLoRA 訓練
```bash
# 在 T4 或 DGX-1 上執行
jupyter notebook 10-10-qlora-finetune\ bloomz-YeungNLP-T4-支援DGX1.ipynb
```

### 2. 合併模型
```bash
jupyter notebook 10-15-合併lora與基礎模型.ipynb
```

### 3. 啟動 Gradio 介面

#### 使用 LoRA 模型
```bash
python gradio-app-v1-qlora-chatbot對話模式-自訂streaming-可調整多個參數.py
```

#### 使用合併後的模型
```bash
python gradio-app-v2-for-merged-model-gr對話式與TransformersTextIteratorStreamer.py
```

## 📈 訓練效果

- **訓練輪數**: 1 epoch
- **Checkpoint**: Step 640
- **顯存需求**: ~10GB (QLoRA) vs ~40GB (Full Fine-tuning)
- **訓練時間**: 約 2-3 小時 (T4 GPU)

## 💾 顯存優化

| 方法 | 顯存需求 | 訓練速度 |
|------|---------|---------|
| Full Fine-tuning | ~40GB | 快 |
| QLoRA (本專案) | ~10GB | 中等 |
| LoRA (8-bit) | ~15GB | 較快 |

## 🔧 環境需求

```bash
transformers >= 4.30.0
peft >= 0.4.0
bitsandbytes >= 0.40.0
accelerate >= 0.20.0
gradio >= 3.35.0
torch >= 2.0.0
```

## 🎨 Gradio 介面特色

- 📊 **可調參數**: Temperature, Top-p, Top-k, Repetition Penalty
- 🔄 **串流輸出**: 即時生成文字
- 💬 **對話歷史**: 支援多輪對話
- 🌏 **簡繁轉換**: 自動處理繁簡體中文
- 📝 **範例問題**: 預設多個示範問題

## ⚠️ 注意事項

1. **GPU 需求**: 至少 10GB 顯存 (T4/V100/A100)
2. **CUDA 版本**: 需要 CUDA 11.7+
3. **量化支援**: 需要安裝 bitsandbytes (僅支援 Linux)
4. **模型路徑**: 確認 Gradio 腳本中的模型路徑正確

## 📚 參考資源

- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [YeungNLP Models](https://huggingface.co/YeungNLP)

---
*訓練日期: 2023-12 ~ 2024-01*
*支援硬體: T4, DGX-1*
