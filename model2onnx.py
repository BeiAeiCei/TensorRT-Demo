import torch
from torch.nn import functional as F
import numpy as np
import os
from transformers import BertTokenizer, BertForMaskedLM

# import onnxruntime as ort
import transformers

print("pytorch:", torch.__version__)
# print("onnxruntime version:", ort.__version__)
# print("onnxruntime device:", ort.get_device())
print("transformers:", transformers.__version__)

# BERT 模型本地保存路径
BERT_PATH = 'bert-base-uncased'


def download_bert_model():
    """自动下载 bert-base-uncased 模型并保存到本地"""
    print(f"本地未找到 {BERT_PATH} 模型，开始自动下载...")
    # 下载 Tokenizer 并保存
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.save_pretrained(BERT_PATH)
    # 下载 Masked LM 版本的 BERT 模型并保存
    model = BertForMaskedLM.from_pretrained("bert-base-uncased", return_dict=True)
    model.save_pretrained(BERT_PATH)
    print(f"模型下载完成，已保存到 {os.path.abspath(BERT_PATH)} 目录")


def model_test(model, tokenizer, text):
    print("==============model test===================")
    encoded_input = tokenizer(text, return_tensors="pt")
    mask_index = torch.where(encoded_input["input_ids"][0] == tokenizer.mask_token_id)

    output = model(**encoded_input)
    print(f"模型输出 logits 形状: {output.logits.shape}")

    logits = output.logits
    softmax = F.softmax(logits, dim=-1)
    mask_word = softmax[0, mask_index, :]
    top_10 = torch.topk(mask_word, 10, dim=1)[1][0]
    print("model test topk10 output:")
    for token in top_10:
        word = tokenizer.decode([token])
        new_sentence = text.replace(tokenizer.mask_token, word)
        print(new_sentence)

    # 保存 inputs and output
    print("Saving inputs and output to case_data.npz ...")
    position_ids = torch.arange(0, encoded_input['input_ids'].shape[1]).int().view(1, -1)
    input_ids = encoded_input['input_ids'].int().detach().numpy()
    token_type_ids = encoded_input['token_type_ids'].int().detach().numpy()
    print(f"input_ids 形状: {input_ids.shape}")

    # 保存数据到 npz 文件
    npz_file = os.path.join(BERT_PATH, 'case_data.npz')
    np.savez(
        npz_file,
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        position_ids=position_ids.numpy(),
        logits=output.logits.detach().numpy()
    )

    # 验证保存的数据
    data = np.load(npz_file)
    print(f"保存的 input_ids: {data['input_ids']}")


def model2onnx(model, tokenizer, text):
    print("===================model2onnx=======================")
    encoded_input = tokenizer(text, return_tensors="pt")
    print(f"示例输入: {encoded_input}")

    # 模型切换到推理模式（必须，避免 Dropout 影响导出）
    model.eval()
    export_model_path = os.path.join(BERT_PATH, "model.onnx")
    opset_version = 12  # 兼容 TensorRT 的 ONNX 算子版本
    symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}  # 动态维度命名

    # 导出 ONNX 模型
    torch.onnx.export(
        model,                                            # 待导出的 PyTorch 模型
        args=tuple(encoded_input.values()),               # 示例输入（用于推断输入形状）
        f=export_model_path,                              # ONNX 保存路径
        opset_version=opset_version,                      # ONNX 算子集版本
        do_constant_folding=False,                         # 关闭常量折叠（避免 TensorRT 解析问题）
        input_names=['input_ids', 'attention_mask', 'token_type_ids'],  # 输入名（和 TensorRT 对齐）
        output_names=['logits'],                           # 输出名（和 TensorRT 对齐）
        dynamic_axes={                                     # 配置动态维度
            'input_ids': symbolic_names,
            'attention_mask': symbolic_names,
            'token_type_ids': symbolic_names,
            'logits': symbolic_names
        }
    )
    print(f"ONNX 模型导出完成，路径: {os.path.abspath(export_model_path)}")


if __name__ == '__main__':
    # 校验并自动下载模型
    if not os.path.exists(BERT_PATH):
        download_bert_model()
    else:
        print(f"本地已找到 {BERT_PATH} 模型，跳过下载")

    # 加载 Tokenizer 和 BERT 模型
    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
    model = BertForMaskedLM.from_pretrained(BERT_PATH, return_dict=True)
    
    # 示例测试文本（包含 [MASK] 占位符）
    text = "The capital of France, " + tokenizer.mask_token + ", contains the Eiffel Tower."

    # 执行模型测试（保存测试数据）
    model_test(model, tokenizer, text)
    # 导出 ONNX 模型
    model2onnx(model, tokenizer, text)