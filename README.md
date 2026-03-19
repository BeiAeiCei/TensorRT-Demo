
# TensorRT-Demo

### 一. 文件信息
1. model2onnx.py 使用pytorch 运行 bert 模型，生成demo 输入输出和onnx模型
2. Myplugin 存放我的Plugin
3. build.sh 脚本，编译并注册自定义Plugin
4. Replace.py 替换onnx模型内的算子为自定义Plugin算子
5. Generateengine.sh 脚本，调用 trtexec 生成base和Plugin替换版本
6. infer.sh 运行trt随机生成的数据来进行推理（检验自定义是否成功）
7. comparewithbase.py 检验精度和推理效率
8. 基础款LayerNormPlugin.zip 用于学习的layer_norm_plugin

## 二. 模型信息
### 2.1 介绍
1. 标准BERT 模型，12 层, hidden_size = 768
2. 不考虑tokensizer部分，输入是ids，输出是score
3. 为了更好的理解，降低作业难度，将mask逻辑去除，只支持batch=1 的输入

BERT模型可以实现多种NLP任务，作业选用了fill-mask任务的模型

```
输入：
The capital of France, [mask], contains the Eiffel Tower.

topk10输出：
The capital of France, paris, contains the Eiffel Tower.
The capital of France, lyon, contains the Eiffel Tower.
The capital of France,, contains the Eiffel Tower.
The capital of France, tolilleulouse, contains the Eiffel Tower.
The capital of France, marseille, contains the Eiffel Tower.
The capital of France, orleans, contains the Eiffel Tower.
The capital of France, strasbourg, contains the Eiffel Tower.
The capital of France, nice, contains the Eiffel Tower.
The capital of France, cannes, contains the Eiffel Tower.
The capital of France, versailles, contains the Eiffel Tower.
```

### 2.2 输入输出信息
输入
1. input_ids[1, -1]： int 类型，input ids，从BertTokenizer获得
2. token_type_ids[1, -1]：int 类型，全0
3. position_ids[1, -1]：int 类型，[0, 1, ..., len(input_ids) - 1]

输出
1. logit[1, -1, 768]

### 三. 插件练习

1. Add Plugin 示例

    - 实现了一个简单的加法操作的 Plugin，用于理解 TensorRT Plugin 的创建、注册和使用

    - 当前实现的实例 id: 68

    - 后续可以在 ONNX 转 TRT 或 trtexec 推理时替换对应的层

2. LayerNorm Plugin 练习

    - 下一步练习目标：实现 BERT 中 LayerNorm 操作的 Plugin

    - 可以参考 基础款 LayerNormPlugin.zip 来理解接口和序列化方式

四. 项目说明

本项目来源于 https://github.com/shenlan2017/TensorRT

由于原项目时间较久远，本项目对部分代码进行了重构
本项目仅仅作为个人 TensorRT Plugin 学习练习

