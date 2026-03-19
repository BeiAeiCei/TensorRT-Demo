#pragma once
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <string>

class Plugin : public nvinfer1::IPluginV2DynamicExt {
protected:
    std::string mNamespace;

    // 构造函数
    Plugin() = default;
    Plugin(const void* data, size_t length) {
        (void)data;
        (void)length;
    }

public:
    ~Plugin() override = default;

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept = 0;

    const char* getPluginType() const noexcept = 0;
    const char* getPluginVersion() const noexcept = 0;
    int getNbOutputs() const noexcept = 0;

    nvinfer1::DimsExprs getOutputDimensions(
        int outputIndex,
        const nvinfer1::DimsExprs* inputs,
        int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept = 0;

    bool supportsFormatCombination(
        int pos,
        const nvinfer1::PluginTensorDesc* inOut,
        int nbInputs,
        int nbOutputs) noexcept = 0;

    nvinfer1::DataType getOutputDataType(
        int index,
        const nvinfer1::DataType* inputTypes,
        int nbInputs) const noexcept = 0;

    size_t getSerializationSize() const noexcept = 0;
    void serialize(void* buffer) const noexcept = 0;

    int initialize() noexcept = 0;
    void terminate() noexcept = 0;
    void destroy() noexcept = 0;

    size_t getWorkspaceSize(
        const nvinfer1::PluginTensorDesc* inputs,
        int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs,
        int nbOutputs) const noexcept = 0;

    int enqueue(
        const nvinfer1::PluginTensorDesc* inputDesc,
        const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs,
        void* const* outputs,
        void* workspace,
        cudaStream_t stream) noexcept = 0;

    void setPluginNamespace(const char* pluginNamespace) noexcept {
        mNamespace = pluginNamespace;
    }

    const char* getPluginNamespace() const noexcept {
        return mNamespace.c_str();
    }
};