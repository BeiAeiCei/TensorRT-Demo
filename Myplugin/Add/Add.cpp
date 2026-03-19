#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <NvInfer.h>
#include <string>
#include <iostream>
#include <cstring>
#include "Add.cuh"
#include "BaseStaticPlugin.h"
#include "BaseStaticIPluginCreator.h"

// ================= AddPlugin =================
class AddPlugin : public Plugin {
public:
    AddPlugin() : Plugin() {}
    AddPlugin(const void* data, size_t length) : Plugin() { (void)data; (void)length; }

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override {
        return new AddPlugin(*this);
    }

    ~AddPlugin() { terminate(); }

    const char* getPluginType() const noexcept override { return "AddPlugin"; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    int getNbOutputs() const noexcept override { return 1; }

    nvinfer1::DimsExprs getOutputDimensions(int outputIndex,
                                            const nvinfer1::DimsExprs* inputs,
                                            int nbInputs,
                                            nvinfer1::IExprBuilder& exprBuilder) noexcept override
    {
        (void)outputIndex; (void)nbInputs; (void)exprBuilder;
        return inputs[0];
    }

    bool supportsFormatCombination(int pos,
                                   const nvinfer1::PluginTensorDesc* inOut,
                                   int nbInputs,
                                   int nbOutputs) noexcept override
    {
        const auto& desc = inOut[pos];
        bool typeSupported = desc.type == nvinfer1::DataType::kFLOAT || desc.type == nvinfer1::DataType::kHALF;
        bool formatSupported = desc.format == nvinfer1::TensorFormat::kLINEAR;
        (void)nbInputs; (void)nbOutputs;
        return typeSupported && formatSupported;
    }

    nvinfer1::DataType getOutputDataType(int index,
                                         const nvinfer1::DataType* inputTypes,
                                         int nbInputs) const noexcept override
    {
        (void)index; (void)nbInputs;
        return inputTypes[0];
    }

    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                         int nbInputs,
                         const nvinfer1::DynamicPluginTensorDesc* out,
                         int nbOutputs) noexcept override
    {
        // 目前无需特殊配置，留空
        (void)in; (void)nbInputs; (void)out; (void)nbOutputs;
    }

    size_t getSerializationSize() const noexcept override { return 0; }
    void serialize(void* buffer) const noexcept override { (void)buffer; }
    int initialize() noexcept override { return 0; }
    void terminate() noexcept override {}
    void destroy() noexcept override { delete this; }

    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                            int nbInputs,
                            const nvinfer1::PluginTensorDesc* outputs,
                            int nbOutputs) const noexcept override
    {
        (void)inputs; (void)nbInputs; (void)outputs; (void)nbOutputs;
        return 0;
    }

    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                const nvinfer1::PluginTensorDesc* outputDesc,
                const void* const* inputs,
                void* const* outputs,
                void* workspace,
                cudaStream_t stream) noexcept override{
                    (void)workspace;

                    int totalElements = 1;
                    for (int i = 0; i < outputDesc[0].dims.nbDims; ++i)
                        totalElements *= outputDesc[0].dims.d[i];

                    nvinfer1::DataType dtype = inputDesc[0].type;

                    if (dtype == nvinfer1::DataType::kFLOAT)
                    {
                        launchAddKernelFloat(
                            static_cast<const float*>(inputs[0]),
                            static_cast<const float*>(inputs[1]),
                            static_cast<float*>(outputs[0]),
                            totalElements,
                            stream
                        );
                    }
                    else if (dtype == nvinfer1::DataType::kHALF)
                    {
                        launchAddKernelHalf(
                            static_cast<const __half*>(inputs[0]),
                            static_cast<const __half*>(inputs[1]),
                            static_cast<__half*>(outputs[0]),
                            totalElements,
                            stream
                        );
                    }
                    else
                    {
                        return -1;
                    }

                    auto err = cudaGetLastError();
                    if (err != cudaSuccess)
                    {
                        printf("[AddPlugin] kernel launch error: %s\n", cudaGetErrorString(err));
                        return -1;
                    }

                    return 0;
                }

    void setPluginNamespace(const char* pluginNamespace) noexcept override { mNamespace = pluginNamespace; }
    const char* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }

private:
    std::string mNamespace;
};

// ================= AddPluginCreator =================
class AddPluginCreator : public PluginCreator {
public:
    AddPluginCreator() {}
    ~AddPluginCreator() override {}

    const char* getPluginName() const noexcept override { return "AddPlugin"; }
    const char* getPluginVersion() const noexcept override { return "1"; }

    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override {
        return &fc_;
    }

    nvinfer1::IPluginV2* createPlugin(
        const char* name,
        const nvinfer1::PluginFieldCollection* fc) noexcept override
    {
        (void)name; (void)fc;
        return new AddPlugin();
    }

    nvinfer1::IPluginV2* deserializePlugin(
        const char* name,
        const void* serialData,
        size_t serialLength) noexcept override
    {
        (void)name;
        return new AddPlugin(serialData, serialLength);
    }

private:
    nvinfer1::PluginFieldCollection fc_{0, nullptr};
};

// 注册插件
REGISTER_TENSORRT_PLUGIN(AddPluginCreator);