#pragma once
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <string>

class PluginCreator : public nvinfer1::IPluginCreator {
public:
    PluginCreator() = default;
    ~PluginCreator() override = default;

    const char* getPluginName() const noexcept override = 0;
    const char* getPluginVersion() const noexcept override = 0;

    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override = 0;

    nvinfer1::IPluginV2* createPlugin(
        const char* name,
        const nvinfer1::PluginFieldCollection* fc) noexcept override = 0;

    nvinfer1::IPluginV2* deserializePlugin(
        const char* name,
        const void* serialData,
        size_t serialLength) noexcept override = 0;

    void setPluginNamespace(const char* pluginNamespace) noexcept override {
        mNamespace = pluginNamespace;
    }

    const char* getPluginNamespace() const noexcept override {
        return mNamespace.c_str();
    }

private:
    std::string mNamespace;
};