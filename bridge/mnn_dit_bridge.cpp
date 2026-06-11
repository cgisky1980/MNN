#include <cstdio>
#include <cstring>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>

#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/HalideRuntime.h>
#include "half.hpp"

#ifdef _WIN32
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {

struct MNNModel {
    MNN::Interpreter* net;
    MNN::Session* session;
};

static FILE* g_debug_file = nullptr;

static bool debug_callback(const std::vector<MNN::Tensor*>& tensors, const MNN::OperatorInfo* info) {
    if (!g_debug_file) return false;
    const char* op_type = info ? info->type().c_str() : "?";
    const char* op_name = info ? info->name().c_str() : "?";
    for (int i = 0; i < tensors.size(); i++) {
        auto& t = tensors[i];
        if (!t || t->dimensions() == 0) continue;
        int count = 1;
        for (int d = 0; d < t->dimensions(); d++) count *= t->length(d);
        if (count <= 0 || count > 100000000) continue;

        auto host = new MNN::Tensor(t, MNN::Tensor::CAFFE);
        t->copyToHostTensor(host);

        float sum = 0, abs_sum = 0, max_val = -1e30f, min_val = 1e30f;
        int n = std::min(count, 10000);
        auto ptr = host->host<float>();
        for (int j = 0; j < n; j++) {
            float v = ptr[j];
            sum += v;
            abs_sum += fabsf(v);
            if (v > max_val) max_val = v;
            if (v < min_val) min_val = v;
        }
        delete host;

        fprintf(g_debug_file, "OP|%s|%s|%d|count=%d|sum=%.6f|abs_sum=%.6f|min=%.6f|max=%.6f\n",
                op_type, op_name, i, count, sum, abs_sum, min_val, max_val);
    }
    return true;
}

DLL_EXPORT MNNModel* mnn_model_create(
    const char* model_path,
    int use_gpu,
    int threads,
    int precision
) {
    auto net = MNN::Interpreter::createFromFile(model_path);
    if (!net) return nullptr;

    MNN::ScheduleConfig config;
    if (use_gpu == 1) {
        config.type = MNN_FORWARD_CUDA;
        config.numThread = 1;
        config.backupType = MNN_FORWARD_CPU;
    } else if (use_gpu == 2) {
        config.type = MNN_FORWARD_VULKAN;
        config.numThread = 1;
        config.backupType = MNN_FORWARD_CPU;
    } else {
        config.type = MNN_FORWARD_CPU;
        config.numThread = threads;
        config.backupType = MNN_FORWARD_CPU;
    }

    MNN::BackendConfig backend_config;
    backend_config.memory = MNN::BackendConfig::Memory_Normal;
    backend_config.power = MNN::BackendConfig::Power_Normal;
    if (precision == 1) {
        backend_config.precision = MNN::BackendConfig::Precision_High;
    } else {
        backend_config.precision = MNN::BackendConfig::Precision_Normal;
    }
    config.backendConfig = &backend_config;

    if (g_debug_file) {
        net->setSessionMode(MNN::Interpreter::Session_Debug);
    } else {
        net->setSessionMode(MNN::Interpreter::Session_Release);
    }
    net->setSessionMode(MNN::Interpreter::Session_Input_Inside);

    auto session = net->createSession(config);
    if (!session) {
        MNN::Interpreter::destroy(net);
        return nullptr;
    }

    auto handle = new MNNModel();
    handle->net = net;
    handle->session = session;
    return handle;
}

DLL_EXPORT int mnn_model_set_debug(const char* output_path) {
    if (g_debug_file) {
        fclose(g_debug_file);
        g_debug_file = nullptr;
    }
    if (output_path && strlen(output_path) > 0) {
        g_debug_file = fopen(output_path, "w");
        if (!g_debug_file) return -1;
        return 0;
    }
    return 0;
}

DLL_EXPORT int mnn_model_resize(
    MNNModel* handle,
    const char* input_name,
    int ndim,
    const int* dims
) {
    if (!handle) return -1;
    auto tensor = handle->net->getSessionInput(handle->session, input_name);
    if (!tensor) return -2;
    std::vector<int> shape(dims, dims + ndim);
    handle->net->resizeTensor(tensor, shape);
    return 0;
}

DLL_EXPORT int mnn_model_resize_commit(MNNModel* handle) {
    if (!handle) return -1;
    handle->net->resizeSession(handle->session);
    return 0;
}

DLL_EXPORT int mnn_model_set_input(
    MNNModel* handle,
    const char* input_name,
    const float* data,
    int count
) {
    if (!handle) return -1;
    auto tensor = handle->net->getSessionInput(handle->session, input_name);
    if (!tensor) return -2;
    auto host = new MNN::Tensor(tensor, MNN::Tensor::CAFFE);
    auto type = host->getType();

    if (type == halide_type_of<float>()) {
        memcpy(host->host<float>(), data, count * sizeof(float));
    } else if (type == halide_type_of<bool>() || type == halide_type_of<uint8_t>()) {
        auto ptr = host->host<uint8_t>();
        for (int i = 0; i < count; i++) {
            ptr[i] = data[i] != 0.0f ? 1 : 0;
        }
    } else if (type == halide_type_of<int32_t>()) {
        auto ptr = host->host<int32_t>();
        for (int i = 0; i < count; i++) {
            ptr[i] = (int32_t)data[i];
        }
    } else {
        delete host;
        return -3;
    }

    tensor->copyFromHostTensor(host);
    delete host;
    return 0;
}

DLL_EXPORT int mnn_model_set_input_i64(
    MNNModel* handle,
    const char* input_name,
    const int64_t* data,
    int count
) {
    if (!handle) return -1;
    auto tensor = handle->net->getSessionInput(handle->session, input_name);
    if (!tensor) return -2;
    auto host = new MNN::Tensor(tensor, MNN::Tensor::CAFFE);
    auto type = host->getType();

    if (type == halide_type_of<int64_t>()) {
        memcpy(host->host<int64_t>(), data, count * sizeof(int64_t));
    } else if (type == halide_type_of<int32_t>()) {
        auto ptr = host->host<int32_t>();
        for (int i = 0; i < count; i++) {
            ptr[i] = (int32_t)data[i];
        }
    } else {
        delete host;
        return -3;
    }

    tensor->copyFromHostTensor(host);
    delete host;
    return 0;
}

DLL_EXPORT int mnn_model_run(MNNModel* handle) {
    if (!handle) return -1;
    if (g_debug_file) {
        auto before = [](const std::vector<MNN::Tensor*>&, const MNN::OperatorInfo*) -> bool { return true; };
        auto ret = handle->net->runSessionWithCallBackInfo(handle->session, before, debug_callback, true);
        fflush(g_debug_file);
        return (int)ret;
    }
    auto ret = handle->net->runSession(handle->session);
    return (int)ret;
}

DLL_EXPORT int mnn_model_get_output(
    MNNModel* handle,
    const char* output_name,
    float* out_data,
    int max_count
) {
    if (!handle) return -1;
    auto tensor = handle->net->getSessionOutput(handle->session, output_name);
    if (!tensor) return -2;
    auto host = new MNN::Tensor(tensor, MNN::Tensor::CAFFE);
    tensor->copyToHostTensor(host);
    int count = 1;
    for (int i = 0; i < host->dimensions(); i++) {
        count *= host->length(i);
    }
    int copy_count = count < max_count ? count : max_count;
    auto type = host->getType();
    if (type.code == halide_type_float && type.bits == 32) {
        memcpy(out_data, host->host<float>(), copy_count * sizeof(float));
    } else if (type.code == halide_type_float && type.bits == 16) {
        auto src = host->host<uint16_t>();
        for (int i = 0; i < copy_count; i++) {
            half_float::half h;
            memcpy(&h, &src[i], sizeof(half_float::half));
            out_data[i] = (float)h;
        }
    } else {
        for (int i = 0; i < copy_count; i++) {
            out_data[i] = (float)host->host<float>()[i];
        }
    }
    delete host;
    return count;
}

DLL_EXPORT int mnn_model_get_output_dims(
    MNNModel* handle,
    const char* output_name,
    int* out_dims,
    int max_ndim
) {
    if (!handle) return -1;
    auto tensor = handle->net->getSessionOutput(handle->session, output_name);
    if (!tensor) return -2;
    int ndim = tensor->dimensions();
    if (ndim > max_ndim) ndim = max_ndim;
    for (int i = 0; i < ndim; i++) {
        out_dims[i] = tensor->length(i);
    }
    return ndim;
}

DLL_EXPORT void mnn_model_destroy(MNNModel* handle) {
    if (!handle) return;
    handle->net->releaseSession(handle->session);
    MNN::Interpreter::destroy(handle->net);
    delete handle;
}

}
