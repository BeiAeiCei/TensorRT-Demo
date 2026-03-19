import time
import json
import numpy as np
import onnxruntime as ort
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import ctypes
PLUGIN_PATH = "/home/user/Projects/TensorRTStudy/TensorRT/Myplugin/build/libMyPlugin.so"
ctypes.CDLL(PLUGIN_PATH, mode=ctypes.RTLD_GLOBAL)
trt.init_libnvinfer_plugins(None, "")

CASE_PATH = "bert-base-uncased/case_data.npz"
ONNX_PATH = "bert-base-uncased/model.onnx"
ENGINE_BASE = "output_no_plugin.engine"
ENGINE_PLUGIN = "output_with_myplugin.engine"
VOCAB_PATH = "bert-base-uncased/vocab.txt"

MASK_TOKEN_ID = 103


def load_vocab(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = [line.strip() for line in f]
    return vocab


def load_case():
    data = np.load(CASE_PATH)
    input_ids = data["input_ids"].astype(np.int64)
    token_type_ids = data["token_type_ids"].astype(np.int64)
    attention_mask = np.ones_like(input_ids, dtype=np.int64)
    ref_logits = data["logits"].astype(np.float32)
    return input_ids, token_type_ids, attention_mask, ref_logits


def run_ort(input_ids, token_type_ids, attention_mask, warmup=20, runs=100):
    sess = ort.InferenceSession(
        ONNX_PATH,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }

    for _ in range(warmup):
        out = sess.run(None, inputs)[0]

    start = time.perf_counter()
    for _ in range(runs):
        out = sess.run(None, inputs)[0]
    end = time.perf_counter()

    latency_ms = (end - start) * 1000 / runs
    return out, latency_ms


def trt_nptype(dtype):
    return np.dtype(trt.nptype(dtype))


def volume(shape):
    v = 1
    for x in shape:
        v *= x
    return v


def run_trt(engine_path, inputs, warmup=20, runs=100):
    logger = trt.Logger(trt.Logger.ERROR)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # set dynamic shapes
    for name, arr in inputs.items():
        context.set_input_shape(name, arr.shape)

    tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]

    host_buffers = {}
    device_buffers = {}
    bindings = [None] * engine.num_io_tensors

    stream = cuda.Stream()

    for i, name in enumerate(tensor_names):
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        shape = tuple(context.get_tensor_shape(name))
        host_mem = np.empty(shape, dtype=dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        host_buffers[name] = host_mem
        device_buffers[name] = device_mem
        bindings[i] = int(device_mem)

        context.set_tensor_address(name, int(device_mem))

    # copy inputs
    for name, arr in inputs.items():
        np.copyto(host_buffers[name], arr.astype(host_buffers[name].dtype))
        cuda.memcpy_htod_async(device_buffers[name], host_buffers[name], stream)

    # warmup
    for _ in range(warmup):
        context.execute_async_v3(stream.handle)

    stream.synchronize()

    start = time.perf_counter()
    for _ in range(runs):
        context.execute_async_v3(stream.handle)
    stream.synchronize()
    end = time.perf_counter()

    # copy outputs back
    outputs = {}
    for name in tensor_names:
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            cuda.memcpy_dtoh_async(host_buffers[name], device_buffers[name], stream)
            outputs[name] = host_buffers[name].copy()

    stream.synchronize()
    latency_ms = (end - start) * 1000 / runs
    return outputs, latency_ms


def compare(name, out, ref):
    abs_diff = np.abs(out - ref)
    mean_abs = abs_diff.mean()
    max_abs = abs_diff.max()

    denom = np.maximum(np.abs(ref), 1e-6)
    rel_diff = abs_diff / denom
    mean_rel = rel_diff.mean()
    max_rel = rel_diff.max()

    print(f"\n[{name}]")
    print("shape       :", out.shape)
    print("mean abs    :", mean_abs)
    print("max abs     :", max_abs)
    print("mean rel    :", mean_rel)
    print("max rel     :", max_rel)


def topk_at_mask(logits, input_ids, vocab, k=10):
    mask_pos = np.where(input_ids[0] == MASK_TOKEN_ID)[0]
    if len(mask_pos) == 0:
        print("No [MASK] token found.")
        return
    pos = int(mask_pos[0])
    scores = logits[0, pos]
    topk_ids = np.argsort(scores)[-k:][::-1]

    print(f"\n[MASK position = {pos}] top-{k}")
    for rank, tid in enumerate(topk_ids, 1):
        token = vocab[tid] if tid < len(vocab) else f"<id:{tid}>"
        print(f"{rank:2d}. id={tid:5d} token={token:15s} score={scores[tid]:.6f}")


def main():
    vocab = load_vocab(VOCAB_PATH)
    input_ids, token_type_ids, attention_mask, ref_logits = load_case()

    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }

    print("Running ONNX Runtime...")
    ort_out, ort_ms = run_ort(input_ids, token_type_ids, attention_mask)

    print("Running TensorRT base engine...")
    trt_base_outs, trt_base_ms = run_trt(ENGINE_BASE, inputs)
    trt_base = list(trt_base_outs.values())[0]

    print("Running TensorRT plugin engine...")
    trt_plugin_outs, trt_plugin_ms = run_trt(ENGINE_PLUGIN, inputs)
    trt_plugin = list(trt_plugin_outs.values())[0]

    # compare to reference
    compare("ORT vs REF", ort_out, ref_logits)
    compare("TRT_BASE vs REF", trt_base, ref_logits)
    compare("TRT_PLUGIN vs REF", trt_plugin, ref_logits)

    # direct compare
    compare("TRT_PLUGIN vs TRT_BASE", trt_plugin, trt_base)
    compare("TRT_PLUGIN vs ORT", trt_plugin, ort_out)

    print("\nLatency:")
    print(f"ORT        : {ort_ms:.3f} ms")
    print(f"TRT base   : {trt_base_ms:.3f} ms")
    print(f"TRT plugin : {trt_plugin_ms:.3f} ms")

    print("\nTop-k on mask token:")
    print("ORT")
    topk_at_mask(ort_out, input_ids, vocab, k=10)
    print("TRT base")
    topk_at_mask(trt_base, input_ids, vocab, k=10)
    print("TRT plugin")
    topk_at_mask(trt_plugin, input_ids, vocab, k=10)


if __name__ == "__main__":
    main()