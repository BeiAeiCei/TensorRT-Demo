import onnx

src = "bert-base-uncased/model.onnx"
dst = "bert-base-uncased/model_add_custom.onnx"
target_idx = 68  # 只替换这个 Add 节点, 因为其他节点的Add输入输出都不满足插件要求，很复杂，这只是个Demo

model = onnx.load(src)

replaced = 0
for i, node in enumerate(model.graph.node):
    if i == target_idx:
        print("Found target node:")
        print("  graph_idx:", i)
        print("  name     :", node.name)
        print("  op_type  :", node.op_type)
        print("  inputs   :", list(node.input))
        print("  outputs  :", list(node.output))

        if node.op_type != "Add":
            raise RuntimeError(f"Node at graph_idx={i} is not Add")

        node.op_type = "AddPlugin"   # 必须匹配 PluginCreator::getPluginName()
        node.name = f"AddPlugin_{i}"
        replaced += 1
        break

if replaced != 1:
    raise RuntimeError(f"Failed to replace target Add node at graph_idx={target_idx}")

onnx.save(model, dst)
print(f"Replaced {replaced} Add node -> AddPlugin")
print(f"Saved to {dst}")