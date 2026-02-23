import onnx_helpers as oh
import onnx
import numpy as np
import onnxruntime as ort

model_path = '../exported_onnx_models/mae_brats_random.onnx'
modified_model_path = f'../exported_onnx_models/mae_brats_random_5d.onnx'
model = oh.load_model(model_path)

# Print current value_info and initializers
print("\nCurrent value_info:")
for info in model.graph.value_info:
    print(f"{info.name}: {[d.dim_value for d in info.type.tensor_type.shape.dim]}")

print("\nCurrent initializers:")
for init in model.graph.initializer:
    print(f"{init.name}")

# First fix the Slice operations
old_constant = '/mae_model/Constant_3_output_0'
nodes_using_constant = []
for node in model.graph.node:
    if old_constant in node.input:
        nodes_using_constant.append(node)

print("\nNodes using old constant:")
for node in nodes_using_constant:
    print(f"\nNode {node.name} (type: {node.op_type})")
    print("  Inputs:", node.input)
    print("  Outputs:", node.output)
    positions = [i for i, inp in enumerate(node.input) if inp == old_constant]
    print("  Constant used in positions:", positions)

# Create constants for different values
constant_nodes = {
    '1': onnx.helper.make_node(
        'Constant',
        name='_constant_1',
        inputs=[],
        outputs=['_constant_1_output'],
        value=onnx.helper.make_tensor(
            name='const_tensor_1',
            data_type=onnx.TensorProto.INT64,
            dims=[1],
            vals=[1]
        )
    ),
    '0': onnx.helper.make_node(
        'Constant',
        name='_constant_0',
        inputs=[],
        outputs=['_constant_0_output'],
        value=onnx.helper.make_tensor(
            name='const_tensor_0',
            data_type=onnx.TensorProto.INT64,
            dims=[1],
            vals=[0]
        )
    ),
    '50': onnx.helper.make_node(
        'Constant',
        name='_constant_50',
        inputs=[],
        outputs=['_constant_50_output'],
        value=onnx.helper.make_tensor(
            name='const_tensor_50',
            data_type=onnx.TensorProto.INT64,
            dims=[1],
            vals=[50]
        )
    ),
    '197': onnx.helper.make_node(
        'Constant',
        name='_constant_197',
        inputs=[],
        outputs=['_constant_197_output'],
        value=onnx.helper.make_tensor(
            name='const_tensor_197',
            data_type=onnx.TensorProto.INT64,
            dims=[1],
            vals=[197]
        )
    )
}

# Add all constant nodes at the beginning of the graph
for node in constant_nodes.values():
    model.graph.node.insert(0, node)

# Update all nodes that use the old constant
for node in nodes_using_constant:
    if node.op_type == 'Slice':
        inputs = list(node.input)
        
        if node.name == '/mae_model/Slice_5':
            inputs = [
                inputs[0],                    # Keep original input
                '_constant_0_output',         # start at 0
                '_constant_1_output',         # end at 1
                '_constant_1_output',         # axis 1
                '_constant_1_output'          # step 1
            ]
        elif node.name == '/mae_model/Slice_6':
            inputs = [
                inputs[0],                    # Keep original input
                '_constant_1_output',         # start at 1
                '_constant_197_output',       # end at 197
                '_constant_1_output',         # axis 1
                '_constant_1_output'          # step 1
            ]
        elif node.name == '/mae_model/Slice_4':
            inputs = [
                inputs[0],                    # Keep original input
                '_constant_1_output',         # start at 1
                '_constant_50_output',        # end at 50
                '_constant_1_output',         # axis 1
                '_constant_1_output'          # step 1
            ]
        else:
            for i, inp in enumerate(inputs):
                if inp == old_constant:
                    if i == 1:  # start index
                        inputs[i] = '_constant_1_output'
                    elif i == 2:  # end index
                        inputs[i] = '_constant_197_output'
                    else:  # axes and steps
                        inputs[i] = '_constant_1_output'
        
        new_node = onnx.helper.make_node(
            'Slice',
            name=f"{node.name}_new",
            inputs=inputs,
            outputs=node.output
        )
        
        idx = list(model.graph.node).index(node)
        model.graph.node.remove(node)
        model.graph.node.insert(idx, new_node)

# Now simplify the dimensions
# Find the reshape nodes
reshape1 = oh.find_node(model, '/Reshape')
reshape2 = oh.find_node(model, '/Reshape_1')
einsum = oh.find_node(model, '/Einsum')

# Create new shape tensors for 5D representation
shape1_tensor = onnx.helper.make_tensor(
    name='_reshape1_shape',
    data_type=onnx.TensorProto.INT64,
    dims=[5],
    vals=[1, 14, 14, 16, 16]  # (batch, h, w, patch_h, patch_w)
)

shape2_tensor = onnx.helper.make_tensor(
    name='_reshape2_shape',
    data_type=onnx.TensorProto.INT64,
    dims=[4],
    vals=[1, 1, 224, 224]  # Final output shape
)

# Remove old initializers and add new ones
old_initializers = ['_v_3617', '_v_3619', '/mae_model/Constant_3_output_0', '/mae_model/blocks.0/attn/Constant_11_output_0']
new_initializers = [init for init in model.graph.initializer if init.name not in old_initializers]
new_initializers.extend([shape1_tensor, shape2_tensor])

# Replace all initializers at once
del model.graph.initializer[:]  # Clear the repeated field
model.graph.initializer.extend(new_initializers)

# Update reshape nodes
reshape1.input[1] = '_reshape1_shape'
reshape2.input[1] = '_reshape2_shape'

# Update Einsum equation to work with 5D tensors
# Original: nhwpqc->nchpwq (6D)
# New: nhwpq->nhpwq (5D: batch, height, patch_h, width, patch_w)
einsum.attribute[0].s = b"nhwpq->nhpwq"

# Update value_info for the reshaped tensors
reshape1_output_shape = onnx.helper.make_tensor_value_info(
    '/Reshape_output_0',
    onnx.TensorProto.FLOAT,
    [1, 14, 14, 16, 16]  # 5D shape
)

einsum_output_shape = onnx.helper.make_tensor_value_info(
    '/Einsum_output_0',
    onnx.TensorProto.FLOAT,
    [1, 14, 16, 14, 16]  # 5D shape
)

# Remove old value_info and add new ones
old_value_info = ['/Reshape_output_0', '/Einsum_output_0']
new_value_info = [info for info in model.graph.value_info if info.name not in old_value_info]
new_value_info.extend([reshape1_output_shape, einsum_output_shape])

# Replace all value_info at once
del model.graph.value_info[:]  # Clear the repeated field
model.graph.value_info.extend(new_value_info)

# Save the model
onnx.save(model, modified_model_path)

print("\nModel modified with new nodes")
print("\nNew shapes:")
print("First reshape output:", [d.dim_value for d in reshape1_output_shape.type.tensor_type.shape.dim])
print("Einsum output:", [d.dim_value for d in einsum_output_shape.type.tensor_type.shape.dim])

print("\nFinal initializers:")
for init in model.graph.initializer:
    print(f"{init.name}")

# Verify the model works
print("\nVerifying model execution...")
session = ort.InferenceSession(modified_model_path)
input_data = np.random.randn(1, 1, 224, 224).astype(np.float32)
output = session.run(None, {'input_image': input_data})[0]
print(f"Model execution successful, output shape: {output.shape}")

# Compare with original model
print("\nComparing with original model...")
original_session = ort.InferenceSession(model_path)
original_output = original_session.run(None, {'input_image': input_data})[0]

# Calculate differences
abs_diff = np.abs(original_output - output)
max_diff = np.max(abs_diff)
mean_diff = np.mean(abs_diff)
relative_diff = np.mean(np.abs(original_output - output) / (np.abs(original_output) + 1e-6))

print("\nDifferences between outputs:")
print(f"Maximum absolute difference: {max_diff}")
print(f"Mean absolute difference: {mean_diff}")
print(f"Mean relative difference: {relative_diff * 100:.6f}%")


