# **************************************************************************
# ||                        SiMa.ai CONFIDENTIAL                          ||
# ||   Unpublished Copyright (c) 2022-2023 SiMa.ai, All Rights Reserved.  ||
# **************************************************************************
# NOTICE:  All information contained herein is, and remains the property of
# SiMa.ai. The intellectual and technical concepts contained herein are
# proprietary to SiMa and may be covered by U.S. and Foreign Patents,
# patents in process, and are protected by trade secret or copyright law.
#
# Dissemination of this information or reproduction of this material is
# strictly forbidden unless prior written permission is obtained from
# SiMa.ai.  Access to the source code contained herein is hereby forbidden
# to anyone except current SiMa.ai employees, managers or contractors who
# have executed Confidentiality and Non-disclosure agreements explicitly
# covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes information
# that is confidential and/or proprietary, and is a trade secret, of SiMa.ai.
#
# ANY REPRODUCTION, MODIFICATION, DISTRIBUTION, PUBLIC PERFORMANCE, OR PUBLIC
# DISPLAY OF OR THROUGH USE OF THIS SOURCE CODE WITHOUT THE EXPRESS WRITTEN
# CONSENT OF SiMa.ai IS STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE
# LAWS AND INTERNATIONAL TREATIES. THE RECEIPT OR POSSESSION OF THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS TO
# REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE, USE, OR
# SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# **************************************************************************
import numpy as np
import onnx
import onnx.helper as helper
import onnx_helpers as oh

model_name = "../exported_onnx_models/classifier_brats"
mod_model_name = f"{model_name}_split"

print(f"\n{'='*60}")
print(f"Split-ReduceMean-Concat Transformation")
print(f"{'='*60}")

# Load the model
model = oh.load_model(f"{model_name}.onnx", load_only=False)
onnx.checker.check_model(model)

# Find the /vit_model/ReduceMean node
target_node_name = "/vit_model/ReduceMean"
target_node = None

for node in model.graph.node:
    if node.name == target_node_name:
        target_node = node
        break

if not target_node:
    print(f"Error: Could not find node {target_node_name}")
    exit(1)

print(f"\nFound target node: {target_node_name}")
print(f"  Op type: {target_node.op_type}")
print(f"  Inputs: {list(target_node.input)}")
print(f"  Outputs: {list(target_node.output)}")

# Get node attributes
axes = None
keepdims = 1  # default
for attr in target_node.attribute:
    if attr.name == "axes":
        axes = list(attr.ints)
    elif attr.name == "keepdims":
        keepdims = attr.i

print(f"  Axes: {axes}")
print(f"  Keepdims: {keepdims}")

# Get the input and output names
input_name = target_node.input[0]
output_name = target_node.output[0]

# Find the input shape
input_shape = None
for value_info in model.graph.value_info:
    if value_info.name == input_name:
        input_shape = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
        break

# Also check graph inputs
if input_shape is None:
    for graph_input in model.graph.input:
        if graph_input.name == input_name:
            input_shape = [dim.dim_value for dim in graph_input.type.tensor_type.shape.dim]
            break

print(f"\nInput shape: {input_shape}")

# Determine split parameters
# The ReduceMean reduces along axis=1 (or whatever 'axes' specifies)
# We want to split along the LAST axis (features dimension)
# For input [1, 196, 768], we split axis 2 to get 4x [1, 196, 192]
# Then ReduceMean(axis=1) on each gives 4x [1, 192]
# Finally Concat(axis=1) gives [1, 768]

if axes and len(axes) == 1:
    reduce_axis = axes[0]
    # Split along a different axis (the last one, typically features)
    if len(input_shape) == 3:
        split_axis = 2  # Split the 768 dimension
    else:
        split_axis = -1  # Default to last axis
else:
    print(f"Error: Unsupported axes configuration: {axes}")
    exit(1)

num_splits = 4
split_size = input_shape[split_axis] // num_splits if input_shape else 192

print(f"\nSplit configuration:")
print(f"  Reduce axis: {reduce_axis}")
print(f"  Split axis: {split_axis}")
print(f"  Number of splits: {num_splits}")
print(f"  Split size: {split_size}")

# After split, each tensor will be reduced along the reduce_axis
# Then we need to concatenate back along the split_axis
# But wait - after reduction with keepdims=0, dimensions shift!
# Input: [1, 196, 768] -> Split axis 2 -> 4x [1, 196, 192]
# ReduceMean(axis=1, keepdims=0) -> 4x [1, 192]
# Concat(axis=1) -> [1, 768] ✓

concat_axis = split_axis if keepdims == 1 else split_axis - 1 if split_axis > reduce_axis else split_axis
print(f"  Concat axis: {concat_axis}")

# Create the transformation
print(f"\n{'='*60}")
print(f"Creating transformation nodes...")
print(f"{'='*60}")

# 1. Create Split node
split_node_name = f"{target_node_name}/Split"
split_outputs = [f"{split_node_name}_output_{i}" for i in range(num_splits)]

# Create split attribute tensor
split_attr_name = f"{split_node_name}_split_attr"
split_tensor = helper.make_tensor(
    name=split_attr_name,
    data_type=onnx.TensorProto.INT64,
    dims=[num_splits],
    vals=[split_size] * num_splits
)
model.graph.initializer.append(split_tensor)

split_node = helper.make_node(
    'Split',
    inputs=[input_name, split_attr_name],
    outputs=split_outputs,
    name=split_node_name,
    axis=split_axis
)

print(f"1. Created Split node: {split_node_name}")
print(f"   Outputs: {split_outputs}")

# 2. Create Transpose + Split + ReduceMean + ReduceMean nodes for each split
# We need to split the reduction dimension (196) into chunks < 128
# 196 / 2 = 98 (which is < 128)
transpose_nodes = []
inner_split_nodes = []
concat_inner_nodes = []  # Separate list for concat nodes
inner_reduce_nodes = []
outer_reduce_nodes = []
transpose_outputs = []
inner_split_outputs = []
inner_reduce_outputs = []
outer_reduce_outputs = []

# Determine transpose permutation
# For 3D tensor [1, 196, 192], we want to swap last two dims to get [1, 192, 196]
transpose_perm = [0, 2, 1]

# Inner split configuration: split 196 into 2 chunks of 98
inner_num_splits = 2
inner_split_size = 98  # 196 / 2

print(f"\nInner split configuration (to reduce dimension < 128):")
print(f"  Inner splits per branch: {inner_num_splits}")
print(f"  Inner split size: {inner_split_size}")

for i in range(num_splits):
    # Create Transpose node
    transpose_node_name = f"{target_node_name}/Transpose_{i}"
    transpose_output = f"{transpose_node_name}_output"
    transpose_outputs.append(transpose_output)
    
    transpose_node = helper.make_node(
        'Transpose',
        inputs=[split_outputs[i]],
        outputs=[transpose_output],
        name=transpose_node_name,
        perm=transpose_perm
    )
    transpose_nodes.append(transpose_node)
    
    print(f"\n2.{i+1}a. Created Transpose node: {transpose_node_name} (perm={transpose_perm})")
    
    # Create inner Split node to split [1, 192, 196] along axis=-1 into [1, 192, 98] chunks
    inner_split_node_name = f"{target_node_name}/InnerSplit_{i}"
    inner_split_output_list = [f"{inner_split_node_name}_output_{j}" for j in range(inner_num_splits)]
    inner_split_outputs.append(inner_split_output_list)
    
    # Create split attribute tensor
    inner_split_attr_name = f"{inner_split_node_name}_attr"
    inner_split_tensor = helper.make_tensor(
        name=inner_split_attr_name,
        data_type=onnx.TensorProto.INT64,
        dims=[inner_num_splits],
        vals=[inner_split_size] * inner_num_splits
    )
    model.graph.initializer.append(inner_split_tensor)
    
    inner_split_node = helper.make_node(
        'Split',
        inputs=[transpose_output, inner_split_attr_name],
        outputs=inner_split_output_list,
        name=inner_split_node_name,
        axis=-1
    )
    inner_split_nodes.append(inner_split_node)
    
    print(f"2.{i+1}b. Created inner Split node: {inner_split_node_name} (axis=-1, {inner_num_splits} splits of {inner_split_size})")
    
    # Create ReduceMean nodes for each inner split
    inner_reduce_output_list = []
    inner_reduce_node_list = []
    for j in range(inner_num_splits):
        inner_reduce_node_name = f"{target_node_name}/InnerReduce_{i}_{j}"
        inner_reduce_output = f"{inner_reduce_node_name}_output"
        inner_reduce_output_list.append(inner_reduce_output)
        
        inner_reduce_node = helper.make_node(
            'ReduceMean',
            inputs=[inner_split_output_list[j]],
            outputs=[inner_reduce_output],
            name=inner_reduce_node_name,
            axes=[-1],
            keepdims=1  # Keep dims to maintain shape for outer reduction
        )
        inner_reduce_node_list.append(inner_reduce_node)
        
    inner_reduce_nodes.append(inner_reduce_node_list)
    inner_reduce_outputs.append(inner_reduce_output_list)
    
    print(f"2.{i+1}c. Created {inner_num_splits} inner ReduceMean nodes (axis=-1, keepdims=1)")
    
    # Create outer ReduceMean to average the inner means
    # Concatenate the inner reduce outputs first
    concat_inner_node_name = f"{target_node_name}/ConcatInner_{i}"
    concat_inner_output = f"{concat_inner_node_name}_output"
    
    concat_inner_node = helper.make_node(
        'Concat',
        inputs=inner_reduce_output_list,
        outputs=[concat_inner_output],
        name=concat_inner_node_name,
        axis=-1
    )
    concat_inner_nodes.append(concat_inner_node)  # Add to separate concat list
    
    print(f"2.{i+1}d. Created Concat node: {concat_inner_node_name}")
    
    # Now ReduceMean over the concatenated results
    outer_reduce_node_name = f"{target_node_name}/ReduceMean_{i}"
    outer_reduce_output = f"{outer_reduce_node_name}_output"
    outer_reduce_outputs.append(outer_reduce_output)
    
    outer_reduce_node = helper.make_node(
        'ReduceMean',
        inputs=[concat_inner_output],
        outputs=[outer_reduce_output],
        name=outer_reduce_node_name,
        axes=[-1],
        keepdims=keepdims
    )
    outer_reduce_nodes.append(outer_reduce_node)
    
    print(f"2.{i+1}e. Created outer ReduceMean node: {outer_reduce_node_name} (axis=-1, keepdims={keepdims})")

# 3. Create final Concat node
concat_node_name = f"{target_node_name}/Concat"
concat_node = helper.make_node(
    'Concat',
    inputs=outer_reduce_outputs,
    outputs=[output_name],  # Use the original output name
    name=concat_node_name,
    axis=concat_axis
)

print(f"\n3. Created final Concat node: {concat_node_name}")
print(f"   Output: {output_name}")
print(f"   Concat axis: {concat_axis}")

# Insert the new nodes and remove the old one
print(f"\n{'='*60}")
print(f"Modifying graph...")
print(f"{'='*60}")

# Find the position of the target node
target_position = None
for i, node in enumerate(model.graph.node):
    if node.name == target_node_name:
        target_position = i
        break

# Remove the target node
model.graph.node.remove(target_node)
print(f"Removed original ReduceMean node")

# Insert new nodes at the same position
# Build the complete node sequence in correct topological order
interleaved_nodes = []
for i in range(num_splits):
    interleaved_nodes.append(transpose_nodes[i])
    interleaved_nodes.append(inner_split_nodes[i])
    # Add all inner reduce nodes for this split
    for inner_reduce_node in inner_reduce_nodes[i]:
        interleaved_nodes.append(inner_reduce_node)
    # Add concat inner node (from separate list)
    interleaved_nodes.append(concat_inner_nodes[i])
    interleaved_nodes.append(outer_reduce_nodes[i])

new_nodes = [split_node] + interleaved_nodes + [concat_node]
for i, new_node in enumerate(new_nodes):
    model.graph.node.insert(target_position + i, new_node)
    print(f"Inserted: {new_node.name}")

# Clear shape information for the modified tensors
print(f"\nClearing shape information...")
tensors_to_clear = split_outputs + transpose_outputs + [output_name]
# Add all inner split outputs
for inner_list in inner_split_outputs:
    tensors_to_clear.extend(inner_list)
# Add all inner reduce outputs
for inner_list in inner_reduce_outputs:
    tensors_to_clear.extend(inner_list)
# Add outer reduce outputs
tensors_to_clear.extend(outer_reduce_outputs)
# Add concat inner outputs
for i in range(num_splits):
    tensors_to_clear.append(f"{target_node_name}/ConcatInner_{i}_output")

for value_info in list(model.graph.value_info):
    if value_info.name in tensors_to_clear:
        model.graph.value_info.remove(value_info)
        print(f"  Cleared: {value_info.name}")

# Run shape inference
print(f"\nRunning shape inference...")
try:
    model = onnx.shape_inference.infer_shapes(model)
    print("  Shape inference completed successfully")
except Exception as e:
    print(f"  Warning: Shape inference had issues: {e}")
    print("  Continuing anyway...")

# Save the model
print(f"\n{'='*60}")
print(f"Saving modified model...")
print(f"{'='*60}")

onnx.save(model, f"{mod_model_name}.onnx")
print(f"Model saved to: {mod_model_name}.onnx")

# Verify the transformation by comparing outputs
print(f"\n{'='*60}")
print(f"Verifying transformation...")
print(f"{'='*60}")

# Load both models
original_model = onnx.load(f"{model_name}.onnx")
modified_model = onnx.load(f"{mod_model_name}.onnx")

# Get input info
input_info = original_model.graph.input[0]
input_tensor_name = input_info.name
input_shape_full = [dim.dim_value if dim.dim_value > 0 else 1 for dim in input_info.type.tensor_type.shape.dim]

print(f"\nInput: {input_tensor_name}")
print(f"Shape: {input_shape_full}")

# Generate random input
random_input = np.random.randn(*input_shape_full).astype(np.float32)
print(f"Generated random input with shape: {random_input.shape}")

# Run inference on both models
print(f"\nRunning inference on original model...")
original_outputs = oh.run_model(f"{model_name}.onnx", [input_tensor_name], [random_input])
print(f"Original output shape: {original_outputs[0].shape}")

print(f"\nRunning inference on modified model...")
modified_outputs = oh.run_model(f"{mod_model_name}.onnx", [input_tensor_name], [random_input])
print(f"Modified output shape: {modified_outputs[0].shape}")

# Compare outputs
print(f"\n{'='*60}")
print(f"Comparison Results")
print(f"{'='*60}")

orig = original_outputs[0]
mod = modified_outputs[0]

is_close = np.allclose(orig, mod, rtol=1e-5, atol=1e-5)
max_diff = np.max(np.abs(orig - mod))
mean_diff = np.mean(np.abs(orig - mod))

print(f"\nShape match: {orig.shape == mod.shape}")
print(f"All close (rtol=1e-5, atol=1e-5): {is_close}")
print(f"Max absolute difference: {max_diff:.2e}")
print(f"Mean absolute difference: {mean_diff:.2e}")

if orig.size > 10:
    print(f"\nOriginal - Min: {orig.min():.6f}, Max: {orig.max():.6f}, Mean: {orig.mean():.6f}")
    print(f"Modified - Min: {mod.min():.6f}, Max: {mod.max():.6f}, Mean: {mod.mean():.6f}")

print(f"\n{'='*60}")
if is_close and max_diff < 1e-6:
    print(f"✓ SUCCESS: Outputs match perfectly!")
elif is_close:
    print(f"✓ SUCCESS: Outputs match within tolerance!")
else:
    print(f"✗ WARNING: Outputs differ!")
print(f"{'='*60}")

