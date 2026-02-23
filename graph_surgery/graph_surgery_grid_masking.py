import numpy as np
import onnx
from onnx import numpy_helper, helper
import onnx_helpers as oh



def rewrite_gather_elements_with_gather(model: onnx.ModelProto, target_node_name: str) -> onnx.ModelProto:
    """
    Replaces a GatherElements node with a simpler Gather node.
    This is possible because the indices are constant and the rows of a large GatherElements initializer
    contain repeated values along the gather axis. By taking a single representative row,
    we can construct an equivalent Gather operation that produces the same result.

    This function only supports this special case (3D tensors with static indices
    and repeated indices values in each raw).

    Steps:
    1. Locate the target GatherElements node.
    2. Extract its 'data' and 'indices' inputs.
    3. Verify that the 'indices' input is an initializer (static constant).
    4. Extract a simple index vector along the gather axis.
    5. Create a new Gather node with the same output and axis.
    6. Replace the original node in the graph.

    Args:
        model (onnx.ModelProto): The ONNX model to modify.
        target_node_name (str): Name of the GatherElements node to replace.

    Returns:
        onnx.ModelProto: The modified ONNX model.
    """
    # Find the target GatherElements node
    target_node = oh.find_node(model, target_node_name)
    data_input = target_node.input[0]
    indices_input = target_node.input[1]

    # Ensure indices are static (initializer)
    indices_init = next(
        (init for init in model.graph.initializer if init.name == indices_input),
        None
    )
    if indices_init is None:
        raise ValueError(
            f"indices '{indices_input}' is not an initializer; dynamic indices are not supported."
        )

    indices_arr = numpy_helper.to_array(indices_init)

    # Validate tensor dimensions
    if indices_arr.ndim != 3:
        raise ValueError(f"Only 3D tensors supported, got shape {indices_arr.shape}")

    # Extract simple index vector
    # Take the first column of the first batch along axis 2 to get 1D indices
    gather_indices = indices_arr[:, :, 0][0].astype(np.int64)
    gather_indices_tensor = numpy_helper.from_array(
        gather_indices,
        name=indices_input + "_gather"
    )

    # Add the new constant initializer to the model
    model.graph.initializer.append(gather_indices_tensor)

    # Create new Gather node
    axis = next((attr.i for attr in target_node.attribute if attr.name == "axis"), 0)
    gather_node = helper.make_node(
        "Gather",
        name=target_node.name + "/Gather",
        inputs=[data_input, gather_indices_tensor.name],
        outputs=target_node.output,
        axis=axis
    )

    # Replace original node in graph
    node_index = list(model.graph.node).index(target_node)
    model.graph.node.remove(target_node)
    model.graph.node.insert(node_index, gather_node)

    return model


def rewrite_reshape_6d_and_einsum_6d_part(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Rewrites a specific 6D Reshape + 6D Einsum pattern  by replacing it with a sequence of Reshape and Transpose nodes.

    Steps:
    1. Remove the original Reshape and Einsum nodes.
    2. Insert a sequence of two Reshape + Transpose operations to achieve equivalent behavior.
    3. Update downstream nodes to consume the output of the new sequence.

    Args:
        model (onnx.ModelProto): The input ONNX model to modify.

    Returns:
        onnx.ModelProto: The modified ONNX model.
    """
    # Find the original Reshape node and its index in the graph
    reshape_node = oh.find_node(model, "/Reshape")
    reshape_idx = list(model.graph.node).index(reshape_node)

    # Remove the original Reshape and Einsum nodes
    oh.remove_node(model, "/Reshape")
    oh.remove_node(model, "/Einsum")

    # First Reshape
    shape_1 = helper.make_tensor(
        name='shape1',
        data_type=onnx.TensorProto.INT64,
        dims=[4],
        vals=[1, 196, 16, 16]
    )
    reshape_1 = helper.make_node(
        'Reshape',
        name="/Reshape1",
        inputs=['/mae_model/Slice_5_output_0', 'shape1'],
        outputs=['/Reshape1_out']
    )

    # Add initializer and insert the node
    model.graph.initializer.append(shape_1)
    model.graph.node.insert(reshape_idx, reshape_1)

    # First Transpose
    transpose_1 = helper.make_node(
        'Transpose',
        name="/Transpose1",
        inputs=['/Reshape1_out'],
        outputs=['/Transpose1_out'],
        perm=[0, 2, 1, 3]  # Permute dimensions
    )
    model.graph.node.insert(reshape_idx + 1, transpose_1)

    # Second Reshape
    shape_2 = helper.make_tensor(
        name='shape2',
        data_type=onnx.TensorProto.INT64,
        dims=[4],
        vals=[1, 16, 14, 224]
    )
    reshape_2 = helper.make_node(
        'Reshape',
        name='/Reshape2',
        inputs=['/Transpose1_out', 'shape2'],
        outputs=['/Reshape2_out']
    )

    model.graph.initializer.append(shape_2)
    model.graph.node.insert(reshape_idx + 2, reshape_2)

    # Second Transpose
    transpose_2 = helper.make_node(
        'Transpose',
        name='/Transpose2',
        inputs=['/Reshape2_out'],
        outputs=['/Transpose2_out'],
        perm=[0, 2, 1, 3]  # Permute dimensions again
    )
    model.graph.node.insert(reshape_idx + 3, transpose_2)

    # Update downstream nodes (connect last Reshape node in the graph)
    last_reshape = oh.find_node(model, "/Reshape_1")
    last_reshape.input[0] = "/Transpose2_out"

    return model


def main():
    onnx_model_path = "../exported_onnx_models/mae_brats_deterministic.onnx"
    output_model_path = "../exported_onnx_models/mae_brats_deterministic_grid_masking_simplified.onnx"

    # Load the original ONNX model
    model = oh.load_model(onnx_model_path)

    # Replace constant GatherElements nodes with simpler Gather nodes
    rewrite_gather_elements_with_gather(model, "/mae_model/GatherElements_1")
    rewrite_gather_elements_with_gather(model, "/mae_model/GatherElements_2")

    # Rewrite 6D Reshape + 6D Einsum pattern to simpler Reshape + Transpose nodes
    rewrite_reshape_6d_and_einsum_6d_part(model)

    # Remove the 'mask' output from the model
    # This output does not appear in Netron visualization but exists in the graph.
    # Likely auto-added during ONNX export.
    oh.remove_output(model, keep_list=["reconstruction"])

    # Remove any stale inferred shape information
    oh.remove_infer_shape(model)

    # Save the modified ONNX model
    oh.save_model(model, output_model_path)

if __name__ == "__main__":
    main()