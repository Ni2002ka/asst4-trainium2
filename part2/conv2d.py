import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal


"""
A fused convolution - maxpool kernel that you need to implement for Part 2.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.
    pool_size: the size of the pool filter and pool stride.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: pool_size == 1 || pool_size == 2
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height // pool_size
out_pool_width = out_width // pool_size

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""
@nki.compiler.skip_middle_end_transformations
@nki.jit
def fused_conv2d_maxpool(X, W, bias, pool_size=1):

    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height // pool_size
    out_pool_width = out_width // pool_size
    
    # Can assume multiple of 128 to avoid using mask
    assert in_channels % 128 == out_channels % 128 == 0

    # TODO: add pooling
    # assert pool_size == 1

    # Can assume one PSUM bank can at least fit one row of the pixels
    assert nl.tile_size.gemm_moving_fmax >= out_width

    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Various tiling dimensions (You may want to define more of them)
    c_in_pmax = nl.tile_size.pmax
    n_tiles_c_in = in_channels // c_in_pmax
    TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
    TILE_K = nl.tile_size.pmax  # 128
    TILE_N = out_width # Can do this because PSUM can fit a full row

    # Define matmul shapes
    K = in_channels
    M = out_channels
    N = out_height * out_width

    assert M % TILE_M == 0, "out_channels not multiple of TILE_M"
    assert K % TILE_K == 0, "in_channels not multiple of TILE_K"
    assert N % TILE_N == 0, "out_height*out_width not multiple of TILE_N"

    bias_sbuf = nl.ndarray((TILE_M, M // TILE_M), dtype=bias.dtype, buffer=nl.sbuf)
    w_transposed_sbuf = nl.ndarray((TILE_K, K // TILE_K, TILE_M, M // TILE_M, filter_height, filter_width), dtype=W.dtype, buffer=nl.sbuf)
    # Keep W and bias in sbuf 
    for m in nl.affine_range(M // TILE_M):
        nisa.dma_copy(dst=bias_sbuf[:, m], src=bias[m * TILE_M:(m + 1) * TILE_M])
        for k in nl.affine_range(K // TILE_K):
            for i in nl.affine_range(filter_height):
                for j in nl.affine_range(filter_width):
                    w_block = W[m * TILE_M:(m + 1) * TILE_M, k * TILE_K:(k + 1) * TILE_K, i, j]
                    w_block_sbuf = nl.ndarray(
                                shape=(TILE_M, TILE_K),
                                dtype=W.dtype,
                                buffer=nl.sbuf,
                            )
                    nisa.dma_copy(dst=w_block_sbuf, src=w_block)
                    w_block_transposed = nisa.nc_transpose(w_block_sbuf)
                    w_transposed_sbuf[:, k, :, m, i, j] = nisa.tensor_copy(w_block_transposed)
                    
    # Process the images in batches
    for b in nl.affine_range(batch_size):

        for m in nl.affine_range(M // TILE_M):

            bias_tile_sbuf = bias_sbuf[:, m]
            bias_broadcast = nl.ndarray((TILE_M, TILE_N), dtype=bias.dtype, buffer=nl.sbuf)
            for col in nl.affine_range(TILE_N):
                bias_broadcast[:, col] = bias_tile_sbuf

            if pool_size == 2:
                prev_pool_row = nl.ndarray((TILE_M, TILE_N // pool_size, 1), dtype=bias.dtype, buffer=nl.sbuf) 
            # Tile one row at a time
            for row_idx in nl.affine_range(out_height):
                res_psum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)

                for k in nl.affine_range(K // TILE_K):
                    # Iterate over the filter height
                    for i in nl.affine_range(filter_height):
                        # Iterate over the filter width
                        for j in nl.affine_range(filter_width):
                            lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=W.dtype, buffer=nl.sbuf)
                            rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=X.dtype, buffer=nl.sbuf)


                            # Shift the Input tensor by (i, j) to align with the filter's current position
                            input_shifted = X[b, :, i:i+out_height, j:j+out_width]
                            lhsT_tile = w_transposed_sbuf[:, k, :, m, i, j]

                            # Load tiles from lhsT and rhs
                            nisa.dma_copy(dst=rhs_tile, src=input_shifted[k * TILE_K:(k + 1) * TILE_K, row_idx, 0:out_width])

                            # Accumulate partial-sums into PSUM
                            res_psum += nisa.nc_matmul(lhsT_tile, rhs_tile)

                res_sb = nisa.tensor_tensor(res_psum, bias_broadcast, dtype=bias.dtype, op=nl.add)
                # Maxpool
                if pool_size == 2:
                    pooled_w = nisa.tensor_tensor(
                        res_sb[:, 0:out_width:2],
                        res_sb[:, 1:out_width:2],
                        op=nl.maximum
                    )

                    if (row_idx % 2) == 0:
                        # store even row for next pooling step
                        prev_pool_row[:, :, row_idx % 2] = pooled_w 
                    else:
                        # combine with the stored even row
                        pooled = nisa.tensor_tensor(prev_pool_row, pooled_w, op=nl.maximum)
                        nisa.dma_copy(
                            dst=X_out[b, m*TILE_M:(m+1)*TILE_M, row_idx//2, :],
                            src=pooled
                        )
                else:
                    nisa.dma_copy(
                        dst=X_out[b, m*TILE_M:(m+1)*TILE_M, row_idx, :],
                        src=res_sb
                    )



    return X_out


