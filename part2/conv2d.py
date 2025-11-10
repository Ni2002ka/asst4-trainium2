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
    assert pool_size == 1

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


    # Process the images in batches
    for b in nl.affine_range(batch_size):

        for m in nl.affine_range(M // TILE_M):

            for n in nl.affine_range(N // TILE_N):
                # Figure out the mapping of the flattened arrays
                row_idx = n * TILE_N // out_width
                col_start = n * TILE_N % out_width

                res_psum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)

                for k in nl.affine_range(K // TILE_K):
                    # Iterate over the filter height
                    for i in nl.affine_range(filter_height):
                        # Iterate over the filter width
                        for j in nl.affine_range(filter_width):
                            lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=W.dtype, buffer=nl.sbuf)
                            rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=X.dtype, buffer=nl.sbuf)
                            w_block = W[m * TILE_M:(m + 1) * TILE_M, k * TILE_K:(k + 1) * TILE_K, i, j]
                            w_block_sbuf = nl.ndarray(
                                        shape=(TILE_M, TILE_K),
                                        dtype=W.dtype,
                                        buffer=nl.sbuf,
                                    )
                            nisa.dma_copy(dst=w_block_sbuf, src=w_block)
                            w_block_transposed = nisa.nc_transpose(w_block_sbuf) # This lives in psum
                            lhsT_tile = nisa.tensor_copy(w_block_transposed) # Move to sbuf

                            # Shift the Input tensor by (i, j) to align with the filter's current position
                            input_shifted = X[b, :, i:i+out_height, j:j+out_width]

                            # Load tiles from lhsT and rhs
                            # nisa.dma_copy(dst=lhsT_tile, src=w_block_transposed)
                            nisa.dma_copy(dst=rhs_tile, src=input_shifted[k * TILE_K:(k + 1) * TILE_K, row_idx, col_start:col_start+TILE_N])

                            # Accumulate partial-sums into PSUM
                            res_psum += nisa.nc_matmul(lhsT_tile, rhs_tile)

                # Copy block m,n to output. PSUM->SBUF
                bias_tile_sbuf = nl.zeros(TILE_M, dtype=bias.dtype, buffer=nl.sbuf)
                # res_sb = nisa.tensor_copy(res_psum)
                res_sb = nl.zeros((TILE_M, TILE_N), dtype=bias.dtype, buffer=nl.sbuf)
                for nn in nl.affine_range(TILE_N):
                    res_sb = nisa.tensor_tensor(res_psum[:, nn], bias_tile_sbuf, op=nl.add)
                    
                nisa.dma_copy(dst=X_out[b, m * TILE_M:(m + 1) * TILE_M, row_idx, col_start:col_start+TILE_N], src=res_sb)
                
        # TODO: implement pooling


    return X_out


