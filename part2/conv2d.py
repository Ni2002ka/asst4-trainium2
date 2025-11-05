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
    # TILE_N = nl.tile_size.gemm_moving_fmax  # 512
    TILE_N = 12
    
    output_flat = nl.ndarray(shape=(out_channels, out_height * out_width),
                        dtype=X.dtype,
                        buffer=nl.hbm,)

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

                res_psum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)
                for k in nl.affine_range(K // TILE_K):
                    lhsT_tile = nl.ndarray((TILE_M, TILE_K), dtype=W.dtype, buffer=nl.sbuf)
                    rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=X.dtype, buffer=nl.sbuf)
                    # Iterate over the filter height
                    for i in nl.affine_range(filter_height):
                        # Iterate over the filter width
                        for j in nl.affine_range(filter_width):
                            w_block = W[m * TILE_M:(m + 1) * TILE_M, k * TILE_K:(k + 1) * TILE_K, i, j]

                            # Shift the Input tensor by (i, j) to align with the filter's current position
                            input_shifted = X[b, :, i:i+out_height, j:j+out_width]
                            # Flatten the shifted input before the matmul
                            input_shifted_flat = input_shifted.reshape((in_channels, out_height * out_width))

                            # Load tiles from lhsT and rhs
                            nisa.dma_copy(dst=lhsT_tile, src=w_block)
                            nisa.dma_copy(dst=rhs_tile, src=input_shifted_flat[k * TILE_K:(k + 1) * TILE_K, n * TILE_N:(n + 1) * TILE_N])

                            # Accumulate partial-sums into PSUM
                            res_psum += nisa.nc_matmul(lhsT_tile, rhs_tile, is_transpose=True)

                            # accumulate in total output
                # Copy block m,n to output. PSUM->SBUF->output_flat
                res_sb = nisa.tensor_copy(res_psum)
                # row_idx = n * TILE_N // out_pool_width
                # col_start = n * TILE_N % out_pool_width
                # nisa.dma_copy(dst=X_out[b, m * TILE_M:(m + 1) * TILE_M, row_idx, col_start:col_start+TILE_N], src=res_sb)
                nisa.dma_copy(dst=output_flat[m * TILE_M:(m + 1) * TILE_M, n * TILE_N:(n + 1) * TILE_N], src=res_sb)
        # TODO: implement pooling
        # out_img = output_flat.reshape((out_channels, out_height, out_width))
        out_img = output_flat.reshape((out_channels, out_height, out_width))

        # Allocate a scratch buffer of the same shape
        tmp_sbuf = nl.ndarray(out_img.shape, dtype=out_img.dtype, buffer=nl.sbuf)

        # Copy HBM -> SBUF -> HBM
        nl.device_print("bruh", tmp_sbuf)
        nisa.dma_copy(dst=tmp_sbuf, src=out_img)
        nisa.dma_copy(dst=X_out[b, :, :, :], src=tmp_sbuf)
        # nisa.dma_copy(dst=X_out[b, :, :, :], src=out_img)

    return X_out
