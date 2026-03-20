"""
convert.py - Convert a Keras NNAA model (.keras) to a ReShade FX compute shader.

This script loads the trained neural network anti-aliasing model and generates
the equivalent HLSL compute shader code that can run in ReShade.

The architecture is:
  Conv2D(32, 8x8, stride=2) + PReLU  ->  Space-to-Depth optimization
  Conv2D(32, 3x3, stride=1) + PReLU  ->  Standard 3x3 conv
  Conv2D(32, 3x3, stride=1) + PReLU  ->  Standard 3x3 conv
  Conv2D(32, 3x3, stride=1) + PReLU  ->  Fused with final layer
  Conv2DTranspose(1, 2x2, stride=2)  ->  Depth-to-Space (fused above)

Usage:
  python convert.py [model_path] [output_path]
  python convert.py                          # defaults: nnaa.keras -> out_nnaa.fx
"""

import sys
import numpy as np

def load_model_weights(model_path):
    """Load Keras model and extract all layer weights in order."""
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path)
    model.summary()

    layers = []
    for layer in model.layers:
        weights = layer.get_weights()
        if len(weights) > 0:
            layers.append({
                'name': layer.name,
                'type': layer.__class__.__name__,
                'weights': weights
            })
    return layers


def fmt(v):
    """Format a float value as a string, matching the original shader precision."""
    return repr(float(v))


def generate_header():
    """Generate the license, includes, defines, and texture/storage declarations."""
    return """/**
 * MIT License
 * 
 * Copyright (c) 2025 Leo Calvis
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "ReShadeUI.fxh"
#include "ReShade.fxh"

#define K_SIZE 128

texture2D texLuma_nnaa1
{
    Width = BUFFER_WIDTH / 2 + 1;
    Height = BUFFER_HEIGHT / 2 + 1;
    MipLevels = 0;
    Format = RGBA16F;
};

storage2D storageLuma_nnaa1
{
    Texture = texLuma_nnaa1;
    MipLevel = 0;
};

texture2D texTarget0_nnaa1
{
    Width = (BUFFER_WIDTH / 2);
    Height = (BUFFER_HEIGHT / 2) * 8;
    MipLevels = 0;
    Format = RGBA16F;
};

storage2D storageTarget0_nnaa1
{
    Texture = texTarget0_nnaa1;
    MipLevel = 0;
};

texture2D texTarget1_nnaa1
{
    Width = (BUFFER_WIDTH / 2);
    Height = (BUFFER_HEIGHT / 2) * 8;
    MipLevels = 0;
    Format = RGBA16F;
};

storage2D storageTarget1_nnaa1
{
    Texture = texTarget1_nnaa1;
    MipLevel = 0;
};

texture2D texResult_nnaa1
{
    Width = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
    MipLevels = 0;
    Format = R16F;
};

storage2D storageResult_nnaa1
{
    Texture = texResult_nnaa1;
    MipLevel = 0;
};

sampler2D samplerResult_nnaa1
{
\tTexture = texResult_nnaa1;
};

[shader("compute")]
void GetLuma(int3 id : SV_DispatchThreadID)
{
    min16float4 luma = float4(
        dot(tex2Dfetch(ReShade::BackBuffer, id.xy * int2(2, 2) + int2(-1, -1)).rgb, min16float3(0.299, 0.587, 0.114)),
        dot(tex2Dfetch(ReShade::BackBuffer, id.xy * int2(2, 2) + int2(0, -1)).rgb, min16float3(0.299, 0.587, 0.114)),
        dot(tex2Dfetch(ReShade::BackBuffer, id.xy * int2(2, 2) + int2(-1, 0)).rgb, min16float3(0.299, 0.587, 0.114)),
        dot(tex2Dfetch(ReShade::BackBuffer, id.xy * int2(2, 2)).rgb, min16float3(0.299, 0.587, 0.114)));
    tex2Dstore(storageLuma_nnaa1, id.xy, luma);
}

[shader("pixel")]
float4 ApplyNN(float4 pos : SV_Position) : SV_Target
{
\tmin16float luma = tex2Dfetch(samplerResult_nnaa1, pos.xy).r;
\t
\tmin16float4 old_color = tex2Dfetch(ReShade::BackBuffer, pos.xy);
\t
\tmin16float y = dot(old_color.rgb, min16float3(0.299, 0.587, 0.114)) + luma;
\tmin16float cb = dot(old_color.rgb, min16float3(-0.1687, -0.3313, 0.5));
\tmin16float cr = dot(old_color.rgb, min16float3(0.5, -0.4187, -0.0813));
\t
\treturn float4(y + 1.402 * cr, y - 0.34414 * cb - 0.71414 * cr, y + 1.772 * cb, old_color.a);
}

"""


def generate_first_conv_layer(conv_kernel, conv_bias, prelu_alpha):
    """
    Generate Layer_conv2d: the first Conv2D(32, 8x8, stride=2) + PReLU.
    
    The original 8x8 kernel with stride 2 on a single-channel full-res image
    is equivalent to a 4x4 kernel with stride 1 on a 4-channel half-res image
    (Space-to-Depth transformation).
    
    The luma texture packs 4 neighboring pixels as RGBA:
      R = pixel(-1,-1), G = pixel(0,-1), B = pixel(-1,0), A = pixel(0,0)
    
    The 8x8 kernel is reshaped: kernel[ky*2+sub_y, kx*2+sub_x, 0, out_ch]
    maps to offset (kx-1, ky-1) in the luma texture, channel sub_x + sub_y*2
    (but the luma channels are: R=(-1,-1), G=(0,-1), B=(-1,0), A=(0,0))
    So sub_channel mapping: (sub_x=0,sub_y=0)->R, (sub_x=1,sub_y=0)->G,
                             (sub_x=0,sub_y=1)->B, (sub_x=1,sub_y=1)->A
    Which means sub_channel_index = sub_y * 2 + sub_x maps to xyzw directly.
    
    Actually looking at the luma fetch pattern:
      luma.x = pixel at (id.xy*2 + (-1,-1))  
      luma.y = pixel at (id.xy*2 + (0,-1))
      luma.z = pixel at (id.xy*2 + (-1,0))
      luma.w = pixel at (id.xy*2 + (0,0))
    
    And the conv offsets go from (-1,-1) to (2,2) in luma space (4x4 = 16 fetches).
    Each fetch reads 4 sub-pixels (RGBA). So effectively we cover:
      full-res x: from (id.x*2 + (-1)*2 + (-1)) to (id.x*2 + 2*2 + 1)
      = id.x*2 - 3  to  id.x*2 + 5  => 9 positions but kernel is 8, so range is 8.
    
    Actually: the 8x8 kernel in TF has shape [ky, kx, 1, 32].
    With stride=2 on full res: output[oy,ox] = sum over ky=0..7, kx=0..7 of
      input[oy*2+ky, ox*2+kx] * kernel[ky, kx, 0, :]
    
    With the Space-to-Depth packing, luma at (lx, ly) contains:
      .x = input[ly*2-1, lx*2-1]   (offset -1,-1 relative to the "center" ly*2, lx*2)
      .y = input[ly*2-1, lx*2]     (offset 0,-1)
      .z = input[ly*2,   lx*2-1]   (offset -1,0)
      .w = input[ly*2,   lx*2]     (offset 0,0)
    
    The output thread id maps to half-res: oy=id.y, ox=id.x.
    input[oy*2+ky, ox*2+kx] = luma at position
      lx = (ox*2+kx+1)//2,  channel depends on (ox*2+kx+1)%2 and (oy*2+ky+1)%2
    
    Actually, let me just directly observe what the original does:
    It fetches luma at id.xy + int2(dx, dy) for dx in {-1,0,1,2}, dy in {-1,0,1,2}.
    That's 4x4 = 16 fetches. Each fetch is 4 channels (xyzw).
    So 64 input values total. The 8x8 kernel also has 64 values per output channel.
    
    For offset (dx,dy) channel c (0=x,1=y,2=z,3=w):
      c=0 (.x): pixel at full-res ((id.x+dx)*2-1, (id.y+dy)*2-1)
      c=1 (.y): pixel at full-res ((id.x+dx)*2,   (id.y+dy)*2-1)
      c=2 (.z): pixel at full-res ((id.x+dx)*2-1, (id.y+dy)*2)
      c=3 (.w): pixel at full-res ((id.x+dx)*2,   (id.y+dy)*2)
    
    So full-res position relative to id*2:
      fx = dx*2 + (1 if c in {1,3} else -1) = dx*2 + (c%2)*2 - 1  => wait:
      c=0: fx = dx*2-1, fy = dy*2-1
      c=1: fx = dx*2,   fy = dy*2-1
      c=2: fx = dx*2-1, fy = dy*2
      c=3: fx = dx*2,   fy = dy*2
    
    Relative to output center (id.x*2, id.y*2), the full-res input coordinate is:
      kx = fx = dx*2 + (c%2==1 ? 0 : -1)
      ky = fy = dy*2 + (c//2==1 ? 0 : -1)
    
    Wait, let me simplify. We have kx = dx*2 - 1 + (c % 2), ky = dy*2 - 1 + (c // 2).
    
    The TF conv2d with stride 2: output[id.y, id.x] = sum_{ky=0..7, kx=0..7} 
      input[id.y*2 + ky - pad, id.x*2 + kx - pad] * kernel[ky, kx, 0, out_ch]
    With padding='same' and stride=2, pad = (kernel_size - 1) / 2 = 3.5, 
    but TF uses asymmetric padding for even kernels. For 'same' with stride s:
      out_size = ceil(in_size / s)
      pad_total = max(0, (out_size - 1) * s + kernel_size - in_size)
      pad_before = pad_total // 2
    For stride=2, kernel=8: pad_total = 6, pad_before = 3.
    
    So: output[oy, ox] = sum_{ky=0..7, kx=0..7}
      input[oy*2 + ky - 3, ox*2 + kx - 3] * kernel[ky, kx, 0, out_ch]
    
    Now matching: the fetch at luma offset (dx, dy) channel c gives us 
      input at (id.x*2 + dx*2 - 1 + c%2, id.y*2 + dy*2 - 1 + c//2)
    
    For this to equal input[id.y*2 + ky - 3, id.x*2 + kx - 3]:
      kx - 3 = dx*2 - 1 + (c % 2)  =>  kx = dx*2 + 2 + (c % 2)
      ky - 3 = dy*2 - 1 + (c // 2)  =>  ky = dy*2 + 2 + (c // 2)
    
    With dx in {-1,0,1,2}: kx ranges from 0+c%2 to 6+c%2. 
    With c%2 in {0,1}: kx ranges from 0 to 7. ✓
    Same for ky.
    
    So: kernel_index for fetch (dx, dy, c) is:
      kx = dx*2 + 2 + (c % 2)
      ky = dy*2 + 2 + (c // 2)
      weight = kernel[ky, kx, 0, out_ch]
    """
    lines = []
    lines.append('[shader("compute")]')
    lines.append('void Layer_conv2d(int3 id : SV_DispatchThreadID)')
    lines.append('{')
    lines.append('    if(id.x >= (BUFFER_WIDTH / 2)) return;')

    num_out = conv_bias.shape[0]  # 32
    num_groups = num_out // 4  # 8 groups of 4 channels

    # Initialize f_out with biases
    for g in range(num_groups):
        b = conv_bias[g*4:(g+1)*4]
        lines.append(f'    min16float4 f_out_{g} = min16float4({fmt(b[0])},{fmt(b[1])},{fmt(b[2])},{fmt(b[3])});')
    
    lines.append('    min16float4 f_in;')

    # Iterate over 4x4 spatial neighborhood in luma space
    for dy in range(-1, 3):
        for dx in range(-1, 3):
            lines.append(f'    f_in = tex2Dfetch(storageLuma_nnaa1, id.xy + int2({float(dx)}, {float(dy)}));')
            
            # For each input channel c (0=x, 1=y, 2=z, 3=w)
            for c in range(4):
                kx = dx * 2 + 2 + (c % 2)
                ky = dy * 2 + 2 + (c // 2)
                
                channel_name = ['x', 'y', 'z', 'w'][c]
                
                # For each output group of 4
                for g in range(num_groups):
                    w = conv_kernel[ky, kx, 0, g*4:(g+1)*4]
                    lines.append(f'    f_out_{g} += min16float4({fmt(w[0])},{fmt(w[1])},{fmt(w[2])},{fmt(w[3])}) * f_in.{channel_name};')

    # PReLU activation + store
    # prelu_alpha shape is (1, 1, 32)
    alpha = prelu_alpha.reshape(-1)
    for g in range(num_groups):
        for c_idx, c_name in enumerate(['x', 'y', 'z', 'w']):
            ch = g * 4 + c_idx
            lines.append(f'    if(f_out_{g}.{c_name} < 0)')
            lines.append(f'        f_out_{g}.{c_name} *= {fmt(alpha[ch])};')
        lines.append(f'    tex2Dstore(storageTarget0_nnaa1, id.xy * int2(1, 8) + int2(0, {g}), f_out_{g});')

    lines.append('}')
    return '\n'.join(lines)


def generate_mid_conv_layer(layer_index, conv_kernel, conv_bias, prelu_alpha, 
                            is_last_hidden=False, final_kernel=None, final_bias=None):
    """
    Generate a middle Conv2D(32, 3x3, stride=1) + PReLU layer.
    
    For 3x3 convolution on 32 channels stored in 8 rows of RGBA:
    - Input is read from one storage texture, output written to the other (ping-pong)
    - Each spatial offset (dx, dy) in {-1,0,1} maps to reading 8 rows of 4 channels
    - So 3x3 spatial * 8 rows = 72 fetches, each with 4 channel multiplies
    
    For the last hidden layer (conv2d_3), we fuse the Conv2DTranspose output layer
    directly after the PReLU, producing 4 output pixels in Depth-to-Space fashion.
    """
    # Determine ping-pong: even layer_index reads from Target0, writes to Target1
    # layer_index: 1 = conv2d_1, 2 = conv2d_2, 3 = conv2d_3
    if layer_index % 2 == 1:
        src_storage = 'storageTarget0_nnaa1'
        dst_storage = 'storageTarget1_nnaa1'
    else:
        src_storage = 'storageTarget1_nnaa1'
        dst_storage = 'storageTarget0_nnaa1'

    func_name = f'Layer_conv2d_{layer_index}'
    
    lines = []
    lines.append(f'[shader("compute")]')
    lines.append(f'void {func_name}(int3 id : SV_DispatchThreadID)')
    lines.append('{')
    lines.append('    if(id.x >= (BUFFER_WIDTH / 2)) return;')

    num_out = conv_bias.shape[0]  # 32
    num_groups = num_out // 4  # 8

    # Initialize f_out with biases
    for g in range(num_groups):
        b = conv_bias[g*4:(g+1)*4]
        lines.append(f'    min16float4 f_out_{g} = min16float4({fmt(b[0])},{fmt(b[1])},{fmt(b[2])},{fmt(b[3])});')
    
    lines.append('    min16float4 f_in;')

    # 3x3 spatial convolution over 32 input channels (8 groups of 4)
    # conv_kernel shape: (3, 3, 32, 32)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            for in_g in range(num_groups):  # input group (0..7)
                row_offset = dy * 8 + in_g
                lines.append(f'    f_in = tex2Dfetch({src_storage}, id.xy * int2(1, 8) + int2({float(dx)}, {float(row_offset)}));')
                
                # For each input sub-channel c (0=x, 1=y, 2=z, 3=w)
                for c in range(4):
                    in_ch = in_g * 4 + c
                    ky = dy + 1  # kernel index: 0,1,2
                    kx = dx + 1
                    
                    channel_name = ['x', 'y', 'z', 'w'][c]
                    
                    # For each output group
                    for out_g in range(num_groups):
                        w = conv_kernel[ky, kx, in_ch, out_g*4:(out_g+1)*4]
                        lines.append(f'    f_out_{out_g} += min16float4({fmt(w[0])},{fmt(w[1])},{fmt(w[2])},{fmt(w[3])}) * f_in.{channel_name};')

    # PReLU activation
    alpha = prelu_alpha.reshape(-1)
    for g in range(num_groups):
        for c_idx, c_name in enumerate(['x', 'y', 'z', 'w']):
            ch = g * 4 + c_idx
            lines.append(f'    if(f_out_{g}.{c_name} < 0)')
            lines.append(f'        f_out_{g}.{c_name} *= {fmt(alpha[ch])};')

    if is_last_hidden and final_kernel is not None:
        # Fuse the Conv2DTranspose(1, 2x2, stride=2) as a Depth-to-Space
        # final_kernel shape: (2, 2, 1, 32) for Conv2DTranspose
        # final_bias shape: (1,)
        # 
        # Conv2DTranspose with stride=2 and kernel 2x2 maps each input pixel to
        # a 2x2 output block. The kernel maps:
        #   output[oy*2+0, ox*2+0] += sum_c input[oy, ox, c] * kernel[0, 0, 0, c]
        #   output[oy*2+1, ox*2+0] += sum_c input[oy, ox, c] * kernel[1, 0, 0, c]
        #   output[oy*2+0, ox*2+1] += sum_c input[oy, ox, c] * kernel[0, 1, 0, c]
        #   output[oy*2+1, ox*2+1] += sum_c input[oy, ox, c] * kernel[1, 1, 0, c]
        #
        # So f_out.x = pixel(0,0), f_out.y = pixel(1,0), f_out.z = pixel(0,1), f_out.w = pixel(1,1)
        # This means xyzw map to:
        #   x -> (0,0) -> kernel[0,0,0,:]
        #   y -> (1,0) -> kernel[0,1,0,:]
        #   z -> (0,1) -> kernel[1,0,0,:]
        #   w -> (1,1) -> kernel[1,1,0,:]
        
        fb = float(final_bias[0])
        lines.append(f'')
        lines.append(f'    min16float4 f_out = min16float4({fmt(fb)},{fmt(fb)},{fmt(fb)},{fmt(fb)});')
        
        for g in range(num_groups):
            lines.append(f'    f_in = f_out_{g};')
            for c in range(4):
                in_ch = g * 4 + c
                c_name = ['x', 'y', 'z', 'w'][c]
                # Output mapping: x->(0,0), y->(1,0), z->(0,1), w->(1,1)
                w_x = float(final_kernel[0, 0, 0, in_ch])
                w_y = float(final_kernel[0, 1, 0, in_ch])
                w_z = float(final_kernel[1, 0, 0, in_ch])
                w_w = float(final_kernel[1, 1, 0, in_ch])
                lines.append(f'    f_out += min16float4({fmt(w_x)},{fmt(w_y)},{fmt(w_z)},{fmt(w_w)}) * f_in.{c_name};')

        lines.append(f'    tex2Dstore(storageResult_nnaa1, (id.xy * 2) + uint2(0, 0), f_out.x);')
        lines.append(f'    tex2Dstore(storageResult_nnaa1, (id.xy * 2) + uint2(1, 0), f_out.y);')
        lines.append(f'    tex2Dstore(storageResult_nnaa1, (id.xy * 2) + uint2(0, 1), f_out.z);')
        lines.append(f'    tex2Dstore(storageResult_nnaa1, (id.xy * 2) + uint2(1, 1), f_out.w);')
    else:
        # Store to destination texture
        for g in range(num_groups):
            lines.append(f'    tex2Dstore({dst_storage}, id.xy * int2(1, 8) + int2(0, {g}), f_out_{g});')

    lines.append('}')
    return '\n'.join(lines)


def generate_technique():
    """Generate the technique block that dispatches all passes."""
    return """

#if (BUFFER_WIDTH % (2 * K_SIZE)) == 0
#define X_DISPATCH (BUFFER_WIDTH / (2 * K_SIZE))
#else
#define X_DISPATCH ((BUFFER_WIDTH / (2 * K_SIZE)) + 1)
#endif

#define Y_DISPATCH (BUFFER_HEIGHT / 2)


technique Sarenya_NNAA < ui_tooltip = "Sar\\xe9nya NNAA"; >
{
    pass luma
    {
        ComputeShader = GetLuma<16, 16>;

        DispatchSizeX = BUFFER_WIDTH / (2 * 16) + 1;
        DispatchSizeY = BUFFER_HEIGHT / (2 * 16) + 1;
    }


    pass pass_conv2d
    {
        ComputeShader = Layer_conv2d<K_SIZE, 1>;

        DispatchSizeX = X_DISPATCH;
        DispatchSizeY = Y_DISPATCH;
    }

    pass pass_conv2d_1
    {
        ComputeShader = Layer_conv2d_1<K_SIZE, 1>;

        DispatchSizeX = X_DISPATCH;
        DispatchSizeY = Y_DISPATCH;
    }

    pass pass_conv2d_2
    {
        ComputeShader = Layer_conv2d_2<K_SIZE, 1>;

        DispatchSizeX = X_DISPATCH;
        DispatchSizeY = Y_DISPATCH;
    }

    pass pass_conv2d_3
    {
        ComputeShader = Layer_conv2d_3<K_SIZE, 1>;

        DispatchSizeX = X_DISPATCH;
        DispatchSizeY = Y_DISPATCH;
    }

    pass apply_nn
\t{
\t\tVertexShader = PostProcessVS;
\t\tPixelShader = ApplyNN;
\t}
}
"""


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'nnaa.keras'
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'out_nnaa.fx'

    print(f"Loading model from: {model_path}")
    layers = load_model_weights(model_path)

    # Print layer info
    print("\nExtracted layers:")
    for i, layer in enumerate(layers):
        shapes = [w.shape for w in layer['weights']]
        print(f"  [{i}] {layer['name']} ({layer['type']}): {shapes}")

    # Map layers by their roles:
    # 0: conv2d        -> kernel (8,8,1,32), bias (32,)
    # 1: p_re_lu       -> alpha (1,1,32)
    # 2: conv2d_1      -> kernel (3,3,32,32), bias (32,)
    # 3: p_re_lu_1     -> alpha (1,1,32)
    # 4: conv2d_2      -> kernel (3,3,32,32), bias (32,)
    # 5: p_re_lu_2     -> alpha (1,1,32)
    # 6: conv2d_3      -> kernel (3,3,32,32), bias (32,)
    # 7: p_re_lu_3     -> alpha (1,1,32)
    # 8: conv2d_final  -> kernel (2,2,1,32), bias (1,)

    conv0_kernel, conv0_bias = layers[0]['weights']
    prelu0_alpha = layers[1]['weights'][0]

    conv1_kernel, conv1_bias = layers[2]['weights']
    prelu1_alpha = layers[3]['weights'][0]

    conv2_kernel, conv2_bias = layers[4]['weights']
    prelu2_alpha = layers[5]['weights'][0]

    conv3_kernel, conv3_bias = layers[6]['weights']
    prelu3_alpha = layers[7]['weights'][0]

    final_kernel, final_bias = layers[8]['weights']

    print("\nGenerating shader code...")

    parts = []
    parts.append(generate_header())

    print("  - Layer conv2d (8x8 stride 2, Space-to-Depth)...")
    parts.append(generate_first_conv_layer(conv0_kernel, conv0_bias, prelu0_alpha))

    print("  - Layer conv2d_1 (3x3 stride 1)...")
    parts.append('\n')
    parts.append(generate_mid_conv_layer(1, conv1_kernel, conv1_bias, prelu1_alpha))

    print("  - Layer conv2d_2 (3x3 stride 1)...")
    parts.append('\n')
    parts.append(generate_mid_conv_layer(2, conv2_kernel, conv2_bias, prelu2_alpha))

    print("  - Layer conv2d_3 (3x3 stride 1) + fused Conv2DTranspose (Depth-to-Space)...")
    parts.append('\n')
    parts.append(generate_mid_conv_layer(3, conv3_kernel, conv3_bias, prelu3_alpha,
                                         is_last_hidden=True, 
                                         final_kernel=final_kernel,
                                         final_bias=final_bias))

    parts.append(generate_technique())

    shader_code = '\n'.join(parts)

    print(f"\nWriting shader to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(shader_code)

    print(f"Done! Generated {len(shader_code)} bytes, {shader_code.count(chr(10))} lines.")


if __name__ == '__main__':
    main()
