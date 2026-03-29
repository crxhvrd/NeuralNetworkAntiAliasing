"""
convert.py - Convert a Keras NNAA model (.keras) to a ReShade FX compute shader.

Architecture:
  Conv2D(32, 8x8, stride=2) + PReLU  ->  Space-to-Depth
  Conv2D(32, 3x3, stride=1) + PReLU  ->  Standard conv (x3)
  Conv2DTranspose(1, 2x2, stride=2)  ->  Depth-to-Space (fused with last conv)

Usage:
  python convert.py [model_path] [output_path]
  python convert.py                          # defaults: nnaa.keras -> out_nnaa.fx

API:
  convert_model(model_path, output_path, log_fn=print)
"""

import sys
import numpy as np


# ============================================================================
# Model loading & validation
# ============================================================================

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


def validate_and_extract(layers):
    """
    Validate the model has the expected NNAA architecture and extract weights.
    
    Expected structure (9 weight-bearing layers):
      [0] Conv2D       -> kernel (8,8,1,32), bias (32,)
      [1] PReLU        -> alpha (1,1,32)
      [2] Conv2D       -> kernel (3,3,32,32), bias (32,)
      [3] PReLU        -> alpha (1,1,32)
      [4] Conv2D       -> kernel (3,3,32,32), bias (32,)
      [5] PReLU        -> alpha (1,1,32)
      [6] Conv2D       -> kernel (3,3,32,32), bias (32,)
      [7] PReLU        -> alpha (1,1,32)
      [8] Conv2DTranspose -> kernel (2,2,1,32), bias (1,)
    
    Returns a dict with all weights keyed by role.
    Raises ValueError with a clear message if validation fails.
    """
    if len(layers) < 9:
        raise ValueError(
            f"Expected 9 weight-bearing layers (4 Conv+PReLU pairs + 1 ConvTranspose), "
            f"but found {len(layers)}. Is this an NNAA model?"
        )

    expected = [
        ('Conv2D',          [(8, 8, 1, 32), (32,)]),
        ('PReLU',           [(1, 1, 32)]),
        ('Conv2D',          [(3, 3, 32, 32), (32,)]),
        ('PReLU',           [(1, 1, 32)]),
        ('Conv2D',          [(3, 3, 32, 32), (32,)]),
        ('PReLU',           [(1, 1, 32)]),
        ('Conv2D',          [(3, 3, 32, 32), (32,)]),
        ('PReLU',           [(1, 1, 32)]),
        ('Conv2DTranspose', [(2, 2, 1, 32), (1,)]),
    ]

    for i, (exp_type, exp_shapes) in enumerate(expected):
        actual_type = layers[i]['type']
        actual_shapes = [w.shape for w in layers[i]['weights']]
        
        if actual_type != exp_type:
            raise ValueError(
                f"Layer {i} ({layers[i]['name']}): expected type '{exp_type}', "
                f"got '{actual_type}'"
            )
        if actual_shapes != exp_shapes:
            raise ValueError(
                f"Layer {i} ({layers[i]['name']}): expected shapes {exp_shapes}, "
                f"got {actual_shapes}"
            )

    return {
        'conv0_kernel':  layers[0]['weights'][0],
        'conv0_bias':    layers[0]['weights'][1],
        'prelu0_alpha':  layers[1]['weights'][0],
        'conv1_kernel':  layers[2]['weights'][0],
        'conv1_bias':    layers[2]['weights'][1],
        'prelu1_alpha':  layers[3]['weights'][0],
        'conv2_kernel':  layers[4]['weights'][0],
        'conv2_bias':    layers[4]['weights'][1],
        'prelu2_alpha':  layers[5]['weights'][0],
        'conv3_kernel':  layers[6]['weights'][0],
        'conv3_bias':    layers[6]['weights'][1],
        'prelu3_alpha':  layers[7]['weights'][0],
        'final_kernel':  layers[8]['weights'][0],
        'final_bias':    layers[8]['weights'][1],
    }


# ============================================================================
# HLSL code generation
# ============================================================================

def _fmt(v):
    """Format a float value for HLSL, matching original shader precision."""
    return repr(float(v))


def generate_header():
    """Generate the shader header: license, includes, textures, utility functions."""
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
    Generate Layer_conv2d: Conv2D(32, 8x8, stride=2) + PReLU.
    Uses Space-to-Depth: the 8x8 kernel on 1-ch full-res becomes 
    a 4x4 kernel on 4-ch half-res luma texture.
    
    Kernel mapping: fetch at luma offset (dx,dy) channel c maps to
      kernel[dy*2+2+c//2, dx*2+2+c%2, 0, out_ch]
    """
    lines = []
    lines.append('[shader("compute")]')
    lines.append('void Layer_conv2d(int3 id : SV_DispatchThreadID)')
    lines.append('{')
    lines.append('    if(id.x >= (BUFFER_WIDTH / 2)) return;')

    num_out = conv_bias.shape[0]
    num_groups = num_out // 4

    for g in range(num_groups):
        b = conv_bias[g*4:(g+1)*4]
        lines.append(f'    min16float4 f_out_{g} = min16float4({_fmt(b[0])},{_fmt(b[1])},{_fmt(b[2])},{_fmt(b[3])});')
    
    lines.append('    min16float4 f_in;')

    for dy in range(-1, 3):
        for dx in range(-1, 3):
            lines.append(f'    f_in = tex2Dfetch(storageLuma_nnaa1, id.xy + int2({float(dx)}, {float(dy)}));')
            
            for c in range(4):
                kx = dx * 2 + 2 + (c % 2)
                ky = dy * 2 + 2 + (c // 2)
                channel_name = 'xyzw'[c]
                
                for g in range(num_groups):
                    w = conv_kernel[ky, kx, 0, g*4:(g+1)*4]
                    lines.append(f'    f_out_{g} += min16float4({_fmt(w[0])},{_fmt(w[1])},{_fmt(w[2])},{_fmt(w[3])}) * f_in.{channel_name};')

    alpha = prelu_alpha.reshape(-1)
    for g in range(num_groups):
        for c_idx in range(4):
            c_name = 'xyzw'[c_idx]
            ch = g * 4 + c_idx
            lines.append(f'    if(f_out_{g}.{c_name} < 0)')
            lines.append(f'        f_out_{g}.{c_name} *= {_fmt(alpha[ch])};')
        lines.append(f'    tex2Dstore(storageTarget0_nnaa1, id.xy * int2(1, 8) + int2(0, {g}), f_out_{g});')

    lines.append('}')
    return '\n'.join(lines)


def generate_mid_conv_layer(layer_index, conv_kernel, conv_bias, prelu_alpha, 
                            is_last_hidden=False, final_kernel=None, final_bias=None):
    """
    Generate a Conv2D(32, 3x3, stride=1) + PReLU layer with ping-pong storage.
    For the last hidden layer, fuses Conv2DTranspose as Depth-to-Space.
    """
    if layer_index % 2 == 1:
        src_storage = 'storageTarget0_nnaa1'
        dst_storage = 'storageTarget1_nnaa1'
    else:
        src_storage = 'storageTarget1_nnaa1'
        dst_storage = 'storageTarget0_nnaa1'

    func_name = f'Layer_conv2d_{layer_index}'
    num_out = conv_bias.shape[0]
    num_groups = num_out // 4
    
    lines = []
    lines.append(f'[shader("compute")]')
    lines.append(f'void {func_name}(int3 id : SV_DispatchThreadID)')
    lines.append('{')
    lines.append('    if(id.x >= (BUFFER_WIDTH / 2)) return;')

    for g in range(num_groups):
        b = conv_bias[g*4:(g+1)*4]
        lines.append(f'    min16float4 f_out_{g} = min16float4({_fmt(b[0])},{_fmt(b[1])},{_fmt(b[2])},{_fmt(b[3])});')
    
    lines.append('    min16float4 f_in;')

    for dy in range(-1, 2):
        for dx in range(-1, 2):
            for in_g in range(num_groups):
                row_offset = dy * 8 + in_g
                lines.append(f'    f_in = tex2Dfetch({src_storage}, id.xy * int2(1, 8) + int2({float(dx)}, {float(row_offset)}));')
                
                for c in range(4):
                    in_ch = in_g * 4 + c
                    ky = dy + 1
                    kx = dx + 1
                    channel_name = 'xyzw'[c]
                    
                    for out_g in range(num_groups):
                        w = conv_kernel[ky, kx, in_ch, out_g*4:(out_g+1)*4]
                        lines.append(f'    f_out_{out_g} += min16float4({_fmt(w[0])},{_fmt(w[1])},{_fmt(w[2])},{_fmt(w[3])}) * f_in.{channel_name};')

    alpha = prelu_alpha.reshape(-1)
    for g in range(num_groups):
        for c_idx in range(4):
            c_name = 'xyzw'[c_idx]
            ch = g * 4 + c_idx
            lines.append(f'    if(f_out_{g}.{c_name} < 0)')
            lines.append(f'        f_out_{g}.{c_name} *= {_fmt(alpha[ch])};')

    if is_last_hidden and final_kernel is not None:
        # Fuse Conv2DTranspose(1, 2x2, stride=2) as Depth-to-Space
        # Output mapping: x->(0,0), y->(1,0), z->(0,1), w->(1,1)
        fb = float(final_bias[0])
        lines.append(f'')
        lines.append(f'    min16float4 f_out = min16float4({_fmt(fb)},{_fmt(fb)},{_fmt(fb)},{_fmt(fb)});')
        
        for g in range(num_groups):
            lines.append(f'    f_in = f_out_{g};')
            for c in range(4):
                in_ch = g * 4 + c
                c_name = 'xyzw'[c]
                w_x = float(final_kernel[0, 0, 0, in_ch])
                w_y = float(final_kernel[0, 1, 0, in_ch])
                w_z = float(final_kernel[1, 0, 0, in_ch])
                w_w = float(final_kernel[1, 1, 0, in_ch])
                lines.append(f'    f_out += min16float4({_fmt(w_x)},{_fmt(w_y)},{_fmt(w_z)},{_fmt(w_w)}) * f_in.{c_name};')

        lines.append(f'    tex2Dstore(storageResult_nnaa1, (id.xy * 2) + uint2(0, 0), f_out.x);')
        lines.append(f'    tex2Dstore(storageResult_nnaa1, (id.xy * 2) + uint2(1, 0), f_out.y);')
        lines.append(f'    tex2Dstore(storageResult_nnaa1, (id.xy * 2) + uint2(0, 1), f_out.z);')
        lines.append(f'    tex2Dstore(storageResult_nnaa1, (id.xy * 2) + uint2(1, 1), f_out.w);')
    else:
        for g in range(num_groups):
            lines.append(f'    tex2Dstore({dst_storage}, id.xy * int2(1, 8) + int2(0, {g}), f_out_{g});')

    lines.append('}')
    return '\n'.join(lines)


def generate_technique():
    """Generate the ReShade technique block."""
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


# ============================================================================
# Public API
# ============================================================================

def convert_model(model_path, output_path, log_fn=None):
    """
    Convert a Keras NNAA model to a ReShade FX shader file.
    
    Args:
        model_path: Path to the .keras model file
        output_path: Path to write the generated .fx shader
        log_fn: Optional callback(message, tag) for progress logging.
                tag is one of: None, 'success', 'error', 'accent', 'warning'
    
    Returns:
        dict with 'bytes' and 'lines' counts on success.
    
    Raises:
        FileNotFoundError: if model_path doesn't exist
        ValueError: if model architecture doesn't match NNAA
    """
    import os
    
    def log(msg, tag=None):
        if log_fn:
            log_fn(msg, tag)

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    log("Loading model...\n", 'accent')
    layers = load_model_weights(model_path)

    log("Extracted layers:\n")
    for i, layer in enumerate(layers):
        shapes = [w.shape for w in layer['weights']]
        log(f"  [{i}] {layer['name']} ({layer['type']}): {shapes}\n")

    log("\nValidating architecture... ", None)
    w = validate_and_extract(layers)
    log("OK\n", 'success')

    log("\nGenerating shader...\n", 'accent')
    parts = []
    parts.append(generate_header())

    log("  Layer conv2d (8x8)... ")
    parts.append(generate_first_conv_layer(w['conv0_kernel'], w['conv0_bias'], w['prelu0_alpha']))
    log("OK\n", 'success')

    log("  Layer conv2d_1 (3x3)... ")
    parts.append('\n')
    parts.append(generate_mid_conv_layer(1, w['conv1_kernel'], w['conv1_bias'], w['prelu1_alpha']))
    log("OK\n", 'success')

    log("  Layer conv2d_2 (3x3)... ")
    parts.append('\n')
    parts.append(generate_mid_conv_layer(2, w['conv2_kernel'], w['conv2_bias'], w['prelu2_alpha']))
    log("OK\n", 'success')

    log("  Layer conv2d_3 + final (fused)... ")
    parts.append('\n')
    parts.append(generate_mid_conv_layer(3, w['conv3_kernel'], w['conv3_bias'], w['prelu3_alpha'],
                                         is_last_hidden=True,
                                         final_kernel=w['final_kernel'],
                                         final_bias=w['final_bias']))
    log("OK\n", 'success')

    parts.append(generate_technique())
    shader_code = '\n'.join(parts)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(shader_code)

    result = {
        'bytes': len(shader_code),
        'lines': shader_code.count('\n'),
    }
    log(f"\n✓ Shader saved to: {output_path}\n", 'success')
    log(f"  {result['bytes']:,} bytes, {result['lines']:,} lines\n", 'success')
    return result


# ============================================================================
# CLI entry point
# ============================================================================

def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'nnaa.keras'
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'out_nnaa.fx'

    def cli_log(msg, tag=None):
        print(msg, end='')

    try:
        result = convert_model(model_path, output_path, log_fn=cli_log)
        print(f"\nDone! Generated {result['bytes']} bytes, {result['lines']} lines.")
    except (FileNotFoundError, ValueError) as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
