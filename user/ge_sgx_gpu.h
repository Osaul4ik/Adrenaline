/*
	Adrenaline
	Copyright (C) 2016-2018, TheFloW

	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

/*
	SGX renderer: what the draw builder and the vertex programs of the GPU
	transform share (pure C: also used by the host tests).
*/

#ifndef __GE_SGX_GPU_H__
#define __GE_SGX_GPU_H__

/*
	Vertex programs of the GPU transform (3D draws whose vertices the GPU
	reads as the PSP stores them): 0x2000 | the attributes the vertex has
	(GE_SGX_VS_*) | lighting. Everything else comes from two uniform buffers
	(see below), so one program serves every state of that vertex shape.
*/
#define GE_SGX_VS_BASE   0x2000u
#define GE_SGX_VS_TC     0x01u
#define GE_SGX_VS_COLOR  0x02u
#define GE_SGX_VS_NORMAL 0x04u
#define GE_SGX_VS_W4     0x08u // 1..4 weights
#define GE_SGX_VS_W8     0x10u // 5..8 weights
#define GE_SGX_VS_LIGHT  0x20u
#define GE_SGX_VS_COLOR16 0x40u // 16 bit colour (format in GE_SGX_UC_MATUPD.y), instead of GE_SGX_VS_COLOR

// Fragment program key bit: fixed blend colours applied by the program (ge_sgx_blend.h)
#define GE_SGX_FS_FIX (1u << 12)

/*
	Uniform buffer 0 (float4 entries): the state shared by many objects.
	Matrices are PSP layout (4x3 or 4x4 column-major), one float4 a column.
*/
enum {
	GE_SGX_UC_PROJ = 0,      // 4 columns
	GE_SGX_UC_VIEW = 4,      // 4 columns (xyz)
	GE_SGX_UC_VP = 8,        // x scale, x centre, y scale, y centre (960x544 pixels)
	GE_SGX_UC_VPZ = 9,       // z scale, z centre (0..1), fog end, fog slope
	GE_SGX_UC_FLAGS = 10,    // uv generation, projection source, fog on, normal sign
	GE_SGX_UC_MATAMB = 11,   // material ambient + alpha (the colour without vertex colours)
	GE_SGX_UC_EMISSIVE = 12, // rgb, specular exponent
	GE_SGX_UC_MATDIF = 13,   // rgb, ambient from the vertex colour
	GE_SGX_UC_MATSPC = 14,   // rgb, diffuse from the vertex colour
	GE_SGX_UC_AMBIENT = 15,  // global ambient rgba
	GE_SGX_UC_MATUPD = 16,   // specular from the vertex colour, 16 bit colour format (0 565, 1 5551, 2 4444), separate specular
	GE_SGX_UC_LIGHT = 17,    // 4 lights of GE_SGX_UC_LIGHT_SIZE
	GE_SGX_UC_ENVLS = 45,    // 2: environment map light directions
	GE_SGX_UC_TGEN = 47,     // 4 columns
	GE_SGX_UC_COUNT = 51,
};
// One light: position + type (-1 off, 0 directional, 1 point, 2 spot),
// direction + computation, attenuation + spot exponent, spot cutoff,
// ambient, diffuse, specular colour
#define GE_SGX_UC_LIGHT_SIZE 7

// Uniform buffer 1: the object
enum {
	GE_SGX_UO_WORLD = 0,     // 4 columns
	GE_SGX_UO_SCALE = 4,     // position, normal, texture coordinate, weight scale
	GE_SGX_UO_UV = 5,        // u scale, v scale, u offset, v offset
	GE_SGX_UO_WMASK = 6,     // 2: 1 for the weights the vertex has
	GE_SGX_UO_BONES = 8,     // 8 x 4 columns
	GE_SGX_UO_COUNT = 40,
};

#endif
