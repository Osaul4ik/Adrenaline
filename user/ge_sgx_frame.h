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
	SGX renderer: one frame of GE work in the form the GXM renderer draws it.

	The walker thread fills a frame while it follows the lists (vertices
	decoded and scaled to 960x544, one record per draw with the GE state it
	needs); the draw thread renders it with GXM. Pure data, no Vita types.
*/

#ifndef __GE_SGX_FRAME_H__
#define __GE_SGX_FRAME_H__

#include <stdint.h>

// Output resolution: the PSP's 480x272 times this
#define GE_SGX_SCALE 2

// Vertices one draw may reference (16 bit indices)
#define GE_SGX_MAX_DRAW_VERTICES 65536

/*
	Homogeneous: x, y are 960x544 pixels and z the depth 0..1, all three
	multiplied by w (the clip w of 3D draws, 1 for 2D), so the GPU clips and
	interpolates perspective correct; the vertex shader only maps the pixels
	to clip space.
*/
typedef struct {
	float x, y, z, w;
	float u, v;     // normalized
	uint32_t color; // 0xAABBGGRR
	float fog;      // 1: no fog, 0: fog colour
} GeSgxVertex; // 32 bytes, the vertex stream of the GXM program

enum GeSgxDrawPrims {
	GE_SGX_DRAW_TRIANGLES,
	GE_SGX_DRAW_LINES,
	GE_SGX_DRAW_POINTS,
};

enum GeSgxDrawFlags {
	GE_SGX_DRAW_TEXTURED     = 1 << 0,
	GE_SGX_DRAW_BLEND        = 1 << 1,
	GE_SGX_DRAW_ALPHATEST    = 1 << 2,
	GE_SGX_DRAW_TCC          = 1 << 3, // texture alpha used (TEXFUNC RGBA)
	GE_SGX_DRAW_DOUBLE       = 1 << 4, // colour doubling
	GE_SGX_DRAW_CLEAR        = 1 << 5, // clear mode
	GE_SGX_DRAW_WRITE_COLOR  = 1 << 6, // clear mode: colour is written
	GE_SGX_DRAW_WRITE_ALPHA  = 1 << 7, // clear mode: alpha (stencil) is written
	GE_SGX_DRAW_WRITE_DEPTH  = 1 << 8, // depth is written
	GE_SGX_DRAW_FOG          = 1 << 9, // fog (the vertices carry the factor)
};

enum GeSgxTexModes {
	GE_SGX_TEX_CLAMP_U    = 1 << 0,
	GE_SGX_TEX_CLAMP_V    = 1 << 1,
	GE_SGX_TEX_LINEAR_MIN = 1 << 2,
	GE_SGX_TEX_LINEAR_MAG = 1 << 3,
};

typedef struct {
	uint8_t prim;   // GeSgxDrawPrims
	uint8_t tfunc;  // TEXFUNC 0..4 (modulate, decal, blend, replace, add)
	uint8_t atest_func; // GE alpha test function
	uint8_t atest_ref;
	uint16_t flags; // GeSgxDrawFlags
	uint8_t tex_mode;   // GeSgxTexModes
	uint8_t ztest;      // GE depth function (1: always)
	uint32_t blend;     // BLENDMODE: src 0-3, dst 4-7, equation 8-10
	uint32_t fix_a, fix_b; // fixed blend colours, 0x00BBGGRR
	uint32_t env;       // TEXENVCOLOR, 0x00BBGGRR
	uint32_t fog_color; // 0x00BBGGRR
	int16_t scissor[4]; // x0, y0, x1, y1 in 960x544 pixels, x1/y1 exclusive
	uint32_t base;      // first vertex: the indices count from it (GPU transform: byte offset in the raw vertex arena)
	uint32_t first;     // first index in the frame's index array
	uint32_t count;     // indices
	int32_t texture;    // handle from the texture callback, -1: none
	uint32_t fb;        // VRAM offset of the framebuffer drawn into

	// GPU transform: the vertices are the PSP's, the vertex program transforms them
	uint8_t gpu;
	uint8_t reserved2;
	uint16_t vkey;      // vertex program key (GE_SGX_VS_*)
	uint32_t vtype;     // GE vertex type: the attribute formats and offsets
	uint32_t ucommon;   // byte offsets of its uniform buffers in the uniform arena
	uint32_t uobject;
} GeSgxDraw;

typedef struct {
	GeSgxVertex *vertices;
	uint32_t max_vertices;
	uint32_t nvertices;

	uint16_t *indices;
	uint32_t max_indices;
	uint32_t nindices;

	/*
		GPU transform. vertices, indices and uniforms are built in cached
		memory and copied to the GPU's copies (*_gpu) when the frame is
		published; the raw arena is GPU memory the PSP vertex data is copied
		into directly. NULL raw: CPU transform only.
	*/
	uint8_t *uniforms;
	uint32_t max_uniforms;
	uint32_t nuniforms;  // bytes
	uint8_t *raw;
	uint32_t max_raw;
	uint32_t nraw;       // bytes
	const GeSgxVertex *vertices_gpu;
	const uint16_t *indices_gpu;
	const uint8_t *uniforms_gpu;

	GeSgxDraw *draws;
	uint32_t max_draws;
	uint32_t ndraws;

	uint32_t draws_3d;      // transformed (not through mode) primitives built
	uint32_t prims_gpu;     // of them with the GPU transform
	uint32_t skipped_other; // unsupported primitives, formats, full arrays
	uint32_t vertices_built; // vertices transformed
	uint32_t state_builds;   // times the draw state was built anew
	// Build profile (with a clock): state setup, texture lookups, vertices
	// (reading and transforming), indices and the rest of a primitive
	uint32_t us_state, us_texture, us_vertices, us_indices;
} GeSgxFrameData;

#endif
