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

// Framebuffers a game draws into that are tracked (rendered and usable as textures)
#define GE_SGX_MAX_FBS 8

// Depth textures a frame may have (each with its palette: GeSgxFrameData.luts)
#define GE_SGX_MAX_LUTS 8

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

// Why a 3D primitive used the CPU transform
enum {
	GE_SGX_CPU_SPRITES, // 3D sprites
	GE_SGX_CPU_LAYOUT,  // morphing, 32 bit indices or a vertex type without program
	GE_SGX_CPU_FULL,    // the frame's raw or uniform arena is full
};

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
	GE_SGX_DRAW_FB_ALPHA     = 1 << 10, // the framebuffer has alpha (not 565): clears write it
	GE_SGX_DRAW_CONST_ALPHA  = 1 << 11, // every pixel's source alpha is calpha
};

/*
	Face culling (CULLFACEENABLE, CULL), as the PSP does it (PPSSPP's
	software renderer): triangles, strips and fans, not rectangles or clear
	mode, through mode included. CULL 1 draws the triangles clockwise on
	the screen, 0 the counterclockwise ones. Not done, The 3rd Birthday's
	shop drew a grey quad turned away over its background.
*/
enum GeSgxDrawCull {
	GE_SGX_CULL_NONE,
	GE_SGX_CULL_CW,  // clockwise ones not drawn (CULL 0)
	GE_SGX_CULL_CCW, // counterclockwise ones not drawn (CULL 1)
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

	// Framebuffer texture: VRAM offset + 1 of a framebuffer drawn here, sampled from
	// its own 960x544 rendering instead of the PSP's (0: a normal texture)
	uint32_t fbtex;
	/*
		Depth texture: VRAM offset + 1 of a framebuffer whose depth buffer is
		read (a CLUT16 texture at its depth's VRAM mirror, 0x04600000 + z
		address: Prince of Persia Revelations' fog), from its rendering's
		depth through the palette frame->luts[lut] (0: none)
	*/
	uint32_t depthtex;
	uint8_t lut;
	// Colour channels not written (MASKRGB): bit 0 red, 1 green, 2 blue
	uint8_t cmask;
	// Stencil: GE function (0xFF: no test), reference, compare and write masks,
	// operations (fail, depth fail, pass: 4 bits each)
	uint8_t sfunc, sref, smask, swmask;
	uint16_t sops;

	// GPU transform: the vertices are the PSP's, the vertex program transforms them
	uint8_t gpu;
	uint8_t direct;     // base is an offset in PSP RAM, which the GPU reads where it is
	uint8_t fbfmt;      // the framebuffer's pixel format (FRAMEBUFPIXFMT: 0 565, 1 5551, 2 4444, 3 8888)
	uint8_t texfmt;     // framebuffer texture (fbtex): the texture format it is read with (0: 5650, alpha 1)
	uint8_t cull;       // GeSgxDrawCull: triangles of one winding not drawn
	uint8_t calpha;     // GE_SGX_DRAW_CONST_ALPHA: the source alpha
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
	const uint8_t *direct; // PSP RAM as the GPU sees it (NULL: not mapped), see GeSgxWalker.direct_*
	const GeSgxVertex *vertices_gpu;
	const uint16_t *indices_gpu;
	const uint8_t *uniforms_gpu;

	GeSgxDraw *draws;
	uint32_t max_draws;
	uint32_t ndraws;

	// The framebuffers drawn (and how big they are in PSP pixels): their renderings are sized by this
	struct { uint32_t fb; uint16_t w, h; } fbs[GE_SGX_MAX_FBS];
	uint32_t nfbs;

	// Depth textures' palettes: the colour (0xAABBGGRR) for each depth's high byte (GeSgxDraw.depthtex)
	uint32_t luts[GE_SGX_MAX_LUTS][256];
	uint32_t nluts;

	uint32_t draws_3d;      // transformed (not through mode) primitives built
	uint32_t prims_gpu;     // of them with the GPU transform
	uint32_t cpu_reason[3]; // and why the others were not (GE_SGX_CPU_*)
	uint32_t skipped_other; // unsupported primitives, formats, full arrays
	uint32_t skipped_range; // primitives past the depth range (MINZ, MAXZ): no pixel drawn
	uint32_t cull_off;      // culled primitives crossing the near plane drawn without culling (too big to cull here)
	uint32_t cull_checked;  // culled primitives their format's bound could not put in front of the near plane: every vertex's clip position
	uint32_t cull_vertices; // their vertices
	uint32_t cull_cpu;      // primitives crossing the near plane culled here triangle by triangle (as the GE), the GPU's culling off
	uint32_t cull_dropped;  // their triangles not drawn
	uint32_t us_cull;       // looking at their vertices (sampled, as the other us_)
	uint32_t asked; // the frame of an asked trace (L + SELECT): its number, the renderer traces it too
	uint32_t vertices_built; // vertices transformed
	uint32_t state_builds;   // times the draw state was built anew
	// Build profile (with a clock): state setup, texture lookups, vertices
	// (reading and transforming), indices and the rest of a primitive
	// Build profile, in the walker's clock (see clock_fn): vertices includes vindex (reading the indices) and vuni (uniforms), texture clut
	uint32_t us_state, us_texture, us_vertices, us_indices, us_vindex, us_vuni, us_clut;
} GeSgxFrameData;

#endif
