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
	SGX renderer: GE display list walker.

	Follows the lists of the game in the order the PSP GE executes them
	(queue order, up to the stall address, through CALL/JUMP/RET and the
	signal jumps) and keeps the GE state that the renderer needs. Pure C
	without Vita dependencies: PSP memory is read through a callback, so the
	walker is tested on the host (tests/ge_sgx_test.c).
*/

#ifndef __GE_SGX_WALK_H__
#define __GE_SGX_WALK_H__

#include <stdint.h>

#define GE_SGX_MAX_LISTS 64
#define GE_SGX_MAX_STACK 32
#define GE_SGX_MAX_RECTS 256
// Unit in which PSP memory is fetched (ScePspemu maps the PSP RAM uncached:
// the Vita side copies blocks with burst reads instead of reading words)
#define GE_SGX_BLOCK_SIZE 256

// Longest walk of one list in one go, against lists that loop forever
#define GE_SGX_MAX_COMMANDS (1 << 18)

/*
	Returns a host pointer to the GE_SGX_BLOCK_SIZE bytes of PSP memory at
	page (a block address, aligned), or NULL if the address is not readable.
	The pointer stays valid until the next call.
*/
typedef const void *(* GeSgxPageFn)(void *user, uint32_t page);

typedef struct {
	uint32_t lists;       // lists finished
	uint32_t commands;    // commands walked
	uint32_t prims;       // PRIM commands
	uint32_t vertices;    // vertices drawn by them
	uint32_t curves;      // BEZIER/SPLINE commands
	uint32_t clears;      // clear mode sprites
	uint32_t signals;     // signal commands
	uint32_t bad;         // lists stopped at an unreadable address or runaway
	uint32_t dropped;     // lists dropped because the queue was full
} GeSgxStats;

typedef struct {
	int16_t x0, y0, x1, y1; // framebuffer pixels, x1/y1 exclusive
	uint32_t color;         // 0xAABBGGRR
	uint32_t fb;            // VRAM offset of the framebuffer
	uint32_t frame;         // flip count when it was drawn
} GeSgxRect;

typedef struct {
	int qid;
	uint32_t start;
	uint32_t pc;
	uint32_t stall; // 0: none
	int started;
	int depth;
	uint32_t stack[GE_SGX_MAX_STACK][2]; // return address, offset
} GeSgxList;

typedef struct {
	uint32_t size;      // bytes per vertex (all morph targets)
	uint32_t tc_off;    // offsets inside the vertex
	uint32_t color_off;
	uint32_t pos_off;
	uint32_t tc_fmt;    // 0: none, 1: u8, 2: u16, 3: float
	uint32_t color_fmt; // GE_VTYPE color field (0: none, 4..7)
	uint32_t pos_fmt;   // 1: s8, 2: s16, 3: float
	uint32_t index_size;
	uint32_t stride;    // bytes of one morph target
	uint32_t morphs;    // 1..8
	uint32_t wt_fmt;    // 0: none, 1: u8, 2: u16, 3: float (at offset 0)
	uint32_t weights;   // 1..8
	uint32_t nrm_fmt;   // 0: none, 1: s8, 2: s16, 3: float
	uint32_t nrm_off;
} GeSgxVertexLayout;

// Matrices of the GE as the lists upload them (PPSSPP layout: 4x3 column-major)
typedef struct {
	float bone[8 * 12];
	float world[12];
	float view[12];
	float proj[16];
	float tgen[12];
	uint8_t bone_n, world_n, view_n, proj_n, tgen_n;
} GeSgxMatrices;

#include "ge_sgx_frame.h"

/*
	Texture of a draw, as the GE state describes it. The texture callback
	(the Vita side texture cache) turns it into a handle for the draw
	records, or -1 if it cannot be used.
*/
typedef struct {
	uint32_t addr;        // level 0 in PSP memory
	uint32_t bufw;        // texels per row in memory
	uint16_t w, h;        // power of two sizes
	uint8_t format;       // TEXFORMAT 0..10
	uint8_t swizzle;
	uint8_t levels;       // mipmap levels (1: level 0 only), each half the size of the one before
	uint8_t reserved;
	uint32_t clut_format; // CLUTFORMAT: format 0-1, shift 2-6, mask 8-15, start 16-20
	const uint8_t *clut;  // the CLUT as LOADCLUT loaded it
	uint32_t clut_bytes;
	uint32_t clut_hash;
	uint32_t level_addr[8]; // levels 1.. (0: addr, bufw)
	uint16_t level_bufw[8];
} GeSgxTexDesc;

#define GE_SGX_MAX_LEVELS 8

typedef int32_t (* GeSgxTextureFn)(void *user, const GeSgxTexDesc *desc);

/*
	Optional: copies size bytes of PSP memory at addr into dst in one go
	(DMA), for the vertices of big primitives. Returns 0 if it cannot.
*/
typedef int (* GeSgxBulkFn)(uint32_t addr, void *dst, uint32_t size);

#define GE_SGX_CLUT_BYTES 1024
#define GE_SGX_GPU_BAD_MAX 8
#define GE_SGX_MAX_ALIASES 4

// Framebuffer copies (block transfers) of a frame: the shown buffer may be a copy
typedef struct {
	uint32_t src, dst; // VRAM offsets
} GeSgxAlias;

typedef struct {
	GeSgxPageFn page_fn;
	void *user;

	// Cached page
	uint32_t page;
	const uint8_t *page_ptr;

	// Queue in execution order
	GeSgxList lists[GE_SGX_MAX_LISTS];
	int count;

	// GE state (kept across lists, like the hardware): the last command
	// of each kind, and the addresses the GE keeps itself
	uint32_t regs[256];
	uint32_t offset;
	uint32_t vaddr;
	uint32_t iaddr;
	GeSgxMatrices mtx;

	// Vertex layout of regs[VERTEXTYPE], computed when first needed
	GeSgxVertexLayout layout;
	uint32_t layout_vtype;
	int layout_valid;

	GeSgxStats stats;

	// The last list stopped as bad, for the log
	struct {
		uint32_t count; // lists stopped as bad so far
		int qid;
		uint32_t start, pc, depth;
	} last_bad;

	GeSgxRect rects[GE_SGX_MAX_RECTS];
	int nrects;
	uint32_t flips;

	// Draw records for the GXM renderer (NULL: not built)
	GeSgxFrameData *frame;
	GeSgxTextureFn texture_fn;
	void *texture_user;
	GeSgxBulkFn bulk_fn;
	// Optional microsecond clock, for the build profile in the frame
	uint32_t (* clock_fn)(void);
	// A register, matrix or CLUT changed since the last primitive built
	int state_dirty;
	// 3D draws with the GPU transform where they can (frame->raw needed),
	// but not these vertex types (attribute bits of VERTEXTYPE)
	int gpu_transform;
	uint32_t gpu_bad[GE_SGX_GPU_BAD_MAX];
	int ngpu_bad;
	// PSP RAM the GPU can read directly (frame->direct): vertices there are not copied
	uint32_t direct_start, direct_size;
	GeSgxAlias aliases[GE_SGX_MAX_ALIASES];
	int naliases;

	// CLUT of the last LOADCLUT: where it is, read only when a draw needs it
	// (3D draws load CLUTs all the time)
	uint32_t clut_addr;
	uint32_t clut_load_bytes;
	int clut_read;
	uint8_t clut[GE_SGX_CLUT_BYTES];
	uint32_t clut_bytes;
	uint32_t clut_hash;
} GeSgxWalker;

// Copies size bytes of PSP memory, returns 0 if some are not readable
int geSgxRead(GeSgxWalker *w, uint32_t addr, void *dst, uint32_t size);

// Draw building (ge_sgx_build.c), called by the walker
void geSgxBuildPrim(GeSgxWalker *w, uint32_t type, uint32_t count, const GeSgxVertexLayout *l);
void geSgxBuildCurve(GeSgxWalker *w, int spline, uint32_t data, const GeSgxVertexLayout *l);
void geSgxBuildLoadClut(GeSgxWalker *w, uint32_t data);
void geSgxBuildTransfer(GeSgxWalker *w);
// Before the frame is published: copies the vertex data still pending
void geSgxBuildFinish(GeSgxWalker *w);


void geSgxVertexLayout(uint32_t vtype, GeSgxVertexLayout *layout);

// GE float: the 24 upper bits of an IEEE float
static inline float geSgxFloat24(uint32_t data) {
	union { uint32_t u; float f; } v;
	v.u = (data & 0x00FFFFFFu) << 8;
	return v.f;
}

// Last value of a GE command (24 bits)
static inline uint32_t geSgxReg(const GeSgxWalker *w, uint32_t cmd) {
	return w->regs[cmd & 0xFF] & 0x00FFFFFF;
}

void geSgxWalkerInit(GeSgxWalker *w, GeSgxPageFn page_fn, void *user);
void geSgxWalkerReset(GeSgxWalker *w);

// Ring events, in the order the PSP side reported them
void geSgxWalkerList(GeSgxWalker *w, int qid, uint32_t list, uint32_t stall, int head);
void geSgxWalkerStall(GeSgxWalker *w, int qid, uint32_t stall);
void geSgxWalkerDequeue(GeSgxWalker *w, int qid);

/*
	Display flip of top. Copies the clear rectangles drawn into that
	framebuffer since its last flip into out (up to max), returns their
	number. stats receives the statistics since the previous flip.
*/
int geSgxWalkerFlip(GeSgxWalker *w, uint32_t top, GeSgxRect *out, int max, GeSgxStats *stats);

// Walks the queued lists as far as they are available
void geSgxWalkerRun(GeSgxWalker *w);

#endif
