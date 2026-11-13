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
	uint32_t pc;
	uint32_t stall; // 0: none
	int started;
	int depth;
	uint32_t stack[GE_SGX_MAX_STACK][2]; // return address, offset
} GeSgxList;

typedef struct {
	uint32_t size;      // bytes per vertex (all morph targets)
	uint32_t color_off; // offsets inside the vertex
	uint32_t pos_off;
	uint32_t color_fmt; // GE_VTYPE color field (0: none, 4..7)
	uint32_t pos_fmt;   // 1: s8, 2: s16, 3: float
	uint32_t index_size;
} GeSgxVertexLayout;

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

	// Vertex layout of regs[VERTEXTYPE], computed when first needed
	GeSgxVertexLayout layout;
	uint32_t layout_vtype;
	int layout_valid;

	GeSgxStats stats;

	GeSgxRect rects[GE_SGX_MAX_RECTS];
	int nrects;
	uint32_t flips;
} GeSgxWalker;


void geSgxVertexLayout(uint32_t vtype, GeSgxVertexLayout *layout);

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
