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
	SGX renderer: GE display list walker. See ge_sgx_walk.h.

	Command semantics follow PPSSPP (GPU/GPUCommon.cpp), which matches the
	hardware for everything the walker needs.
*/

#include <string.h>

#include "ge_sgx_walk.h"

#define GE_CMD_NOP        0x00u
#define GE_CMD_VADDR      0x01u
#define GE_CMD_IADDR      0x02u
#define GE_CMD_PRIM       0x04u
#define GE_CMD_BEZIER     0x05u
#define GE_CMD_SPLINE     0x06u
#define GE_CMD_JUMP       0x08u
#define GE_CMD_BJUMP      0x09u
#define GE_CMD_CALL       0x0Au
#define GE_CMD_RET        0x0Bu
#define GE_CMD_END        0x0Cu
#define GE_CMD_SIGNAL     0x0Eu
#define GE_CMD_FINISH     0x0Fu
#define GE_CMD_BASE       0x10u
#define GE_CMD_VERTEXTYPE 0x12u
#define GE_CMD_OFFSETADDR 0x13u
#define GE_CMD_ORIGIN     0x14u
#define GE_CMD_FRAMEBUFPTR   0x9Cu
#define GE_CMD_FRAMEBUFWIDTH 0x9Du
#define GE_CMD_CLEARMODE  0xD3u

#define GE_PRIM_SPRITES 6u

#define GE_VTYPE_THROUGH (1u << 23)

// Signal behaviours that move the list
#define GE_SIGNAL_JUMP  0x10u
#define GE_SIGNAL_CALL  0x11u
#define GE_SIGNAL_RET   0x12u
#define GE_SIGNAL_RJUMP 0x13u
#define GE_SIGNAL_RCALL 0x14u
#define GE_SIGNAL_OJUMP 0x15u
#define GE_SIGNAL_OCALL 0x16u

#define NO_PAGE 0xFFFFFFFFu

enum {
	WALK_STALLED,
	WALK_FINISHED,
	WALK_BAD,
};

static inline uint32_t normalize(uint32_t addr) {
	return addr & 0x0FFFFFFFu;
}

static int readByte(GeSgxWalker *w, uint32_t addr, uint8_t *out) {
	addr = normalize(addr);
	uint32_t page = addr & ~(uint32_t)(GE_SGX_BLOCK_SIZE - 1);
	if (page != w->page) {
		w->page_ptr = (const uint8_t *)w->page_fn(w->user, page);
		w->page = w->page_ptr ? page : NO_PAGE;
		if (!w->page_ptr)
			return 0;
	}
	*out = w->page_ptr[addr & (GE_SGX_BLOCK_SIZE - 1)];
	return 1;
}

static int readWord(GeSgxWalker *w, uint32_t addr, uint32_t *out) {
	addr = normalize(addr);
	if (addr & 3)
		return 0;
	uint32_t page = addr & ~(uint32_t)(GE_SGX_BLOCK_SIZE - 1);
	if (page != w->page) {
		w->page_ptr = (const uint8_t *)w->page_fn(w->user, page);
		w->page = w->page_ptr ? page : NO_PAGE;
		if (!w->page_ptr)
			return 0;
	}
	// One aligned load (little endian like the PSP)
	memcpy(out, w->page_ptr + (addr & (GE_SGX_BLOCK_SIZE - 1)), 4);
	return 1;
}

// Little endian value of n (1, 2 or 4) bytes, may cross pages
static int readValue(GeSgxWalker *w, uint32_t addr, int n, uint32_t *out) {
	uint32_t v = 0;
	int i;
	for (i = 0; i < n; i++) {
		uint8_t b;
		if (!readByte(w, addr + i, &b))
			return 0;
		v |= (uint32_t)b << (8 * i);
	}
	*out = v;
	return 1;
}

static uint32_t relativeAddress(const GeSgxWalker *w, uint32_t data) {
	return normalize((((w->base & 0x000F0000u) << 8) | data) + w->offset);
}

static uint32_t alignUp(uint32_t v, uint32_t a) {
	return a ? (v + a - 1) & ~(a - 1) : v;
}

void geSgxVertexLayout(uint32_t vtype, GeSgxVertexLayout *layout) {
	static const uint8_t comp_size[4] = { 0, 1, 2, 4 };
	static const uint8_t color_size[8] = { 0, 0, 0, 0, 2, 2, 2, 4 };
	static const uint8_t index_size[4] = { 0, 1, 2, 4 };

	uint32_t tc = vtype & 3;
	uint32_t col = (vtype >> 2) & 7;
	uint32_t nrm = (vtype >> 5) & 3;
	uint32_t pos = (vtype >> 7) & 3;
	uint32_t wt = (vtype >> 9) & 3;
	uint32_t idx = (vtype >> 11) & 3;
	uint32_t weights = ((vtype >> 14) & 7) + 1;
	uint32_t morphs = ((vtype >> 18) & 7) + 1;

	uint32_t size = 0, biggest = 0;

	if (wt) {
		size = alignUp(size, comp_size[wt]);
		size += comp_size[wt] * weights;
		if (comp_size[wt] > biggest)
			biggest = comp_size[wt];
	}

	if (tc) {
		size = alignUp(size, comp_size[tc]);
		size += comp_size[tc] * 2;
		if (comp_size[tc] > biggest)
			biggest = comp_size[tc];
	}

	layout->color_off = 0;
	if (color_size[col]) {
		size = alignUp(size, color_size[col]);
		layout->color_off = size;
		size += color_size[col];
		if (color_size[col] > biggest)
			biggest = color_size[col];
	}

	if (nrm) {
		size = alignUp(size, comp_size[nrm]);
		size += comp_size[nrm] * 3;
		if (comp_size[nrm] > biggest)
			biggest = comp_size[nrm];
	}

	layout->pos_off = 0;
	if (pos) {
		size = alignUp(size, comp_size[pos]);
		layout->pos_off = size;
		size += comp_size[pos] * 3;
		if (comp_size[pos] > biggest)
			biggest = comp_size[pos];
	}

	size = alignUp(size, biggest);

	layout->size = size * morphs;
	layout->color_fmt = color_size[col] ? col : 0;
	layout->pos_fmt = pos;
	layout->index_size = index_size[idx];
}

static uint32_t expand(uint32_t v, int bits) {
	v &= (1u << bits) - 1;
	return (v << (8 - bits)) | (v >> (2 * bits - 8 > 0 ? 2 * bits - 8 : 0));
}

static uint32_t decodeColor(uint32_t fmt, uint32_t c) {
	uint32_t r, g, b;
	switch (fmt) {
		case 4: // 565
			r = expand(c, 5);
			g = expand(c >> 5, 6);
			b = expand(c >> 11, 5);
			break;
		case 5: // 5551
			r = expand(c, 5);
			g = expand(c >> 5, 5);
			b = expand(c >> 10, 5);
			break;
		case 6: // 4444
			r = (c & 0xF) * 0x11;
			g = ((c >> 4) & 0xF) * 0x11;
			b = ((c >> 8) & 0xF) * 0x11;
			break;
		case 7: // 8888
			r = c & 0xFF;
			g = (c >> 8) & 0xFF;
			b = (c >> 16) & 0xFF;
			break;
		default:
			r = g = b = 0;
			break;
	}
	// The alpha of a clear is the stencil value, the picture is opaque
	return 0xFF000000u | (b << 16) | (g << 8) | r;
}

static int vertexAddress(GeSgxWalker *w, const GeSgxVertexLayout *l, uint32_t i, uint32_t *addr) {
	uint32_t index = i;
	if (l->index_size) {
		if (!readValue(w, w->iaddr + i * l->index_size, l->index_size, &index))
			return 0;
	}
	*addr = w->vaddr + index * l->size;
	return 1;
}

static int16_t clampCoord(int32_t v) {
	if (v < 0)
		return 0;
	if (v > 4095)
		return 4095;
	return (int16_t)v;
}

static int readPosition(GeSgxWalker *w, const GeSgxVertexLayout *l, uint32_t vertex, int32_t *x, int32_t *y) {
	uint32_t a = vertex + l->pos_off;
	uint32_t vx, vy;
	switch (l->pos_fmt) {
		case 1:
			if (!readValue(w, a, 1, &vx) || !readValue(w, a + 1, 1, &vy))
				return 0;
			*x = (int8_t)vx;
			*y = (int8_t)vy;
			return 1;
		case 2:
			if (!readValue(w, a, 2, &vx) || !readValue(w, a + 2, 2, &vy))
				return 0;
			*x = (int16_t)vx;
			*y = (int16_t)vy;
			return 1;
		case 3: {
			float fx, fy;
			if (!readValue(w, a, 4, &vx) || !readValue(w, a + 4, 4, &vy))
				return 0;
			memcpy(&fx, &vx, 4);
			memcpy(&fy, &vy, 4);
			if (!(fx > -32768.0f && fx < 32768.0f && fy > -32768.0f && fy < 32768.0f))
				return 0;
			*x = (int32_t)fx;
			*y = (int32_t)fy;
			return 1;
		}
		default:
			return 0;
	}
}

static void addRect(GeSgxWalker *w, int32_t x0, int32_t y0, int32_t x1, int32_t y1, uint32_t color) {
	if (x1 < x0) {
		int32_t t = x0;
		x0 = x1;
		x1 = t;
	}
	if (y1 < y0) {
		int32_t t = y0;
		y0 = y1;
		y1 = t;
	}
	if (x0 == x1 || y0 == y1)
		return;

	if (w->nrects == GE_SGX_MAX_RECTS) {
		memmove(&w->rects[0], &w->rects[1], sizeof(GeSgxRect) * (GE_SGX_MAX_RECTS - 1));
		w->nrects--;
	}

	GeSgxRect *r = &w->rects[w->nrects++];
	r->x0 = clampCoord(x0);
	r->y0 = clampCoord(y0);
	r->x1 = clampCoord(x1);
	r->y1 = clampCoord(y1);
	r->color = color;
	r->fb = (((w->fbw & 0x00FF0000u) << 8) | (w->fbp & 0x00FFFFFFu)) & 0x001FFFFFu;
	r->frame = w->flips;
}

// Clear mode sprites in through mode: rectangles of one colour
static void clearSprites(GeSgxWalker *w, const GeSgxVertexLayout *l, uint32_t count) {
	if (!(w->vtype & GE_VTYPE_THROUGH) || !l->pos_fmt)
		return;

	uint32_t i;
	for (i = 0; i + 1 < count; i += 2) {
		uint32_t v0, v1;
		int32_t x0, y0, x1, y1;
		if (!vertexAddress(w, l, i, &v0) || !vertexAddress(w, l, i + 1, &v1))
			return;
		if (!readPosition(w, l, v0, &x0, &y0) || !readPosition(w, l, v1, &x1, &y1))
			return;

		// The second vertex gives the colour of a sprite
		uint32_t color = 0xFF000000u;
		if (l->color_fmt) {
			uint32_t c;
			if (!readValue(w, v1 + l->color_off, l->color_fmt == 7 ? 4 : 2, &c))
				return;
			color = decodeColor(l->color_fmt, c);
		}

		addRect(w, x0, y0, x1, y1, color);
		w->stats.clears++;
	}
}

static void prim(GeSgxWalker *w, uint32_t data) {
	uint32_t count = data & 0xFFFF;
	uint32_t type = (data >> 16) & 7;

	GeSgxVertexLayout l;
	geSgxVertexLayout(w->vtype, &l);

	w->stats.prims++;
	w->stats.vertices += count;

	if ((w->clearmode & 1) && type == GE_PRIM_SPRITES)
		clearSprites(w, &l, count);

	// The next PRIM continues after these vertices (or indices)
	if (l.index_size)
		w->iaddr = normalize(w->iaddr + count * l.index_size);
	else
		w->vaddr = normalize(w->vaddr + count * l.size);
}

static int push(GeSgxList *list, uint32_t ret, uint32_t offset) {
	if (list->depth >= GE_SGX_MAX_STACK)
		return 0;
	list->stack[list->depth][0] = ret;
	list->stack[list->depth][1] = offset;
	list->depth++;
	return 1;
}

static int pop(GeSgxWalker *w, GeSgxList *list) {
	if (list->depth == 0)
		return 0;
	list->depth--;
	list->pc = list->stack[list->depth][0];
	w->offset = list->stack[list->depth][1];
	return 1;
}

// SIGNAL at pc, followed by END. Returns 0 if the list cannot go on.
static int signal(GeSgxWalker *w, GeSgxList *list, uint32_t data) {
	uint32_t end;
	w->stats.signals++;

	if (!readWord(w, list->pc + 4, &end))
		return 0;
	if ((end >> 24) != GE_CMD_END) {
		// Not a signal block: just go on
		list->pc += 4;
		return 1;
	}

	uint32_t behaviour = (data >> 16) & 0xFF;
	uint32_t target = (((data & 0xFFFF) << 16) | (end & 0xFFFF)) & ~3u;
	uint32_t next = list->pc + 8;

	switch (behaviour) {
		case GE_SIGNAL_JUMP:
		case GE_SIGNAL_CALL:
			target = normalize(target);
			break;
		case GE_SIGNAL_RJUMP:
		case GE_SIGNAL_RCALL:
			target = normalize(target + list->pc);
			break;
		case GE_SIGNAL_OJUMP:
		case GE_SIGNAL_OCALL:
			target = normalize((((w->base & 0x000F0000u) << 8) | target) + w->offset);
			break;
		case GE_SIGNAL_RET:
			if (!pop(w, list))
				list->pc = next;
			return 1;
		default:
			// Handler signals and the rest: the list goes on after the END
			list->pc = next;
			return 1;
	}

	if (behaviour == GE_SIGNAL_CALL || behaviour == GE_SIGNAL_RCALL || behaviour == GE_SIGNAL_OCALL) {
		if (!push(list, next, w->offset))
			return 0;
	}

	list->pc = target;
	return 1;
}

static int walkList(GeSgxWalker *w, GeSgxList *list) {
	uint32_t n;
	list->started = 1;

	for (n = 0; n < GE_SGX_MAX_COMMANDS; n++) {
		if (list->stall && list->pc == list->stall)
			return WALK_STALLED;

		uint32_t op;
		if (!readWord(w, list->pc, &op))
			return WALK_BAD;

		w->stats.commands++;

		uint32_t cmd = op >> 24;
		uint32_t data = op & 0x00FFFFFF;

		switch (cmd) {
			case GE_CMD_VADDR:
				w->vaddr = relativeAddress(w, data);
				break;
			case GE_CMD_IADDR:
				w->iaddr = relativeAddress(w, data);
				break;
			case GE_CMD_PRIM:
				prim(w, data);
				break;
			case GE_CMD_BEZIER:
			case GE_CMD_SPLINE:
				w->stats.curves++;
				break;
			case GE_CMD_JUMP:
				list->pc = relativeAddress(w, data & 0x00FFFFFC);
				continue;
			case GE_CMD_CALL:
				if (!push(list, list->pc + 4, w->offset))
					return WALK_BAD;
				list->pc = relativeAddress(w, data & 0x00FFFFFC);
				continue;
			case GE_CMD_RET:
				if (pop(w, list))
					continue;
				break;
			case GE_CMD_END:
				return WALK_FINISHED;
			case GE_CMD_SIGNAL:
				if (!signal(w, list, data))
					return WALK_BAD;
				continue;
			case GE_CMD_BASE:
				w->base = data;
				break;
			case GE_CMD_VERTEXTYPE:
				w->vtype = data;
				break;
			case GE_CMD_OFFSETADDR:
				w->offset = data << 8;
				break;
			case GE_CMD_ORIGIN:
				w->offset = list->pc;
				break;
			case GE_CMD_FRAMEBUFPTR:
				w->fbp = data;
				break;
			case GE_CMD_FRAMEBUFWIDTH:
				w->fbw = data;
				break;
			case GE_CMD_CLEARMODE:
				w->clearmode = data;
				break;
			default:
				// BJUMP: the bounding box result is not known, the draws
				// after it are walked (as if the box was visible)
				break;
		}

		list->pc += 4;
	}

	return WALK_BAD;
}

static void removeList(GeSgxWalker *w, int i) {
	memmove(&w->lists[i], &w->lists[i + 1], sizeof(GeSgxList) * (w->count - i - 1));
	w->count--;
}

static int findList(GeSgxWalker *w, int qid) {
	int i;
	for (i = 0; i < w->count; i++) {
		if (w->lists[i].qid == qid)
			return i;
	}
	return -1;
}

void geSgxWalkerInit(GeSgxWalker *w, GeSgxPageFn page_fn, void *user) {
	memset(w, 0, sizeof(*w));
	w->page_fn = page_fn;
	w->user = user;
	w->page = NO_PAGE;
}

void geSgxWalkerReset(GeSgxWalker *w) {
	geSgxWalkerInit(w, w->page_fn, w->user);
}

void geSgxWalkerList(GeSgxWalker *w, int qid, uint32_t list, uint32_t stall, int head) {
	int i = findList(w, qid);
	if (i >= 0)
		removeList(w, i);

	if (w->count == GE_SGX_MAX_LISTS) {
		w->stats.dropped++;
		return;
	}

	// A list queued at the head runs after the one already running
	int at = w->count;
	if (head)
		at = (w->count > 0 && w->lists[0].started) ? 1 : 0;

	memmove(&w->lists[at + 1], &w->lists[at], sizeof(GeSgxList) * (w->count - at));
	w->count++;

	GeSgxList *l = &w->lists[at];
	memset(l, 0, sizeof(*l));
	l->qid = qid;
	l->pc = normalize(list) & ~3u;
	l->stall = stall ? (normalize(stall) & ~3u) : 0;
}

void geSgxWalkerStall(GeSgxWalker *w, int qid, uint32_t stall) {
	int i = findList(w, qid);
	if (i >= 0)
		w->lists[i].stall = stall ? (normalize(stall) & ~3u) : 0;
}

void geSgxWalkerDequeue(GeSgxWalker *w, int qid) {
	int i = findList(w, qid);
	if (i >= 0)
		removeList(w, i);
}

void geSgxWalkerRun(GeSgxWalker *w) {
	// Memory may have changed since the last run: fetch pages again
	w->page = NO_PAGE;
	w->page_ptr = NULL;

	while (w->count > 0) {
		int res = walkList(w, &w->lists[0]);
		if (res == WALK_STALLED)
			break;
		if (res == WALK_BAD)
			w->stats.bad++;
		else
			w->stats.lists++;
		removeList(w, 0);
	}
}

int geSgxWalkerFlip(GeSgxWalker *w, uint32_t top, GeSgxRect *out, int max, GeSgxStats *stats) {
	uint32_t addr = normalize(top);
	uint32_t fb = 0xFFFFFFFFu;
	if (addr >= 0x04000000u && addr < 0x04800000u)
		fb = addr & 0x001FFFFFu;

	int n = 0, kept = 0, i;
	for (i = 0; i < w->nrects; i++) {
		GeSgxRect *r = &w->rects[i];
		if (r->fb == fb) {
			if (n < max)
				out[n++] = *r;
		} else if (r->frame == w->flips) {
			// Drawn into another buffer this frame: may be shown next
			w->rects[kept++] = *r;
		}
	}
	w->nrects = kept;
	w->flips++;

	if (stats)
		*stats = w->stats;
	memset(&w->stats, 0, sizeof(w->stats));

	return n;
}
