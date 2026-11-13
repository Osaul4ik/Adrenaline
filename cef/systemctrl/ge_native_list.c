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
	GE display list patcher for native resolution rendering.
	Pure list/memory logic, no firmware calls: it is also compiled on the host
	for unit tests (GE_NATIVE_HOST_TEST).
*/

#include <string.h>

#include "ge_native.h"

#ifdef GE_NATIVE_TRACE
/* Frame trace (debug builds): ge_native.c collects the lines */
int geNativeTraceOn = 0;
#define TRACE(...) do { if (geNativeTraceOn) geNativeTrace(__VA_ARGS__); } while (0)
#else
#define TRACE(...) do { } while (0)
#endif

#define GE_MAX_COMMANDS   (1 << 20)
#define GE_STACK_SIZE     64
#define GE_MAX_FRAMEBUFS  16
#define GE_ORIG_MAP_SHIFT 14 /* log2(GE_NATIVE_ORIG_MAP_ENTRIES) */
#define GE_ORIG_MAP_MASK  (GE_NATIVE_ORIG_MAP_ENTRIES - 1)
#define GE_ORIG_MAP_PROBE 8
#define GE_VTX_MAP_SHIFT  13 /* log2(GE_NATIVE_VTX_MAP_ENTRIES) */
#define GE_VTX_MAP_MASK   (GE_NATIVE_VTX_MAP_ENTRIES - 1)
#define GE_VTX_MAP_PROBE  8

#define ALIGN_UP(x, a) (((x) + ((a) - 1)) & ~((a) - 1))

typedef struct {
	u32 list;
	u32 base;
	u32 offset;
} GeStackEntry;

typedef struct {
	u32 ge_cmd[0x100]; /* last original data per command */

	u32 texbufptr[8];
	u32 texbufwidth[8];
	u32 framebufptr;
	u32 framebufwidth;
	u32 framebufwidth_addr;
	u32 framebuf;

	u32 base;
	u32 offset;

	u32 index_addr;
	u32 vertex_addr;
	u32 vertex_type;

	u32 ignore_framebuf;
	u32 ignore_texture;

	u32 has_draws;

	GeStackEntry stack[GE_STACK_SIZE];
	u32 curr_stack;

	u32 framebuf_addr[GE_MAX_FRAMEBUFS];
	u32 framebuf_count;

	u32 tex0; /* address of texture level 0, for the trace */
} GeState;

typedef struct {
	u32 addr;
	u32 orig;
	u32 patched;
	u32 seen; /* frame the entry was last used, for eviction */
} GeOrigEntry;

typedef struct {
	u32 addr;
	u32 tag;   /* vertex type */
	u32 count; /* vertices scaled from addr */
	u32 sum;   /* fingerprint of their scaled positions */
	u32 seen;
} GeVtxEntry;

typedef union {
	float f;
	u32 i;
} GeFloat;

static GeState state;

/* Draws into the upscaled framebuffer since the last display flip.
   Not part of GeState: it must survive list resets. */
static u32 frame_draws;

/* Display flip counter, ages the original-value map */
static u32 frame_epoch;

static const u8 tcsize[4] = { 0, 2, 4, 8 }, tcalign[4] = { 0, 1, 2, 4 };
static const u8 colsize[8] = { 0, 0, 0, 0, 2, 2, 2, 4 }, colalign[8] = { 0, 0, 0, 0, 2, 2, 2, 4 };
static const u8 nrmsize[4] = { 0, 3, 6, 12 }, nrmalign[4] = { 0, 1, 2, 4 };
static const u8 possize[4] = { 3, 3, 6, 12 }, posalign[4] = { 1, 1, 2, 4 };
static const u8 wtsize[4] = { 0, 1, 2, 4 }, wtalign[4] = { 0, 1, 2, 4 };

/* ------------------------------------------------------------------------ */
/* Helpers                                                                    */
/* ------------------------------------------------------------------------ */

static int geAddrOk(u32 addr, u32 size) {
	addr &= 0x0FFFFFFF;

	/* Main RAM (incl. Adrenaline's extended memory) */
	if (addr >= 0x08000000 && addr < 0x0C000000 && size <= 0x0C000000 - addr)
		return 1;

	/* VRAM */
	if (addr >= 0x04000000 && addr < 0x04200000 && size <= 0x04200000 - addr)
		return 1;

	return 0;
}

static inline u32 geRead(u32 addr) {
	return *(u32 *)GE_PTR(addr);
}

static inline void geWrite(u32 addr, u32 op) {
	u32 *p = (u32 *)GE_PTR(addr);
	if (*p != op)
		*p = op;
}

static inline float geF24(u32 data) {
	GeFloat t;
	t.i = data << 8;
	return t.f;
}

static inline u32 geToF24(float f) {
	GeFloat t;
	t.f = f;
	return t.i >> 8;
}

static inline GeOrigEntry *origMap() {
	return (GeOrigEntry *)GE_PTR(GE_NATIVE_ORIG_MAP);
}

static inline u32 origHash(u32 addr) {
	return ((addr >> 2) * 2654435761u) >> (32 - GE_ORIG_MAP_SHIFT);
}

/*
	Returns the value the game originally wrote at addr. If the word still
	holds what we wrote there, it was rewritten by us and the stored original
	is returned. Otherwise the game wrote something new and cur is original.
*/
static u32 origGet(u32 addr, u32 cur) {
	GeOrigEntry *map = origMap();
	u32 h = origHash(addr);

	int i;
	for (i = 0; i < GE_ORIG_MAP_PROBE; i++) {
		GeOrigEntry *e = &map[(h + i) & GE_ORIG_MAP_MASK];
		if (e->addr == addr) {
			if (e->patched != cur)
				return cur;
			e->seen = frame_epoch;
			return e->orig;
		}
		if (e->addr == 0)
			break;
	}

	return cur;
}

static void origPut(u32 addr, u32 orig, u32 patched) {
	GeOrigEntry *map = origMap();
	u32 h = origHash(addr);
	GeOrigEntry *e = NULL;
	GeOrigEntry *oldest = NULL;

	int i;
	for (i = 0; i < GE_ORIG_MAP_PROBE; i++) {
		GeOrigEntry *c = &map[(h + i) & GE_ORIG_MAP_MASK];
		if (c->addr == addr || c->addr == 0) {
			e = c;
			break;
		}
		if (!oldest || (frame_epoch - c->seen) > (frame_epoch - oldest->seen))
			oldest = c;
	}

	/* Probe window full: evict the entry unused for the longest time. Lists
	   that run every frame keep their entries fresh, so in practice only
	   commands of lists the game no longer uses are dropped. */
	if (!e)
		e = oldest;

	e->addr = addr;
	e->orig = orig;
	e->patched = patched;
	e->seen = frame_epoch;
}

/* Rewrite a command and remember the original */
static void geSet(u32 addr, u32 orig, u32 op) {
	geWrite(addr, op);
	if (op != orig)
		origPut(addr, orig, op);
}

static int isTrackedCommand(u32 cmd) {
	switch (cmd) {
		case GE_CMD_VIEWPORTXSCALE:
		case GE_CMD_VIEWPORTYSCALE:
		case GE_CMD_VIEWPORTXCENTER:
		case GE_CMD_VIEWPORTYCENTER:
		case GE_CMD_OFFSETX:
		case GE_CMD_OFFSETY:
		case GE_CMD_REGION1:
		case GE_CMD_REGION2:
		case GE_CMD_SCISSOR1:
		case GE_CMD_SCISSOR2:
			return 1;
	}

	return 0;
}

static u32 clampU(int v, int max) {
	if (v < 0)
		return 0;
	if (v > max)
		return max;
	return v;
}

/* (y << 10) | x, start corner: scale by 2 */
static u32 scaleCornerStart(u32 data) {
	u32 x = clampU((data & 0x3FF) * 2, GE_NATIVE_WIDTH - 1);
	u32 y = clampU(((data >> 10) & 0x3FF) * 2, GE_NATIVE_HEIGHT - 1);
	return (y << 10) | x;
}

/* (y << 10) | x, inclusive end corner: last pixel of a 2x2 block */
static u32 scaleCornerEnd(u32 data) {
	u32 x = clampU((data & 0x3FF) * 2 + 1, GE_NATIVE_WIDTH - 1);
	u32 y = clampU(((data >> 10) & 0x3FF) * 2 + 1, GE_NATIVE_HEIGHT - 1);
	return (y << 10) | x;
}

/* ------------------------------------------------------------------------ */
/* Vertex helpers                                                             */
/* ------------------------------------------------------------------------ */

static void getVertexInfo(u32 op, u8 *vertex_size, u8 *pos_off) {
	int tc = (op & GE_VTYPE_TC_MASK) >> GE_VTYPE_TC_SHIFT;
	int col = (op & GE_VTYPE_COL_MASK) >> GE_VTYPE_COL_SHIFT;
	int nrm = (op & GE_VTYPE_NRM_MASK) >> GE_VTYPE_NRM_SHIFT;
	int pos = (op & GE_VTYPE_POS_MASK) >> GE_VTYPE_POS_SHIFT;
	int weight = (op & GE_VTYPE_WEIGHT_MASK) >> GE_VTYPE_WEIGHT_SHIFT;
	int weightCount = ((op & GE_VTYPE_WEIGHTCOUNT_MASK) >> GE_VTYPE_WEIGHTCOUNT_SHIFT) + 1;
	int morphCount = ((op & GE_VTYPE_MORPHCOUNT_MASK) >> GE_VTYPE_MORPHCOUNT_SHIFT) + 1;

	u8 biggest = 0;
	u8 size = 0;
	u8 aligned_size = 0;
	u8 posoff = 0;

	if (weight) {
		size += wtsize[weight] * weightCount;
		if (wtalign[weight] > biggest)
			biggest = wtalign[weight];
	}

	if (tc) {
		aligned_size = ALIGN_UP(size, tcalign[tc]);
		size = aligned_size;
		size += tcsize[tc];
		if (tcalign[tc] > biggest)
			biggest = tcalign[tc];
	}

	if (col) {
		aligned_size = ALIGN_UP(size, colalign[col]);
		size = aligned_size;
		size += colsize[col];
		if (colalign[col] > biggest)
			biggest = colalign[col];
	}

	if (nrm) {
		aligned_size = ALIGN_UP(size, nrmalign[nrm]);
		size = aligned_size;
		size += nrmsize[nrm];
		if (nrmalign[nrm] > biggest)
			biggest = nrmalign[nrm];
	}

	aligned_size = ALIGN_UP(size, posalign[pos]);
	size = aligned_size;
	posoff = size;
	size += possize[pos];
	if (posalign[pos] > biggest)
		biggest = posalign[pos];

	aligned_size = ALIGN_UP(size, biggest);
	size = aligned_size;
	size *= morphCount;

	*vertex_size = size;
	*pos_off = posoff;
}

static int getIndexBounds(u32 inds, int count, u32 vertType, u16 *lower, u16 *upper) {
	int lowerBound = 0x7FFFFFFF;
	int upperBound = 0;
	u32 idx = vertType & GE_VTYPE_IDX_MASK;
	int i;

	if (idx == GE_VTYPE_IDX_NONE) {
		*lower = 0;
		*upper = count - 1;
		return 0;
	}

	int shift = (idx >> GE_VTYPE_IDX_SHIFT) - 1;
	if (!geAddrOk(inds, count << shift))
		return -1;

	if (idx == GE_VTYPE_IDX_8BIT) {
		const u8 *ind8 = (const u8 *)GE_PTR(inds);
		for (i = 0; i < count; i++) {
			if (ind8[i] > upperBound)
				upperBound = ind8[i];
			if (ind8[i] < lowerBound)
				lowerBound = ind8[i];
		}
	} else if (idx == GE_VTYPE_IDX_16BIT) {
		const u16 *ind16 = (const u16 *)GE_PTR(inds);
		for (i = 0; i < count; i++) {
			if (ind16[i] > upperBound)
				upperBound = ind16[i];
			if (ind16[i] < lowerBound)
				lowerBound = ind16[i];
		}
	} else {
		const u32 *ind32 = (const u32 *)GE_PTR(inds);
		for (i = 0; i < count; i++) {
			u16 value = (u16)ind32[i];
			if (value > upperBound)
				upperBound = value;
			if (value < lowerBound)
				lowerBound = value;
		}
	}

	if (count == 0)
		lowerBound = 0;

	*lower = (u16)lowerBound;
	*upper = (u16)upperBound;
	return 0;
}

static void advanceVerts(int count, int vertex_size) {
	if ((state.vertex_type & GE_VTYPE_IDX_MASK) != GE_VTYPE_IDX_NONE) {
		int index_shift = ((state.vertex_type & GE_VTYPE_IDX_MASK) >> GE_VTYPE_IDX_SHIFT) - 1;
		state.index_addr += count << index_shift;
	} else {
		state.vertex_addr += count * vertex_size;
	}
}

/*
	Through-mode vertices are already in screen space, so their positions are
	doubled. A buffer may be drawn several times per frame or be part of a
	static list, so for every scaled range a fingerprint of the scaled
	positions is kept: if a range still has that fingerprint it was already
	scaled and is left alone, otherwise the game wrote new vertices.
	A position that scaling leaves unchanged (0) cannot cause a wrong skip.
*/
static inline u32 fnvStep(u32 h, u32 v) {
	return (h ^ v) * 16777619u;
}

static inline u32 vtxHash(u32 addr) {
	return ((addr >> 1) * 2654435761u) >> (32 - GE_VTX_MAP_SHIFT);
}

static GeVtxEntry *vtxFind(u32 addr, u32 tag, int create) {
	GeVtxEntry *map = (GeVtxEntry *)GE_PTR(GE_NATIVE_VTX_MAP);
	u32 h = vtxHash(addr);
	GeVtxEntry *oldest = NULL;

	int i;
	for (i = 0; i < GE_VTX_MAP_PROBE; i++) {
		GeVtxEntry *e = &map[(h + i) & GE_VTX_MAP_MASK];
		if ((e->addr == addr && e->tag == tag) || e->addr == 0) {
			if (e->addr == 0 && !create)
				return NULL;
			return e;
		}
		if (!oldest || (frame_epoch - e->seen) > (frame_epoch - oldest->seen))
			oldest = e;
	}

	return create ? oldest : NULL;
}

/* Fingerprint of the x/y positions of n vertices */
static u32 posSum(u32 addr, u32 n, u32 stride, int pos_size) {
	u32 sum = 2166136261u;
	u32 i;
	for (i = 0; i < n; i++, addr += stride) {
		if (pos_size == 2) {
			u32 x = *(u16 *)GE_PTR(addr);
			u32 y = *(u16 *)GE_PTR(addr + 2);
			sum = fnvStep(sum, x | (y << 16));
		} else {
			sum = fnvStep(sum, *(u32 *)GE_PTR(addr));
			sum = fnvStep(sum, *(u32 *)GE_PTR(addr + 4));
		}
	}
	return sum;
}

static void scalePositions(u32 addr, u32 n, u32 stride, int pos_size) {
	u32 i;
	for (i = 0; i < n; i++, addr += stride) {
		int j;
		for (j = 0; j < 2; j++) {
			if (pos_size == 2) {
				s16 *p = (s16 *)GE_PTR(addr + j * 2);
				s16 val = *p;
				if (val > -2048 && val < 2048)
					*p = val * 2;
			} else {
				GeFloat t;
				u32 *p = (u32 *)GE_PTR(addr + j * 4);
				t.i = *p;
				if (t.f > -2048 && t.f < 2048) {
					t.f *= 2;
					*p = t.i;
				}
			}
		}
	}
}

static void patchThroughVertices(u16 count) {
	u8 vertex_size = 0, pos_off = 0;
	getVertexInfo(state.vertex_type, &vertex_size, &pos_off);

	if (vertex_size == 0)
		return;

	int pos = (state.vertex_type & GE_VTYPE_POS_MASK) >> GE_VTYPE_POS_SHIFT;
	int pos_size = possize[pos] / 3;

	/* 8 bit positions are not valid in through mode */
	if (pos_size != 2 && pos_size != 4)
		return;

	u16 lower = 0;
	u16 upper = count;
	if ((state.vertex_type & GE_VTYPE_IDX_MASK) != GE_VTYPE_IDX_NONE) {
		if (getIndexBounds(state.index_addr, count, state.vertex_type, &lower, &upper) < 0)
			return;
		upper += 1;
	}

	if (upper <= lower)
		return;

	u32 n = upper - lower;
	u32 first = state.vertex_addr + lower * vertex_size;
	if (!geAddrOk(first, n * vertex_size))
		return;

	u32 tag = state.vertex_type & 0xFFFFFF;
	u32 done = 0;

	/* A range starting here was scaled before: if its fingerprint still
	   matches, that prefix is done (draws of the same buffer may use more
	   or fewer vertices). */
	GeVtxEntry *e = vtxFind(first, tag, 0);
	if (e && e->count && geAddrOk(first, e->count * vertex_size)) {
		if (posSum(first + pos_off, e->count, vertex_size, pos_size) == e->sum) {
			if (e->count >= n) {
				e->seen = frame_epoch;
				return;
			}
			done = e->count;
		}
	}

	scalePositions(first + pos_off + done * vertex_size, n - done, vertex_size, pos_size);

	if (!e)
		e = vtxFind(first, tag, 1);

	/* Remember the range scaled by this draw */
	e->addr = first;
	e->tag = tag;
	e->count = n;
	e->sum = posSum(first + pos_off, n, vertex_size, pos_size);
	e->seen = frame_epoch;
}

/* ------------------------------------------------------------------------ */
/* Framebuffer tracking                                                       */
/* ------------------------------------------------------------------------ */

static int findFramebuf(u32 framebuf) {
	u32 i;
	for (i = 0; i < state.framebuf_count; i++) {
		/* Some textures point into the framebuffer */
		if (framebuf >= state.framebuf_addr[i] && framebuf < (state.framebuf_addr[i] + 0x400))
			return i;
	}

	return -1;
}

static int insertFramebuf(u32 framebuf) {
	int res = findFramebuf(framebuf);
	if (res >= 0)
		return res;

	if (state.framebuf_count < GE_MAX_FRAMEBUFS) {
		state.framebuf_addr[state.framebuf_count] = framebuf;
		return state.framebuf_count++;
	}

	return -1;
}

static int push(u32 list, u32 base, u32 offset) {
	if (state.curr_stack < GE_STACK_SIZE) {
		GeStackEntry *e = &state.stack[state.curr_stack++];
		e->list = list;
		e->base = base;
		e->offset = offset;
		return 0;
	}

	return -1;
}

static GeStackEntry *pop() {
	if (state.curr_stack > 0)
		return &state.stack[--state.curr_stack];
	return NULL;
}

/* A draw is skipped when it would render into a framebuffer we do not
   upscale, or would sample a framebuffer that was not rendered at its
   original address. */
static int drawIgnored() {
	return state.ignore_framebuf || (state.ignore_texture && state.ge_cmd[GE_CMD_TEXTUREMAPENABLE]);
}

/* Disable or (re-)enable a draw command */
static int setDrawEnabled(u32 addr, u32 cur, u32 op) {
	if (drawIgnored()) {
		geSet(addr, op, (GE_CMD_NOP << 24) | GE_NATIVE_NOP_MARKER);
		return 0;
	}

	if (cur != op)
		geWrite(addr, op); /* restore a draw we disabled on an earlier pass */

	frame_draws++;
	return 1;
}

static void traceDraw(u32 addr, u32 op, int enabled) {
	TRACE("D %08x op %08x vt %06x v %08x clr %d tex %d %08x blend %d fb %08x %s\n", addr, op, state.vertex_type, state.vertex_addr,
	      state.ge_cmd[GE_CMD_CLEARMODE] & 1, state.ge_cmd[GE_CMD_TEXTUREMAPENABLE] & 1, state.tex0,
	      state.ge_cmd[GE_CMD_ALPHABLENDENABLE] & 1, state.framebuf,
	      enabled ? "on" : (state.ignore_framebuf ? "OFF fb" : "OFF tex"));
	(void)addr; (void)op; (void)enabled;
}

/* ------------------------------------------------------------------------ */
/* Public                                                                     */
/* ------------------------------------------------------------------------ */

void geNativeListInit(void) {
	memset(origMap(), 0, GE_NATIVE_ORIG_MAP_ENTRIES * sizeof(GeOrigEntry));
	memset(GE_PTR(GE_NATIVE_VTX_MAP), 0, GE_NATIVE_VTX_MAP_ENTRIES * sizeof(GeVtxEntry));
	memset(&state, 0, sizeof(GeState));
	frame_draws = 0;
	frame_epoch = 0;
}

void geNativeListReset(void) {
	memset(&state, 0, sizeof(GeState));
}

int geNativeListHasDraws(void) {
	return state.has_draws;
}

u32 geNativeListTakeFrameDraws(void) {
	u32 n = frame_draws;
	frame_draws = 0;
	frame_epoch++;
	return n;
}

void geNativeListPatch(u32 list, u32 stall) {
	u32 n;

	list &= 0x0FFFFFFF;
	stall &= 0x0FFFFFFF;

	TRACE("L %08x stall %08x\n", list, stall);

	for (n = 0; list && (!stall || list != stall) && n < GE_MAX_COMMANDS; list += 4, n++) {
		if (!geAddrOk(list, 4))
			return;

		u32 cur = geRead(list);
		u32 op = cur;

		/* Recover the game's value of a command we rewrote on an earlier pass */
		if (isTrackedCommand(cur >> 24) || cur == ((GE_CMD_NOP << 24) | GE_NATIVE_NOP_MARKER))
			op = origGet(list, cur);

		u32 cmd = op >> 24;
		u32 data = op & 0xFFFFFF;
		u32 target;
		GeStackEntry *entry;

		state.ge_cmd[cmd] = data;

		switch (cmd) {
			/* Control flow */

			case GE_CMD_BASE:
				state.base = (data << 8) & 0x0F000000;
				break;

			case GE_CMD_OFFSETADDR:
				state.offset = data << 8;
				break;

			case GE_CMD_ORIGIN:
				state.offset = list;
				break;

			case GE_CMD_CALL:
				target = ((state.base | data) + state.offset) & 0x0FFFFFFC;
				if (!geAddrOk(target, 4))
					return;

				/* Skip bone matrix upload sub-lists */
				if (geAddrOk(target, 13 * 4) &&
				    (geRead(target) >> 24) == GE_CMD_BONEMATRIXDATA &&
				    (geRead(target + 11 * 4) >> 24) == GE_CMD_BONEMATRIXDATA &&
				    (geRead(target + 12 * 4) >> 24) == GE_CMD_RET) {
					break;
				}

				TRACE("C call %08x -> %08x\n", list, target);
				if (push(list, state.base, state.offset) == 0)
					list = target - 4;
				break;

			case GE_CMD_BJUMP: /* conditional, always followed */
			case GE_CMD_JUMP:
				target = ((state.base | data) + state.offset) & 0x0FFFFFFC;
				if (!geAddrOk(target, 4))
					return;
				TRACE("C %s %08x -> %08x\n", cmd == GE_CMD_JUMP ? "jump" : "bjump", list, target);
				list = target - 4;
				break;

			case GE_CMD_RET:
				/* Ignore returns when the stack is empty */
				entry = pop();
				if (entry) {
					list = entry->list;
					state.offset = entry->offset;
				}
				break;

			case GE_CMD_END:
			{
				if (!geAddrOk(list - 4, 4))
					return;

				u32 prev = geRead(list - 4);
				switch (prev >> 24) {
					case GE_CMD_SIGNAL:
					{
						u8 behaviour = (prev >> 16) & 0xFF;
						u16 signal = prev & 0xFFFF;
						u16 enddata = data & 0xFFFF;

						switch (behaviour) {
							case GE_SIGNAL_SYNC:
								list += 2 * 4;
								break;

							case GE_SIGNAL_JUMP:
								target = (((signal << 16) | enddata) & 0x0FFFFFFC);
								if (!geAddrOk(target, 4))
									return;
								list = target - 4;
								break;

							case GE_SIGNAL_CALL:
								target = (((signal << 16) | enddata) & 0x0FFFFFFC);
								if (!geAddrOk(target, 4))
									return;
								if (push(list, state.base, state.offset) == 0)
									list = target - 4;
								break;

							case GE_SIGNAL_RET:
								entry = pop();
								if (entry) {
									list = entry->list;
									state.base = entry->base;
									state.offset = entry->offset;
								}
								break;

							default:
								break;
						}
						break;
					}

					case GE_CMD_FINISH:
						TRACE("C finish %08x\n", list);
						return;

					default:
						break;
				}
				break;
			}

			/* Vertices */

			case GE_CMD_IADDR:
				state.index_addr = ((state.base | data) + state.offset) & 0x0FFFFFFF;
				break;

			case GE_CMD_VADDR:
				state.vertex_addr = ((state.base | data) + state.offset) & 0x0FFFFFFF;
				break;

			case GE_CMD_VERTEXTYPE:
				state.vertex_type = data;
				break;

			case GE_CMD_BEZIER:
			case GE_CMD_SPLINE:
			{
				state.has_draws = 1;

				int enabled = setDrawEnabled(list, cur, op);
				traceDraw(list, op, enabled);
				if (!enabled)
					break;

				if ((state.vertex_type & GE_VTYPE_THROUGH) == GE_VTYPE_THROUGH) {
					u8 vertex_size = 0, pos_off = 0;
					getVertexInfo(state.vertex_type, &vertex_size, &pos_off);
					advanceVerts((data & 0xFF) * ((data >> 8) & 0xFF), vertex_size);
				}
				break;
			}

			case GE_CMD_BOUNDINGBOX:
			{
				if (drawIgnored())
					break;

				if ((state.vertex_type & GE_VTYPE_THROUGH) == GE_VTYPE_THROUGH) {
					u8 vertex_size = 0, pos_off = 0;
					getVertexInfo(state.vertex_type, &vertex_size, &pos_off);
					advanceVerts(data, vertex_size);
				}
				break;
			}

			case GE_CMD_PRIM:
			{
				state.has_draws = 1;

				int enabled = setDrawEnabled(list, cur, op);
				traceDraw(list, op, enabled);
				if (!enabled)
					break;

				if ((state.vertex_type & GE_VTYPE_THROUGH) == GE_VTYPE_THROUGH) {
					u16 count = data & 0xFFFF;
					u8 vertex_size = 0, pos_off = 0;
					getVertexInfo(state.vertex_type, &vertex_size, &pos_off);
					patchThroughVertices(count);
					advanceVerts(count, vertex_size);
				}
				break;
			}

			/* Render target */

			case GE_CMD_FRAMEBUFPIXFORMAT:
				geWrite(list, (cmd << 24) | GE_FORMAT_565);
				break;

			case GE_CMD_FRAMEBUFPTR:
			case GE_CMD_FRAMEBUFWIDTH:
			{
				if (cmd == GE_CMD_FRAMEBUFPTR) {
					geWrite(list, (cmd << 24) | (GE_NATIVE_VRAM_DRAW & 0xFFFFFF));
					state.framebufptr = op;
				} else {
					u16 pitch = (op & 0xFFFF) != 512 ? (op & 0xFFFF) : GE_NATIVE_PITCH;
					geWrite(list, (cmd << 24) | ((GE_NATIVE_VRAM_DRAW >> 24) << 16) | pitch);
					state.framebufwidth = op;
					state.framebufwidth_addr = list;
				}

				if (state.framebufptr && state.framebufwidth) {
					state.framebuf = GE_NATIVE_FAKE_VRAM | (state.framebufptr & 0xFFFFFF);

					/* Only the 512-wide (display) framebuffer is upscaled. Draws
					   into other render targets would land in our buffer and are
					   skipped, their target is pointed at scratch VRAM. */
					u32 pitch = state.framebufwidth & 0xFFFF;
					if (pitch == 512 || pitch == GE_NATIVE_PITCH) {
						state.ignore_framebuf = 0;
					} else {
						geWrite(state.framebufwidth_addr, (GE_CMD_FRAMEBUFWIDTH << 24) | ((GE_NATIVE_VRAM_DUMMY >> 24) << 16) | 0);
						state.ignore_framebuf = 1;
					}

					insertFramebuf(state.framebuf);

					TRACE("F %08x ptr %06x width %06x -> %s\n", list, state.framebufptr & 0xFFFFFF, state.framebufwidth & 0xFFFFFF, state.ignore_framebuf ? "IGNORED" : "hires");

					state.framebufptr = 0;
					state.framebufwidth = 0;
				}
				break;
			}

			case GE_CMD_ZBUFPTR:
				geWrite(list, (cmd << 24) | (GE_NATIVE_VRAM_DEPTH & 0xFFFFFF));
				break;

			case GE_CMD_ZBUFWIDTH:
				geWrite(list, (cmd << 24) | ((GE_NATIVE_VRAM_DEPTH >> 24) << 16) | GE_NATIVE_PITCH);
				break;

			/* Textures that sample one of the framebuffers */

			case GE_CMD_TEXADDR0 ... GE_CMD_TEXADDR7:
			case GE_CMD_TEXBUFWIDTH0 ... GE_CMD_TEXBUFWIDTH7:
			{
				int index;
				if (cmd <= GE_CMD_TEXADDR7) {
					index = cmd - GE_CMD_TEXADDR0;
					state.texbufptr[index] = op;
				} else {
					index = cmd - GE_CMD_TEXBUFWIDTH0;
					state.texbufwidth[index] = op;
				}

				if (state.texbufptr[index] && state.texbufwidth[index]) {
					u32 texaddr = ((state.texbufwidth[index] & 0x0F0000) << 8) | (state.texbufptr[index] & 0xFFFFFF);
					state.ignore_texture = (texaddr != state.framebuf && findFramebuf(texaddr) >= 0);
					if (index == 0)
						state.tex0 = texaddr;
					TRACE("T%d %08x tex %08x w %d%s\n", index, list, texaddr, state.texbufwidth[index] & 0xFFFF, findFramebuf(texaddr) >= 0 ? (texaddr == state.framebuf ? " =current fb" : " =OTHER fb") : "");

					state.texbufptr[index] = 0;
					state.texbufwidth[index] = 0;
				}
				break;
			}

			/*
				Screen mapping, scaled from the game's values:
				  screen = v * scale + center - offset
				  2 * screen = v * (2 * scale) + (2 * center - 2048) - (2 * offset - 2048)
				so every command can be rewritten on its own.
			*/

			case GE_CMD_VIEWPORTXSCALE:
			case GE_CMD_VIEWPORTYSCALE:
				geSet(list, op, (cmd << 24) | geToF24(geF24(data) * 2.0f));
				TRACE("V %02x %08x %06x -> %06x\n", cmd, list, data, geRead(list) & 0xFFFFFF);
				break;

			case GE_CMD_VIEWPORTXCENTER:
			case GE_CMD_VIEWPORTYCENTER:
				geSet(list, op, (cmd << 24) | geToF24(geF24(data) * 2.0f - 2048.0f));
				TRACE("V %02x %08x %06x -> %06x\n", cmd, list, data, geRead(list) & 0xFFFFFF);
				break;

			case GE_CMD_OFFSETX:
			case GE_CMD_OFFSETY:
				geSet(list, op, (cmd << 24) | clampU((int)(data & 0xFFFF) * 2 - 2048 * 16, 0xFFFF));
				TRACE("V %02x %08x %06x -> %06x\n", cmd, list, data, geRead(list) & 0xFFFFFF);
				break;

			case GE_CMD_REGION1:
			case GE_CMD_SCISSOR1:
				geSet(list, op, (cmd << 24) | scaleCornerStart(data));
				TRACE("V %02x %08x %06x -> %06x\n", cmd, list, data, geRead(list) & 0xFFFFFF);
				break;

			case GE_CMD_REGION2:
			case GE_CMD_SCISSOR2:
				geSet(list, op, (cmd << 24) | scaleCornerEnd(data));
				TRACE("V %02x %08x %06x -> %06x\n", cmd, list, data, geRead(list) & 0xFFFFFF);
				break;

			case GE_CMD_TRANSFERSTART:
				TRACE("X %08x src %06x/%06x pos %06x dst %06x/%06x pos %06x size %06x bpp %d\n", list,
				      state.ge_cmd[GE_CMD_TRANSFERSRC], state.ge_cmd[GE_CMD_TRANSFERSRCW], state.ge_cmd[GE_CMD_TRANSFERSRCPOS],
				      state.ge_cmd[GE_CMD_TRANSFERDST], state.ge_cmd[GE_CMD_TRANSFERDSTW], state.ge_cmd[GE_CMD_TRANSFERDSTPOS],
				      state.ge_cmd[GE_CMD_TRANSFERSIZE], (data & 1) ? 32 : 16);
				break;

			case GE_CMD_CLEARMODE:
				TRACE("M %08x clear %06x\n", list, data);
				break;

			default:
				break;
		}
	}
}

int geNativeListBuildCopy(u32 *list) {
	int n = 0;
	u32 src = GE_NATIVE_VRAM_DRAW;
	u32 dst = GE_NATIVE_DISPLAY_BUFFER;

	list[n++] = (GE_CMD_TRANSFERSRC << 24) | (src & 0xFFFFFF);
	list[n++] = (GE_CMD_TRANSFERSRCW << 24) | ((src & 0xFF000000) >> 8) | GE_NATIVE_PITCH;
	list[n++] = (GE_CMD_TRANSFERSRCPOS << 24) | 0;
	list[n++] = (GE_CMD_TRANSFERDST << 24) | (dst & 0xFFFFFF);
	list[n++] = (GE_CMD_TRANSFERDSTW << 24) | ((dst & 0xFF000000) >> 8) | GE_NATIVE_PITCH;
	list[n++] = (GE_CMD_TRANSFERDSTPOS << 24) | 0;
	list[n++] = (GE_CMD_TRANSFERSIZE << 24) | ((GE_NATIVE_HEIGHT - 1) << 10) | (GE_NATIVE_WIDTH - 1);
	list[n++] = (GE_CMD_TRANSFERSTART << 24) | 0; /* 16 bpp */
	list[n++] = (GE_CMD_FINISH << 24);
	list[n++] = (GE_CMD_END << 24);

	return n;
}
