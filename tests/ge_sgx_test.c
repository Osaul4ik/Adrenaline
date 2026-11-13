/*
	Host unit tests for the SGX renderer's GE list walker (user/ge_sgx_walk.c).
	PSP RAM and VRAM are emulated with host buffers; run with: make -C tests
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ge_sgx_walk.h"

#define RAM_BASE  0x08000000u
#define RAM_SIZE  0x04000000u
#define VRAM_BASE 0x04000000u
#define VRAM_SIZE 0x00200000u

static uint8_t *ram, *vram;
static int failures = 0, checks = 0;
static int page_calls = 0;

#define CHECK(cond, ...) do { checks++; if (!(cond)) { failures++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static uint8_t *host(uint32_t addr) {
	addr &= 0x0FFFFFFF;
	if (addr >= RAM_BASE && addr < RAM_BASE + RAM_SIZE) return ram + (addr - RAM_BASE);
	if (addr >= VRAM_BASE && addr < VRAM_BASE + VRAM_SIZE) return vram + (addr - VRAM_BASE);
	return NULL;
}

static const void *pageFn(void *user, uint32_t page) {
	(void)user;
	page_calls++;
	if (page & (GE_SGX_BLOCK_SIZE - 1)) {
		printf("FATAL: unaligned page %08x\n", page);
		abort();
	}
	return host(page);
}

static void w32(uint32_t addr, uint32_t v) { memcpy(host(addr), &v, 4); }
static void w16(uint32_t addr, uint16_t v) { memcpy(host(addr), &v, 2); }
static void wf(uint32_t addr, float v) { memcpy(host(addr), &v, 4); }
static uint32_t C(uint32_t cmd, uint32_t data) { return (cmd << 24) | (data & 0xFFFFFF); }

typedef struct { uint32_t pos; } L;
static void emit(L *l, uint32_t op) { w32(l->pos, op); l->pos += 4; }

static GeSgxWalker walker;
static GeSgxWalker *w = &walker;

static void reset(void) {
	memset(ram, 0, RAM_SIZE);
	memset(vram, 0, VRAM_SIZE);
	geSgxWalkerInit(w, pageFn, NULL);
}

/* GE vertex type bits */
#define VT_TC16    (2u << 0)
#define VT_COL565  (4u << 2)
#define VT_COL5551 (5u << 2)
#define VT_COL4444 (6u << 2)
#define VT_COL8888 (7u << 2)
#define VT_NRM8    (1u << 5)
#define VT_POS16   (2u << 7)
#define VT_POSF    (3u << 7)
#define VT_WT8     (1u << 9)
#define VT_IDX8    (1u << 11)
#define VT_IDX16   (2u << 11)
#define VT_W(n)    (((n) - 1u) << 14)
#define VT_M(n)    (((n) - 1u) << 18)
#define VT_THROUGH (1u << 23)

static void testLayout(void) {
	GeSgxVertexLayout l;

	// sceGu 2D sprite: u16 uv, 8888 color, s16 pos
	geSgxVertexLayout(VT_TC16 | VT_COL8888 | VT_POS16 | VT_THROUGH, &l);
	CHECK(l.size == 16 && l.color_off == 4 && l.pos_off == 8, "tc16/8888/pos16: size %u col %u pos %u", l.size, l.color_off, l.pos_off);

	// color + float position
	geSgxVertexLayout(VT_COL8888 | VT_POSF, &l);
	CHECK(l.size == 16 && l.color_off == 0 && l.pos_off == 4, "8888/posf: size %u col %u pos %u", l.size, l.color_off, l.pos_off);

	// 565 + s16: 2 + 6 = 8
	geSgxVertexLayout(VT_COL565 | VT_POS16, &l);
	CHECK(l.size == 8 && l.pos_off == 2, "565/pos16: size %u pos %u", l.size, l.pos_off);

	// 3 u8 weights, u8 normal, s16 pos: weights 3, align 2 -> nrm at... no color
	geSgxVertexLayout(VT_WT8 | VT_W(3) | VT_NRM8 | VT_POS16, &l);
	// weights 0..2, normal 3..5, pos at 6..11, size 12
	CHECK(l.size == 12 && l.pos_off == 6, "wt8x3/nrm8/pos16: size %u pos %u", l.size, l.pos_off);

	// odd size padded to the biggest component: u8 normal + s8 pos = 6 -> 6
	geSgxVertexLayout(VT_NRM8 | (1u << 7), &l);
	CHECK(l.size == 6 && l.pos_off == 3, "nrm8/pos8: size %u pos %u", l.size, l.pos_off);

	// morph targets multiply the size
	geSgxVertexLayout(VT_COL8888 | VT_POSF | VT_M(2), &l);
	CHECK(l.size == 32, "morph 2: size %u", l.size);

	geSgxVertexLayout(VT_POS16 | VT_IDX16, &l);
	CHECK(l.index_size == 2, "idx16 size %u", l.index_size);
}

/* A clear of the whole 480x272 screen as sceGuClear emits it */
static void emitClear(L *l, uint32_t verts, uint32_t color, uint32_t fbp) {
	emit(l, C(0x9C, fbp));
	emit(l, C(0x9D, 512));
	emit(l, C(0xD3, 0x101 | 0x200 | 0x400)); // clear color, stencil, depth
	emit(l, C(0x12, VT_COL8888 | VT_POS16 | VT_THROUGH));
	emit(l, C(0x10, (verts >> 8) & 0x0F0000));
	emit(l, C(0x01, verts & 0xFFFFFF));
	emit(l, C(0x04, (6u << 16) | 2));
	emit(l, C(0xD3, 0));

	// vertex: color (4) + pos x,y,z (6) + pad (2) = 12
	w32(verts + 0, 0);
	w16(verts + 4, 0);
	w16(verts + 6, 0);
	w16(verts + 8, 0);
	w32(verts + 12, color);
	w16(verts + 16, 480);
	w16(verts + 18, 272);
	w16(verts + 20, 0);
}

static void testClearAndFlip(void) {
	reset();
	L l = { 0x08800000 };
	emitClear(&l, 0x08900000, 0x80332211, 0x000000);
	emit(&l, C(0x0F, 0));
	emit(&l, C(0x0C, 0));

	geSgxWalkerList(w, 1, 0x08800000, 0, 0);
	geSgxWalkerRun(w);
	CHECK(w->count == 0, "list should be finished, %d queued", w->count);

	GeSgxRect out[8];
	GeSgxStats st;
	int n = geSgxWalkerFlip(w, 0x44000000, out, 8, &st);
	CHECK(n == 1, "one clear rect, got %d", n);
	if (n == 1) {
		CHECK(out[0].x0 == 0 && out[0].y0 == 0 && out[0].x1 == 480 && out[0].y1 == 272,
		      "rect %d,%d-%d,%d", out[0].x0, out[0].y0, out[0].x1, out[0].y1);
		CHECK(out[0].color == 0xFF332211u, "color %08x (alpha forced opaque)", out[0].color);
	}
	CHECK(st.lists == 1 && st.prims == 1 && st.clears == 1 && st.vertices == 2 && st.bad == 0,
	      "stats lists %u prims %u clears %u verts %u bad %u", st.lists, st.prims, st.clears, st.vertices, st.bad);
	CHECK(st.commands == 10, "commands %u", st.commands);

	// stats and rects are per frame
	n = geSgxWalkerFlip(w, 0x44000000, out, 8, &st);
	CHECK(n == 0 && st.commands == 0, "second flip: %d rects, %u commands", n, st.commands);
}

static void testDoubleBuffer(void) {
	reset();
	// frame: clear back buffer B (0x88000), flip B. The rect of A drawn
	// earlier must not show on B.
	L l = { 0x08800000 };
	emitClear(&l, 0x08900000, 0xFF0000FF, 0x000000);
	emitClear(&l, 0x08900100, 0xFF00FF00, 0x088000);
	emit(&l, C(0x0C, 0));
	geSgxWalkerList(w, 1, 0x08800000, 0, 0);
	geSgxWalkerRun(w);

	GeSgxRect out[8];
	int n = geSgxWalkerFlip(w, 0x04088000, out, 8, NULL);
	CHECK(n == 1 && out[0].color == 0xFF00FF00u, "flip B: %d rects color %08x", n, n ? out[0].color : 0);

	// A was drawn in the same frame: still there for the next flip
	n = geSgxWalkerFlip(w, 0x04000000, out, 8, NULL);
	CHECK(n == 1 && out[0].color == 0xFF0000FFu, "flip A: %d rects", n);

	// framebuffer in RAM (CPU drawn): no rects
	l.pos = 0x08800000;
	emitClear(&l, 0x08900000, 0xFF0000FF, 0x000000);
	emit(&l, C(0x0C, 0));
	geSgxWalkerList(w, 2, 0x08800000, 0, 0);
	geSgxWalkerRun(w);
	n = geSgxWalkerFlip(w, 0x08800000, out, 8, NULL);
	CHECK(n == 0, "RAM framebuffer: %d rects", n);
}

static void testStall(void) {
	reset();
	// list built while the GE runs: stall at the CLEARMODE, then further
	L l = { 0x08800000 };
	emitClear(&l, 0x08900000, 0xFF123456, 0);
	uint32_t mid = l.pos;
	emit(&l, C(0x0C, 0));
	// at first only part of the list is there
	uint32_t stall1 = 0x08800000 + 12; // before VTYPE
	geSgxWalkerList(w, 3, 0x08800000, stall1, 0);
	geSgxWalkerRun(w);
	CHECK(w->count == 1 && w->lists[0].pc == stall1, "stalled at %08x (pc %08x)", stall1, w->count ? w->lists[0].pc : 0);

	// a second list waits behind the stalled one
	L l2 = { 0x08810000 };
	emit(&l2, C(0x9C, 0x44000));
	emit(&l2, C(0x0C, 0));
	geSgxWalkerList(w, 4, 0x08810000, 0, 0);
	geSgxWalkerRun(w);
	CHECK(w->count == 2 && geSgxReg(w, 0x9C) == 0, "second list must wait: count %d fbp %06x", w->count, geSgxReg(w, 0x9C));

	geSgxWalkerStall(w, 3, mid + 4);
	geSgxWalkerRun(w);
	CHECK(w->count == 0, "both finished, count %d", w->count);
	CHECK(geSgxReg(w, 0x9C) == 0x44000, "second list ran after: fbp %06x", geSgxReg(w, 0x9C));

	GeSgxRect out[4];
	GeSgxStats st;
	int n = geSgxWalkerFlip(w, 0x04000000, out, 4, &st);
	CHECK(n == 1 && st.lists == 2, "rects %d lists %u", n, st.lists);
}

static void testCallJumpOrigin(void) {
	reset();
	// main: CALL sub (relative to BASE), JUMP to tail, (skipped NOP area), tail: END
	L m = { 0x08800000 };
	emit(&m, C(0x10, 0x080000));       // BASE: high byte 0x08
	emit(&m, C(0x0A, 0x820000));       // CALL 0x08820000
	emit(&m, C(0x08, 0x800100));       // JUMP 0x08800100
	emit(&m, C(0x9C, 0x111111));       // skipped
	L tail = { 0x08800100 };
	emit(&tail, C(0x0C, 0));

	L sub = { 0x08820000 };
	emit(&sub, C(0x13, 0x08830));      // OFFSETADDR: offset 0x08830000 - base part...
	emit(&sub, C(0x9D, 0x123));
	emit(&sub, C(0x0B, 0));            // RET restores offset

	geSgxWalkerList(w, 1, 0x08800000, 0, 0);
	geSgxWalkerRun(w);
	CHECK(w->count == 0 && w->stats.bad == 0, "count %d bad %u", w->count, w->stats.bad);
	CHECK(geSgxReg(w, 0x9D) == 0x123, "sub ran: fbw %x", geSgxReg(w, 0x9D));
	CHECK(geSgxReg(w, 0x9C) == 0, "jump skipped the FBP: %x", geSgxReg(w, 0x9C));
	CHECK(w->offset == 0, "RET restored the offset: %x", w->offset);
	CHECK(w->stats.commands == 7, "commands %u", w->stats.commands);

	// ORIGIN: offset = address of the ORIGIN command, then VADDR relative to it
	reset();
	L o = { 0x08800000 };
	emit(&o, C(0x10, 0));
	emit(&o, C(0x14, 0));              // at 0x08800004
	emit(&o, C(0x01, 0x10));           // vaddr = 0x08800014
	emit(&o, C(0x0C, 0));
	geSgxWalkerList(w, 1, 0x08800000, 0, 0);
	geSgxWalkerRun(w);
	CHECK(w->vaddr == 0x08800014, "origin vaddr %08x", w->vaddr);
}

static void testSignals(void) {
	reset();
	// SIGNAL JUMP to 0x08840000, SIGNAL CALL / RET, handler signal continues
	L m = { 0x08800000 };
	emit(&m, C(0x0E, (0x11u << 16) | 0x0885)); // CALL 0x08850000
	emit(&m, C(0x0C, 0x0000));
	emit(&m, C(0x0E, (0x01u << 16) | 0x0001)); // handler signal: continue after END
	emit(&m, C(0x0C, 0x0000));
	emit(&m, C(0x0E, (0x10u << 16) | 0x0884)); // JUMP 0x08840000
	emit(&m, C(0x0C, 0x0000));
	emit(&m, C(0x9C, 0x999999));               // skipped

	L j = { 0x08840000 };
	emit(&j, C(0x9C, 0x22000));
	emit(&j, C(0x0F, 0));
	emit(&j, C(0x0C, 0));

	L s = { 0x08850000 };
	emit(&s, C(0x9D, 0x77));
	emit(&s, C(0x0E, (0x12u << 16)));          // RET
	emit(&s, C(0x0C, 0));

	geSgxWalkerList(w, 1, 0x08800000, 0, 0);
	geSgxWalkerRun(w);
	CHECK(w->count == 0 && w->stats.bad == 0, "count %d bad %u", w->count, w->stats.bad);
	CHECK(geSgxReg(w, 0x9D) == 0x77 && geSgxReg(w, 0x9C) == 0x22000, "fbw %x fbp %x", geSgxReg(w, 0x9D), geSgxReg(w, 0x9C));
	CHECK(w->stats.signals == 4, "signals %u", w->stats.signals);

	// RJUMP: relative to the SIGNAL
	reset();
	L r = { 0x08800000 };
	emit(&r, C(0x0E, (0x13u << 16) | 0x0000));
	emit(&r, C(0x0C, 0x0010));                 // +0x10
	emit(&r, C(0x9C, 0x999999));               // skipped
	emit(&r, C(0x9C, 0x999999));               // skipped
	emit(&r, C(0x9C, 0x33000));                // 0x08800010
	emit(&r, C(0x0C, 0));
	geSgxWalkerList(w, 1, 0x08800000, 0, 0);
	geSgxWalkerRun(w);
	CHECK(geSgxReg(w, 0x9C) == 0x33000, "rjump fbp %x", geSgxReg(w, 0x9C));
}

static void testBadAndRunaway(void) {
	reset();
	// jump into unmapped memory
	L m = { 0x08800000 };
	emit(&m, C(0x10, 0x000000));
	emit(&m, C(0x08, 0x100000)); // 0x00100000: not readable
	geSgxWalkerList(w, 1, 0x08800000, 0, 0);
	geSgxWalkerRun(w);
	CHECK(w->count == 0 && w->stats.bad == 1, "bad list dropped: count %d bad %u", w->count, w->stats.bad);

	// list jumping to itself forever
	reset();
	L loop = { 0x08800000 };
	emit(&loop, C(0x10, 0x080000));
	emit(&loop, C(0x08, 0x800004));
	geSgxWalkerList(w, 1, 0x08800000, 0, 0);
	geSgxWalkerRun(w);
	CHECK(w->count == 0 && w->stats.bad == 1, "runaway list dropped: count %d bad %u", w->count, w->stats.bad);

	// call stack overflow
	reset();
	L rec = { 0x08800000 };
	emit(&rec, C(0x10, 0x080000));
	emit(&rec, C(0x0A, 0x800004)); // calls itself
	geSgxWalkerList(w, 1, 0x08800000, 0, 0);
	geSgxWalkerRun(w);
	CHECK(w->count == 0 && w->stats.bad == 1, "recursion dropped: bad %u", w->stats.bad);

	// unaligned list address is aligned down, not rejected
	reset();
	L a = { 0x08800000 };
	emit(&a, C(0x9C, 0x1000));
	emit(&a, C(0x0C, 0));
	geSgxWalkerList(w, 1, 0x48800002, 0, 0);
	geSgxWalkerRun(w);
	CHECK(geSgxReg(w, 0x9C) == 0x1000 && w->stats.bad == 0, "uncached unaligned list address: fbp %x", geSgxReg(w, 0x9C));
}

static void testQueue(void) {
	reset();
	L a = { 0x08800000 };
	emit(&a, C(0x9D, 0xA));
	emit(&a, C(0x0C, 0));
	L b = { 0x08810000 };
	emit(&b, C(0x9D, 0xB));
	emit(&b, C(0x0C, 0));

	// both queued before anything runs, B at the head: B then A
	geSgxWalkerList(w, 1, 0x08800000, 0x08800000, 0); // stalled at its start
	geSgxWalkerList(w, 2, 0x08810000, 0, 1);
	CHECK(w->lists[0].qid == 2, "head list first: %d", w->lists[0].qid);
	geSgxWalkerStall(w, 1, 0);
	geSgxWalkerRun(w);
	CHECK(geSgxReg(w, 0x9D) == 0xA, "A ran last: fbw %x", geSgxReg(w, 0x9D));

	// head list while one is running: goes after the running one
	reset();
	emit(&(L){ 0x08800000 }, C(0x9D, 0xA));
	geSgxWalkerList(w, 1, 0x08800000, 0x08800004, 0);
	geSgxWalkerRun(w); // started, stalled
	geSgxWalkerList(w, 2, 0x08810000, 0, 1);
	CHECK(w->count == 2 && w->lists[0].qid == 1 && w->lists[1].qid == 2, "running list stays first");

	// dequeue removes it, qid reuse replaces
	geSgxWalkerDequeue(w, 1);
	CHECK(w->count == 1 && w->lists[0].qid == 2, "dequeued");
	geSgxWalkerList(w, 2, 0x08800000, 0x08800000, 0);
	CHECK(w->count == 1, "qid reuse replaces: count %d", w->count);

	// full queue drops
	reset();
	int i;
	for (i = 0; i < GE_SGX_MAX_LISTS + 3; i++)
		geSgxWalkerList(w, 100 + i, 0x08800000, 0x08800000, 0);
	CHECK(w->count == GE_SGX_MAX_LISTS && w->stats.dropped == 3, "count %d dropped %u", w->count, w->stats.dropped);
}

static void testIndexedAndAdvance(void) {
	reset();
	// two clears with one VADDR: the second PRIM continues after the first
	uint32_t v = 0x08900000;
	L l = { 0x08800000 };
	emit(&l, C(0x9C, 0));
	emit(&l, C(0x9D, 512));
	emit(&l, C(0xD3, 0x101));
	emit(&l, C(0x12, VT_COL565 | VT_POS16 | VT_THROUGH)); // 8 bytes
	emit(&l, C(0x10, 0x080000));
	emit(&l, C(0x01, 0x900000));
	emit(&l, C(0x04, (6u << 16) | 2));
	emit(&l, C(0x04, (6u << 16) | 2));
	// indexed: u8 indices 1,0 -> reversed corners
	emit(&l, C(0x12, VT_COL565 | VT_POS16 | VT_THROUGH | VT_IDX8));
	emit(&l, C(0x01, 0x900000));
	emit(&l, C(0x02, 0x900100));
	emit(&l, C(0x04, (6u << 16) | 2));
	emit(&l, C(0x04, (6u << 16) | 2)); // continues at the next indices
	emit(&l, C(0x0C, 0));

	// rect 1: (10,20)-(30,40) white 565; rect 2: (50,60)-(70,80) red
	w16(v + 0, 0); w16(v + 2, 10); w16(v + 4, 20); w16(v + 6, 0);
	w16(v + 8, 0xFFFF); w16(v + 10, 30); w16(v + 12, 40); w16(v + 14, 0);
	w16(v + 16, 0); w16(v + 18, 50); w16(v + 20, 60); w16(v + 22, 0);
	w16(v + 24, 0x001F); w16(v + 26, 70); w16(v + 28, 80); w16(v + 30, 0);
	*host(0x08900100) = 1;
	*host(0x08900101) = 0;
	*host(0x08900102) = 2;
	*host(0x08900103) = 3;

	geSgxWalkerList(w, 1, 0x08800000, 0, 0);
	geSgxWalkerRun(w);

	GeSgxRect out[8];
	int n = geSgxWalkerFlip(w, 0x04000000, out, 8, NULL);
	CHECK(n == 4, "four rects, got %d", n);
	if (n == 4) {
		CHECK(out[0].x0 == 10 && out[0].y1 == 40 && out[0].color == 0xFFFFFFFFu, "rect 1 %d %d %08x", out[0].x0, out[0].y1, out[0].color);
		CHECK(out[1].x0 == 50 && out[1].y1 == 80 && out[1].color == 0xFF0000FFu, "rect 2 %d %d %08x", out[1].x0, out[1].y1, out[1].color);
		// indexed: v1 = vertex 0 (white, 10,20), v0 = vertex 1 (30,40): sorted
		CHECK(out[2].x0 == 10 && out[2].x1 == 30 && out[2].color == 0xFF000000u, "indexed rect %d-%d %08x", out[2].x0, out[2].x1, out[2].color);
		CHECK(out[3].x0 == 50 && out[3].color == 0xFF0000FFu, "second indexed rect %d %08x", out[3].x0, out[3].color);
	}
}

static void testColors(void) {
	reset();
	// 4444 and 5551 colors, float positions
	uint32_t v = 0x08900000;
	L l = { 0x08800000 };
	emit(&l, C(0x9C, 0));
	emit(&l, C(0xD3, 0x101));
	emit(&l, C(0x10, 0x080000));
	emit(&l, C(0x12, VT_COL4444 | VT_POSF | VT_THROUGH)); // 2 + pad 2 + 12 = 16
	emit(&l, C(0x01, 0x900000));
	emit(&l, C(0x04, (6u << 16) | 2));
	emit(&l, C(0x12, VT_COL5551 | VT_POS16 | VT_THROUGH)); // 8
	emit(&l, C(0x01, 0x900100));
	emit(&l, C(0x04, (6u << 16) | 2));
	// not in clear mode: no rect
	emit(&l, C(0xD3, 0));
	emit(&l, C(0x01, 0x900100));
	emit(&l, C(0x04, (6u << 16) | 2));
	emit(&l, C(0x0C, 0));

	wf(v + 4, 1.0f); wf(v + 8, 2.0f);
	w16(v + 16, 0x0F84); wf(v + 20, 100.5f); wf(v + 24, 50.0f);
	w16(0x08900100 + 0, 0); w16(0x08900102, 5); w16(0x08900104, 6);
	w16(0x08900108, 0x7C00); w16(0x0890010A, 7); w16(0x0890010C, 9);

	geSgxWalkerList(w, 1, 0x08800000, 0, 0);
	geSgxWalkerRun(w);
	GeSgxRect out[8];
	GeSgxStats st;
	int n = geSgxWalkerFlip(w, 0x04000000, out, 8, &st);
	CHECK(n == 2 && st.prims == 3, "rects %d prims %u", n, st.prims);
	if (n == 2) {
		// 0x0F84: r=4 g=8 b=F -> 44 88 FF
		CHECK(out[0].color == 0xFFFF8844u && out[0].x1 == 100 && out[0].y1 == 50, "4444 %08x %d %d", out[0].color, out[0].x1, out[0].y1);
		// 0x7C00 5551: b = 31
		CHECK(out[1].color == 0xFFFF0000u, "5551 %08x", out[1].color);
	}
}

static void testLongList(void) {
	reset();
	// a list over many fetch blocks, starting in the middle of one
	L l = { 0x088000F8 };
	int i;
	for (i = 0; i < 1000; i++)
		emit(&l, C(0x9D, (uint32_t)i));
	emit(&l, C(0x0C, 0));
	int calls = page_calls;
	geSgxWalkerList(w, 1, 0x088000F8, 0, 0);
	geSgxWalkerRun(w);
	CHECK(geSgxReg(w, 0x9D) == 999 && w->stats.commands == 1001, "long list: fbw %u commands %u", geSgxReg(w, 0x9D), w->stats.commands);
	// 4004 bytes from offset 0xF8: blocks 0x08800000 .. 0x08800F00 -> 16
	CHECK(page_calls - calls == (int)((0xF8 + 4004 + GE_SGX_BLOCK_SIZE - 1) / GE_SGX_BLOCK_SIZE), "block fetches %d", page_calls - calls);
}

int main(void) {
	ram = calloc(1, RAM_SIZE);
	vram = calloc(1, VRAM_SIZE);
	if (!ram || !vram)
		return 2;

	testLayout();
	testClearAndFlip();
	testDoubleBuffer();
	testStall();
	testCallJumpOrigin();
	testSignals();
	testBadAndRunaway();
	testQueue();
	testIndexedAndAdvance();
	testColors();
	testLongList();

	printf("ge_sgx_test: %d checks, %d failures (page fetches %d)\n", checks, failures, page_calls);
	free(ram);
	free(vram);
	return failures ? 1 : 0;
}
