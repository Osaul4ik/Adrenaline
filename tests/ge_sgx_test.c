/*
	Host unit tests for the SGX renderer's GE list walker (user/ge_sgx_walk.c).
	PSP RAM and VRAM are emulated with host buffers; run with: make -C tests
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ge_sgx_walk.h"
#include "ge_sgx_texconv.h"
#include "ge_sgx_gpu.h"

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
static void w8(uint32_t addr, uint8_t v) { memcpy(host(addr), &v, 1); }
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

/* ---- draw building (stage 1: through mode) ---- */

static GeSgxVertex fverts[4096];
static uint16_t fidx[16384];
static GeSgxDraw fdraws[64];
static GeSgxFrameData frame;
static GeSgxTexDesc last_desc;
static int tex_calls;

static int32_t texFn(void *user, const GeSgxTexDesc *d) {
	(void)user;
	last_desc = *d;
	tex_calls++;
	return d->format == GE_TFMT_DXT1 ? -1 : 7;
}

static void resetFrame(void) {
	reset();
	memset(&frame, 0, sizeof(frame));
	frame.vertices = fverts;
	frame.max_vertices = 4096;
	frame.indices = fidx;
	frame.max_indices = 16384;
	frame.draws = fdraws;
	frame.max_draws = 64;
	w->frame = &frame;
	w->texture_fn = texFn;
	w->texture_user = NULL;
	tex_calls = 0;
}

static int feq(float a, float b) { return a > b - 0.0001f && a < b + 0.0001f; }

/* sprite vertex: u16 uv, 8888 color, s16 xyz = 4 + 4 + 6 (+2 pad) = 16 */
static void spriteVertex(uint32_t a, uint16_t u, uint16_t v, uint32_t color, int16_t x, int16_t y) {
	w16(a, u); w16(a + 2, v); w32(a + 4, color); w16(a + 8, (uint16_t)x); w16(a + 10, (uint16_t)y); w16(a + 12, 0);
}

static void testBuildSprites(void) {
	resetFrame();
	uint32_t v = 0x08900000;
	spriteVertex(v + 0, 0, 0, 0x11111111, 10, 20);
	spriteVertex(v + 16, 32, 16, 0xFF00FF00, 42, 36);
	spriteVertex(v + 32, 32, 0, 0x22222222, 100, 100);
	spriteVertex(v + 48, 64, 16, 0xFF0000FF, 132, 116);

	L l = { 0x08800000 };
	emit(&l, C(0x9C, 0x44000));
	emit(&l, C(0x9D, 512));
	emit(&l, C(0xD4, (0 << 10) | 0));
	emit(&l, C(0xD5, (271 << 10) | 479));
	emit(&l, C(0x1E, 1));                  // texture on
	emit(&l, C(0x21, 1));                  // blend on
	emit(&l, C(0xDF, (3 << 4) | 2));       // src alpha, 1 - src alpha
	emit(&l, C(0xC9, 0x100));              // modulate, RGBA
	emit(&l, C(0xA0, 0x123450));           // texture 0x09123450
	emit(&l, C(0xA8, (0x09 << 16) | 64));  // bufw 64, upper address 0x09
	emit(&l, C(0xB8, (4 << 8) | 6));       // 64 x 16
	emit(&l, C(0xC3, GE_TFMT_8888));
	emit(&l, C(0x12, VT_TC16 | VT_COL8888 | VT_POS16 | VT_THROUGH));
	emit(&l, C(0x10, 0x080000));
	emit(&l, C(0x01, 0x900000));
	emit(&l, C(0x04, (6u << 16) | 2));
	emit(&l, C(0x04, (6u << 16) | 2));     // next two vertices: merged into the same draw
	emit(&l, C(0x0C, 0));
	geSgxWalkerList(w, 1, 0x08800000, 0, 0);
	geSgxWalkerRun(w);

	CHECK(frame.ndraws == 1 && frame.nvertices == 8 && frame.nindices == 12, "draws %u vertices %u indices %u", frame.ndraws, frame.nvertices, frame.nindices);
	/* the second PRIM in the same state: no new lookup, no rebuild */
	CHECK(tex_calls == 1 && frame.state_builds == 1, "texture lookups %d, state builds %u", tex_calls, frame.state_builds);
	CHECK(last_desc.addr == 0x09123450 && last_desc.bufw == 64 && last_desc.w == 64 && last_desc.h == 16 && last_desc.format == GE_TFMT_8888 && !last_desc.swizzle,
	      "desc addr %08x bufw %u %ux%u fmt %u", last_desc.addr, last_desc.bufw, last_desc.w, last_desc.h, last_desc.format);
	if (frame.ndraws == 1) {
		GeSgxDraw *d = &fdraws[0];
		CHECK(d->prim == GE_SGX_DRAW_TRIANGLES && d->base == 0 && d->first == 0 && d->count == 12 && d->texture == 7, "prim %u first %u count %u tex %d", d->prim, d->first, d->count, d->texture);
		CHECK(d->ztest == 1 && !(d->flags & (GE_SGX_DRAW_WRITE_DEPTH | GE_SGX_DRAW_FOG)), "2D depth %u flags %x", d->ztest, d->flags);
		CHECK((d->flags & (GE_SGX_DRAW_TEXTURED | GE_SGX_DRAW_BLEND | GE_SGX_DRAW_TCC)) == (GE_SGX_DRAW_TEXTURED | GE_SGX_DRAW_BLEND | GE_SGX_DRAW_TCC) && !(d->flags & GE_SGX_DRAW_CLEAR),
		      "flags %02x", d->flags);
		CHECK(d->blend == 0x32 && d->tfunc == 0, "blend %x tfunc %u", d->blend, d->tfunc);
		CHECK(d->scissor[0] == 0 && d->scissor[1] == 0 && d->scissor[2] == 960 && d->scissor[3] == 544, "scissor %d %d %d %d", d->scissor[0], d->scissor[1], d->scissor[2], d->scissor[3]);
		CHECK(d->fb == 0x44000, "fb %x", d->fb);
	}
	/* first sprite: corners (20,40) (84,40) (20,72) (84,72), triangles 0 1 2, 1 3 2, colour of vertex 2 */
	CHECK(feq(fverts[0].x, 20) && feq(fverts[0].y, 40) && feq(fverts[0].u, 0) && feq(fverts[0].v, 0) && feq(fverts[0].w, 1), "v0 %f %f %f %f", fverts[0].x, fverts[0].y, fverts[0].u, fverts[0].v);
	CHECK(feq(fverts[1].x, 84) && feq(fverts[1].y, 40) && feq(fverts[1].u, 0.5f) && feq(fverts[1].v, 0), "v1 %f %f %f %f", fverts[1].x, fverts[1].y, fverts[1].u, fverts[1].v);
	CHECK(feq(fverts[3].x, 84) && feq(fverts[3].y, 72) && feq(fverts[3].u, 0.5f) && feq(fverts[3].v, 1.0f), "v3 %f %f %f %f", fverts[3].x, fverts[3].y, fverts[3].u, fverts[3].v);
	CHECK(fverts[0].color == 0xFF00FF00u && fverts[2].color == 0xFF00FF00u && feq(fverts[0].fog, 1), "sprite colour %08x", fverts[0].color);
	CHECK(fidx[0] == 0 && fidx[1] == 1 && fidx[2] == 2 && fidx[3] == 1 && fidx[4] == 3 && fidx[5] == 2, "sprite indices %u %u %u %u %u %u", fidx[0], fidx[1], fidx[2], fidx[3], fidx[4], fidx[5]);
	CHECK(feq(fverts[4].x, 200) && feq(fverts[4].u, 0.5f) && fverts[4].color == 0xFF0000FFu && fidx[6] == 4 && fidx[11] == 6, "second sprite %f %f %08x idx %u", fverts[4].x, fverts[4].u, fverts[4].color, fidx[6]);
}

static void testBuildOther(void) {
	resetFrame();
	uint32_t v = 0x08900000;
	/* 565 colour + s16 pos: 2 + 6 = 8 bytes, no texture */
	int i;
	for (i = 0; i < 5; i++) {
		w16(v + i * 8, 0xF800);           // blue
		w16(v + i * 8 + 2, (uint16_t)(i * 10));
		w16(v + i * 8 + 4, (uint16_t)(i * 5));
		w16(v + i * 8 + 6, 0xFFFF);       // z
	}
	L l = { 0x08800000 };
	emit(&l, C(0x12, VT_COL565 | VT_POS16 | VT_THROUGH));
	emit(&l, C(0x10, 0x080000));
	emit(&l, C(0x01, 0x900000));
	emit(&l, C(0x04, (2u << 16) | 3));   // line strip of 3 -> 2 lines
	emit(&l, C(0x01, 0x900000));
	emit(&l, C(0x04, (5u << 16) | 5));   // fan of 5
	emit(&l, C(0x12, VT_COL565 | VT_POS16)); // 3D, same state: joins the fan's draw
	emit(&l, C(0x04, (3u << 16) | 3));
	emit(&l, C(0x12, VT_COL565 | VT_POS16 | VT_THROUGH));
	emit(&l, C(0x1E, 1));
	emit(&l, C(0xC3, GE_TFMT_DXT1));     // unsupported texture: skipped
	emit(&l, C(0x01, 0x900000));
	emit(&l, C(0x04, (3u << 16) | 3));
	emit(&l, C(0x0C, 0));
	geSgxWalkerList(w, 1, 0x08800000, 0, 0);
	geSgxWalkerRun(w);

	CHECK(frame.ndraws == 2, "draws %u", frame.ndraws);
	CHECK(frame.draws_3d == 1 && frame.skipped_other == 1, "3d %u skipped other %u", frame.draws_3d, frame.skipped_other);
	if (frame.ndraws == 2) {
		CHECK(fdraws[0].prim == GE_SGX_DRAW_LINES && fdraws[0].count == 4 && fidx[0] == 0 && fidx[1] == 1 && fidx[2] == 1 && fidx[3] == 2,
		      "line strip -> %u, %u indices", fdraws[0].prim, fdraws[0].count);
		/* fan 0 1 2, 0 2 3, 0 3 4 over vertices 3..7, then the 3D triangle 8..10 */
		CHECK(fdraws[1].prim == GE_SGX_DRAW_TRIANGLES && fdraws[1].count == 12 && fdraws[1].first == 4 && fdraws[1].base == 3,
		      "fan %u %u %u base %u", fdraws[1].prim, fdraws[1].count, fdraws[1].first, fdraws[1].base);
		CHECK(fidx[4] == 0 && fidx[5] == 1 && fidx[6] == 2 && fidx[7] == 0 && fidx[8] == 2 && fidx[9] == 3 && fidx[13] == 5 && fidx[15] == 7,
		      "fan/3d indices %u %u %u .. %u %u", fidx[4], fidx[5], fidx[6], fidx[13], fidx[15]);
	}
	/* lines: (0,0)-(10,5), (10,5)-(20,10) scaled */
	CHECK(feq(fverts[1].x, 20) && feq(fverts[1].y, 10) && feq(fverts[2].x, 40) && feq(fverts[2].z, 1.0f), "line 2 %f %f %f %f", fverts[1].x, fverts[1].y, fverts[2].x, fverts[2].z);
	CHECK(fverts[0].color == 0xFFFF0000u, "565 blue %08x", fverts[0].color);
}

static void testBuildClearAndClut(void) {
	resetFrame();
	/* CLUT in memory, loaded with LOADCLUT, then the CLUT texture desc carries it */
	int i;
	for (i = 0; i < 16; i++)
		w32(0x08A00000 + i * 4, 0xFF000000u | (uint32_t)i);
	L l = { 0x08800000 };
	emit(&l, C(0xB0, 0xA00000));
	emit(&l, C(0xB1, 0x080000));
	emit(&l, C(0xC5, 3 | (0xFF << 8)));   // 8888, mask 0xFF
	emit(&l, C(0xC4, 2));                 // 64 bytes
	emit(&l, C(0x1E, 1));
	emit(&l, C(0xC3, GE_TFMT_CLUT4));
	emit(&l, C(0xC2, 1));                 // swizzled
	emit(&l, C(0xB8, (3 << 8) | 4));      // 16 x 8
	emit(&l, C(0x12, VT_TC16 | VT_COL8888 | VT_POS16 | VT_THROUGH));
	emit(&l, C(0x10, 0x080000));
	emit(&l, C(0x01, 0x900000));
	emit(&l, C(0x04, (6u << 16) | 2));
	/* clear mode sprite: no texture, own flags */
	emit(&l, C(0xD3, 0x101));
	emit(&l, C(0x01, 0x900000));
	emit(&l, C(0x04, (6u << 16) | 2));
	emit(&l, C(0x0C, 0));
	spriteVertex(0x08900000, 0, 0, 0xFFFFFFFF, 0, 0);
	spriteVertex(0x08900010, 16, 8, 0xFFFFFFFF, 16, 8);
	geSgxWalkerList(w, 1, 0x08800000, 0, 0);
	geSgxWalkerRun(w);

	CHECK(tex_calls == 1 && last_desc.clut_bytes == 64 && last_desc.swizzle == 1 && last_desc.w == 16 && last_desc.h == 8, "clut desc calls %d bytes %u swz %u", tex_calls, last_desc.clut_bytes, last_desc.swizzle);
	for (i = 0; i < 16; i++)              // changed after the load: the snapshot keeps the loaded CLUT
		w32(0x08A00000 + i * 4, 0);
	uint32_t e5 = 0;
	if (last_desc.clut)
		memcpy(&e5, last_desc.clut + 5 * 4, 4);
	CHECK(e5 == 0xFF000005u, "clut snapshot entry 5 %08x", e5);
	CHECK(frame.ndraws == 2 && (fdraws[1].flags & GE_SGX_DRAW_CLEAR) && (fdraws[1].flags & GE_SGX_DRAW_WRITE_COLOR) && !(fdraws[1].flags & GE_SGX_DRAW_TEXTURED) && fdraws[1].texture == -1,
	      "clear draw %u flags %02x", frame.ndraws, frame.ndraws > 1 ? fdraws[1].flags : 0);
}

static uint32_t F24(float f) {
	uint32_t u;
	memcpy(&u, &f, 4);
	return u >> 8;
}


static void matrix(L *l, uint32_t num_cmd, const float *m, int n) {
	int i;
	emit(l, C(num_cmd, 0));
	for (i = 0; i < n; i++)
		emit(l, C(num_cmd + 1, F24(m[i])));
}

/* identity world and view, w = -z, PSP viewport 480x272 around 2048 */
static void setup3D(L *l) {
	static const float ident[12] = { 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0 };
	static const float proj[16] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0 };
	matrix(l, 0x3A, ident, 12);
	matrix(l, 0x3C, ident, 12);
	matrix(l, 0x3E, proj, 16);
	emit(l, C(0x42, F24(240.0f)));
	emit(l, C(0x43, F24(-136.0f)));
	emit(l, C(0x44, F24(-32767.5f)));
	emit(l, C(0x45, F24(2048.0f)));
	emit(l, C(0x46, F24(2048.0f)));
	emit(l, C(0x47, F24(32767.5f)));
	emit(l, C(0x4C, (2048 - 240) * 16));
	emit(l, C(0x4D, (2048 - 136) * 16));
	emit(l, C(0x48, F24(1.0f)));
	emit(l, C(0x49, F24(1.0f)));
	emit(l, C(0x4A, F24(0.0f)));
	emit(l, C(0x4B, F24(0.0f)));
}

static void testBuild3D(void) {
	resetFrame();
	/* float uv, 8888 colour, float xyz: 8 + 4 + 12 = 24 bytes */
	uint32_t v = 0x08900000;
	static const float pos[6][3] = { { 9, 9, 9 }, { 9, 9, 9 }, { 0, 0, -2 }, { 1, 1, -2 }, { -1, 1, -4 }, { 1, -1, -4 } };
	int i;
	for (i = 0; i < 6; i++) {
		wf(v + i * 24, 0.25f);
		wf(v + i * 24 + 4, 0.75f);
		w32(v + i * 24 + 8, 0x80FF0000u + (uint32_t)i);
		wf(v + i * 24 + 12, pos[i][0]);
		wf(v + i * 24 + 16, pos[i][1]);
		wf(v + i * 24 + 20, pos[i][2]);
	}
	w32(0x08910000, 0x05040302u);         // 8 bit indices 2 3 4 5

	L l = { 0x08800000 };
	setup3D(&l);
	emit(&l, C(0x1F, 1));                 // fog: end 2, slope 0.25
	emit(&l, C(0xCD, F24(2.0f)));
	emit(&l, C(0xCE, F24(0.25f)));
	emit(&l, C(0xCF, 0x123456));
	emit(&l, C(0x23, 1));                 // depth test GEQUAL, written
	emit(&l, C(0xDE, 7));
	emit(&l, C(0x12, (3u << 0) | VT_COL8888 | VT_POSF | VT_IDX8));
	emit(&l, C(0x10, 0x080000));
	emit(&l, C(0x01, 0x900000));
	emit(&l, C(0x02, 0x910000));
	emit(&l, C(0x04, (4u << 16) | 4));    // indexed strip of 4
	emit(&l, C(0x0C, 0));
	geSgxWalkerList(w, 1, 0x08800000, 0, 0);
	geSgxWalkerRun(w);

	CHECK(frame.ndraws == 1 && frame.nvertices == 4 && frame.nindices == 6 && frame.draws_3d == 1, "3d draws %u verts %u idx %u", frame.ndraws, frame.nvertices, frame.nindices);
	if (frame.ndraws == 1) {
		GeSgxDraw *d = &fdraws[0];
		CHECK(d->ztest == 7 && (d->flags & GE_SGX_DRAW_WRITE_DEPTH) && (d->flags & GE_SGX_DRAW_FOG) && d->fog_color == 0x123456, "3d state z %u flags %x fog %x", d->ztest, d->flags, d->fog_color);
	}
	/* strip 0 1 2, then 2 1 3 (odd triangle swapped) over vertices 2..5 */
	CHECK(fidx[0] == 0 && fidx[1] == 1 && fidx[2] == 2 && fidx[3] == 2 && fidx[4] == 1 && fidx[5] == 3, "strip %u %u %u %u %u %u", fidx[0], fidx[1], fidx[2], fidx[3], fidx[4], fidx[5]);
	GeSgxVertex *a = &fverts[0], *b = &fverts[1];
	CHECK(feq(a->w, 2) && feq(a->x / a->w, 480) && feq(a->y / a->w, 272) && feq(a->z / a->w, 1.0f), "centre %f %f %f w %f", a->x / a->w, a->y / a->w, a->z / a->w, a->w);
	CHECK(feq(b->x / b->w, 720) && feq(b->y / b->w, 136), "corner %f %f", b->x / b->w, b->y / b->w);
	CHECK(feq(a->u, 0.25f) && feq(a->v, 0.75f) && a->color == 0x80FF0002u, "uv %f %f colour %08x", a->u, a->v, a->color);
	CHECK(feq(a->fog, 0.0f) && feq(fverts[2].fog, 0.0f), "fog %f %f", a->fog, fverts[2].fog);
	CHECK(w->iaddr == 0x08910004, "iaddr advanced %08x", w->iaddr);
}

static void testLightingSkinning(void) {
	resetFrame();
	/* weight u8, normal s8, position s16: 1 + 3 (+pad to 2) + 6 = 10 */
	uint32_t v = 0x08900000;
	w8(v + 0, 128);                       // weight 1.0
	w8(v + 1, 0); w8(v + 2, 0); w8(v + 3, 127);
	w16(v + 4, 0); w16(v + 6, 0); w16(v + 8, (uint16_t)(int16_t)-16384); // z -0.5

	L l = { 0x08800000 };
	setup3D(&l);
	static const float bone[12] = { 1, 0, 0, 0, 1, 0, 0, 0, 1, 0.5f, 0, -1.5f }; // moves to (0.5, 0, -2)
	matrix(&l, 0x2A, bone, 12);
	emit(&l, C(0x17, 1));                 // lighting, light 0 directional (0, 0, 1), white diffuse
	emit(&l, C(0x18, 1));
	emit(&l, C(0x5F, 0));
	emit(&l, C(0x63, F24(0.0f)));
	emit(&l, C(0x64, F24(0.0f)));
	emit(&l, C(0x65, F24(1.0f)));
	emit(&l, C(0x90, 0xFFFFFF));
	emit(&l, C(0x56, 0xFFFFFF));          // material diffuse white
	emit(&l, C(0x58, 0xFF));
	emit(&l, C(0x5D, 0xFF));
	emit(&l, C(0x12, VT_WT8 | VT_NRM8 | VT_POS16));
	emit(&l, C(0x10, 0x080000));
	emit(&l, C(0x01, 0x900000));
	emit(&l, C(0x04, (0u << 16) | 1));    // one point
	emit(&l, C(0x0C, 0));
	geSgxWalkerList(w, 1, 0x08800000, 0, 0);
	geSgxWalkerRun(w);

	CHECK(frame.ndraws == 1 && frame.nvertices == 1 && fdraws[0].prim == GE_SGX_DRAW_POINTS, "point draws %u verts %u", frame.ndraws, frame.nvertices);
	GeSgxVertex *a = &fverts[0];
	CHECK(feq(a->w, 2) && feq(a->x / a->w, 600), "skinned %f w %f", a->x / a->w, a->w);
	uint32_t r = a->color & 0xFF, al = a->color >> 24;
	CHECK(r == 255 && ((a->color >> 8) & 0xFF) == r && al == 255, "lit colour %08x", a->color);
}

static uint8_t fraw[65536];
static uint8_t funi[65536];

static void testGpuTransform(void) {
	resetFrame();
	frame.raw = fraw;
	frame.max_raw = sizeof(fraw);
	frame.uniforms = funi;
	frame.max_uniforms = sizeof(funi);
	w->gpu_transform = 1;

	/* 8 vertices: float uv, 8888, float xyz (24 bytes) at 0x08900008 (alignment kept modulo 16) */
	uint32_t v = 0x08900008;
	int i;
	for (i = 0; i < 8; i++) {
		wf(v + i * 24, (float)i);
		wf(v + i * 24 + 4, 0.5f);
		w32(v + i * 24 + 8, 0xFF000000u | (uint32_t)i);
		wf(v + i * 24 + 12, (float)i);
		wf(v + i * 24 + 16, 1.0f);
		wf(v + i * 24 + 20, -2.0f);
	}
	L l = { 0x08800000 };
	setup3D(&l);
	emit(&l, C(0x12, (3u << 0) | VT_COL8888 | VT_POSF));
	emit(&l, C(0x10, 0x080000));
	emit(&l, C(0x01, 0x900008));
	emit(&l, C(0x04, (4u << 16) | 4));    // strip of 4: vertices 0..3
	emit(&l, C(0x04, (3u << 16) | 3));    // triangles: vertices 4..6, the same run, merged
	emit(&l, C(0x12, VT_COL565 | VT_POSF)); // 16 bit colour: decoded by the vertex program
	emit(&l, C(0x04, (3u << 16) | 3));
	/* indices scattered over a big buffer (0 and 3000): the two vertices are gathered */
	w16(0x08910000, 0); w16(0x08910002, 3000); w16(0x08910004, 0);
	emit(&l, C(0x12, VT_COL8888 | VT_POSF | VT_IDX16));
	emit(&l, C(0x01, 0x900000));
	emit(&l, C(0x02, 0x910000));
	emit(&l, C(0x04, (3u << 16) | 3));
	emit(&l, C(0x0C, 0));
	w32(0x08900000 + 3000 * 16, 0xAABBCCDD);
	geSgxWalkerList(w, 1, 0x08800000, 0, 0);
	geSgxWalkerRun(w);
	geSgxBuildFinish(w);

	CHECK(frame.prims_gpu == 4 && frame.ndraws == 3, "gpu prims %u draws %u", frame.prims_gpu, frame.ndraws);
	if (frame.ndraws == 3) {
		GeSgxDraw *d = &fdraws[0];
		CHECK(d->gpu && d->vkey == (GE_SGX_VS_BASE | GE_SGX_VS_TC | GE_SGX_VS_COLOR) && d->count == 9 && d->first == 0, "gpu draw %u key %x count %u", d->gpu, d->vkey, d->count);
		CHECK((d->base & 15) == 8 && memcmp(fraw + d->base, host(v), 7 * 24) == 0, "raw copy at %u", d->base);
		/* strip 0 1 2, 2 1 3; then 4 5 6 (vertex offset 4 in the same run) */
		CHECK(fidx[0] == 0 && fidx[3] == 2 && fidx[4] == 1 && fidx[5] == 3 && fidx[6] == 4 && fidx[8] == 6, "gpu indices %u %u %u %u", fidx[3], fidx[5], fidx[6], fidx[8]);
		const float *uc = (const float *)(funi + d->ucommon), *uo = (const float *)(funi + d->uobject);
		CHECK(feq(uo[GE_SGX_UO_WORLD * 4 + 0], 1) && feq(uo[(GE_SGX_UO_WORLD + 3) * 4 + 3], 1) && feq(uo[GE_SGX_UO_SCALE * 4], 1) && feq(uo[GE_SGX_UO_UV * 4], 1),
		      "object block world %f scale %f uv %f", uo[0], uo[GE_SGX_UO_SCALE * 4], uo[GE_SGX_UO_UV * 4]);
		CHECK(feq(uc[(GE_SGX_UC_PROJ + 2) * 4 + 3], -1) && feq(uc[GE_SGX_UC_VP * 4], 480) && feq(uc[GE_SGX_UC_VP * 4 + 1], 480) && feq(uc[GE_SGX_UC_LIGHT * 4 + 3], -1),
		      "common block proj %f vp %f %f light %f", uc[(GE_SGX_UC_PROJ + 2) * 4 + 3], uc[GE_SGX_UC_VP * 4], uc[GE_SGX_UC_VP * 4 + 1], uc[GE_SGX_UC_LIGHT * 4 + 3]);
		CHECK(fdraws[1].gpu && fdraws[1].vkey == (GE_SGX_VS_BASE | GE_SGX_VS_COLOR16) && frame.nvertices == 0, "16 bit colour %u key %x verts %u", fdraws[1].gpu, fdraws[1].vkey, frame.nvertices);
		const float *uc1 = (const float *)(funi + fdraws[1].ucommon);
		CHECK(feq(uc1[GE_SGX_UC_MATUPD * 4 + 1], 0), "565 format %f", uc1[GE_SGX_UC_MATUPD * 4 + 1]);
		/* gathered: 2 vertices of 16 bytes, indices 0 1 0 */
		GeSgxDraw *g = &fdraws[2];
		uint32_t c3000 = 0;
		memcpy(&c3000, fraw + g->base + 16, 4);
		CHECK(g->gpu && fidx[g->first] == 0 && fidx[g->first + 1] == 1 && fidx[g->first + 2] == 0 && c3000 == 0xAABBCCDDu, "gather %u: %u %u %u colour %08x",
		      g->gpu, fidx[g->first], fidx[g->first + 1], fidx[g->first + 2], c3000);
	}
	w->gpu_transform = 0;
}

/* With PSP RAM mapped for the GPU, vertices are used where they are */
static void testGpuDirect(void) {
	resetFrame();
	frame.raw = fraw;
	frame.max_raw = sizeof(fraw);
	frame.uniforms = funi;
	frame.max_uniforms = sizeof(funi);
	frame.direct = ram;
	w->gpu_transform = 1;
	w->direct_start = RAM_BASE;
	w->direct_size = RAM_SIZE;

	L l = { 0x08800000 };
	setup3D(&l);
	emit(&l, C(0x12, (3u << 0) | VT_COL8888 | VT_POSF));
	emit(&l, C(0x10, 0x080000));
	emit(&l, C(0x01, 0x900000));
	emit(&l, C(0x04, (4u << 16) | 4));    // strip 0..3 at 0x08900000
	emit(&l, C(0x04, (3u << 16) | 3));    // 4..6 right after: same draw
	emit(&l, C(0x01, 0x900002));           // not 4 byte aligned: copied
	emit(&l, C(0x04, (3u << 16) | 3));
	emit(&l, C(0x0C, 0));
	geSgxWalkerList(w, 1, 0x08800000, 0, 0);
	geSgxWalkerRun(w);
	geSgxBuildFinish(w);

	CHECK(frame.ndraws == 2 && fdraws[0].direct && fdraws[0].base == 0x900000 - (RAM_BASE & 0xFFFFFF) + (0x08000000 - RAM_BASE) && fdraws[0].count == 9 && fidx[6] == 4,
	      "direct draws %u direct %u base %X count %u idx %u", frame.ndraws, fdraws[0].direct, fdraws[0].base, fdraws[0].count, fidx[6]);
	CHECK(frame.ndraws == 2 && !fdraws[1].direct && fdraws[1].gpu && frame.nraw > 0, "unaligned copied: direct %u raw %u", fdraws[1].direct, frame.nraw);
	w->gpu_transform = 0;
	w->direct_size = 0;
}

/* Framebuffers as textures, stencil, colour mask, stencil clear */
static void testFbTexture(void) {
	resetFrame();
	uint32_t v = 0x08900000;
	spriteVertex(v + 0, 0, 0, 0x80FFFFFF, 0, 0);
	spriteVertex(v + 16, 480, 272, 0x80FFFFFF, 480, 272);

	L l = { 0x08800000 };
	/* draw into 0x88000 (a render target, 256 x 128) */
	emit(&l, C(0x9C, 0x88000));
	emit(&l, C(0x9D, 256));
	emit(&l, C(0xD4, 0));
	emit(&l, C(0xD5, (127 << 10) | 255));
	emit(&l, C(0xD3, 0x201));              // clear: stencil (alpha) only
	emit(&l, C(0x12, VT_TC16 | VT_COL8888 | VT_POS16 | VT_THROUGH));
	emit(&l, C(0x10, 0x080000));
	emit(&l, C(0x01, 0x900000));
	emit(&l, C(0x04, (6u << 16) | 2));
	emit(&l, C(0xD3, 0));
	/* then onto the screen with that buffer as texture (512 x 256 texture: half of it) */
	emit(&l, C(0x9C, 0x00000));
	emit(&l, C(0x9D, 512));
	emit(&l, C(0xD5, (271 << 10) | 479));
	emit(&l, C(0x1E, 1));
	emit(&l, C(0xA0, 0x088000));
	emit(&l, C(0xA8, (0x04 << 16) | 256));
	emit(&l, C(0xB8, (8 << 8) | 9));       // 512 x 256
	emit(&l, C(0xC3, GE_TFMT_5650));
	emit(&l, C(0x24, 1));                  // stencil: less, ref 0x12, mask 0xFF, replace on pass
	emit(&l, C(0xDC, (0xFF << 16) | (0x12 << 8) | 4));
	emit(&l, C(0xDD, 2 << 16));
	emit(&l, C(0xE9, 0x0F));               // stencil write mask 0xF0
	emit(&l, C(0xE8, 0x0000FF));           // red not written
	emit(&l, C(0x01, 0x900000));
	emit(&l, C(0x04, (6u << 16) | 2));
	emit(&l, C(0x0C, 0));
	tex_calls = 0;
	geSgxWalkerList(w, 1, 0x08800000, 0, 0);
	geSgxWalkerRun(w);

	CHECK(frame.ndraws == 2 && frame.nfbs == 2, "draws %u fbs %u", frame.ndraws, frame.nfbs);
	if (frame.ndraws == 2) {
		GeSgxDraw *c = &fdraws[0], *d = &fdraws[1];
		CHECK(c->fb == 0x88000 && (c->flags & GE_SGX_DRAW_CLEAR) && c->sfunc == 1 && c->sref == 0x80 && c->sops == 0x222, "stencil clear fb %X func %u ref %X ops %X", c->fb, c->sfunc, c->sref, c->sops);
		CHECK(d->fbtex == 0x88000 + 1 && tex_calls == 0 && d->sfunc == 4 && d->sref == 0x12 && d->smask == 0xFF && d->sops == 0x200 && d->swmask == 0xF0 && d->cmask == 1,
		      "fb texture %X lookups %d stencil %u %X %X ops %X wmask %X cmask %u", d->fbtex, tex_calls, d->sfunc, d->sref, d->smask, d->sops, d->swmask, d->cmask);
		/* texels (480, 272) of a 256 x 128 buffer: past its size, u = 480 / 256 */
		GeSgxVertex *cv = &fverts[d->base + 3];
		CHECK(feq(cv->u, 480.0f / 256.0f) && feq(cv->v, 272.0f / 128.0f), "fb texture uv %f %f", cv->u, cv->v);
	}
	int k, found = 0;
	for (k = 0; k < (int)frame.nfbs; k++) {
		if (frame.fbs[k].fb == 0x88000 && frame.fbs[k].w == 256 && frame.fbs[k].h == 128)
			found = 1;
	}
	CHECK(found, "fb 0x88000 256x128 listed");
}

/* A framebuffer as texture with generated coordinates (texture matrix, environment map): scaled to the texture's part too */
static void testFbTextureGenerated(void) {
	resetFrame();
	uint32_t v = 0x08900000;
	spriteVertex(v + 0, 0, 0, 0x80FFFFFF, 0, 0);
	spriteVertex(v + 16, 256, 128, 0x80FFFFFF, 0, 0);
	/* float uv, float xyz: 20 bytes */
	static const float pos[3][3] = { { 0, 0, -2 }, { 1, 1, -2 }, { -1, 1, -4 } };
	int i;
	for (i = 0; i < 3; i++) {
		wf(0x08901000 + i * 20, 0.25f);
		wf(0x08901000 + i * 20 + 4, 0.75f);
		wf(0x08901000 + i * 20 + 8, pos[i][0]);
		wf(0x08901000 + i * 20 + 12, pos[i][1]);
		wf(0x08901000 + i * 20 + 16, pos[i][2]);
	}
	static const float ident[12] = { 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0 };

	L l = { 0x08800000 };
	/* draw into 0x88000 (256 x 128) */
	emit(&l, C(0x9C, 0x88000));
	emit(&l, C(0x9D, 256));
	emit(&l, C(0xD4, 0));
	emit(&l, C(0xD5, (127 << 10) | 255));
	emit(&l, C(0x12, VT_TC16 | VT_COL8888 | VT_POS16 | VT_THROUGH));
	emit(&l, C(0x10, 0x080000));
	emit(&l, C(0x01, 0x900000));
	emit(&l, C(0x04, (6u << 16) | 2));
	/* then a 3D triangle on the screen, that buffer as a 512 x 256 texture, uv from the texture matrix */
	emit(&l, C(0x9C, 0x00000));
	emit(&l, C(0x9D, 512));
	emit(&l, C(0xD5, (271 << 10) | 479));
	setup3D(&l);
	matrix(&l, 0x40, ident, 12);
	emit(&l, C(0xC0, (1u << 8) | 1));      // texture matrix of the uv
	emit(&l, C(0x1E, 1));
	emit(&l, C(0xA0, 0x088000));
	emit(&l, C(0xA8, (0x04 << 16) | 256));
	emit(&l, C(0xB8, (8 << 8) | 9));       // 512 x 256
	emit(&l, C(0xC3, GE_TFMT_5650));
	emit(&l, C(0x12, (3u << 0) | VT_POSF));
	emit(&l, C(0x01, 0x901000));
	emit(&l, C(0x04, (3u << 16) | 3));
	emit(&l, C(0x0C, 0));
	geSgxWalkerList(w, 1, 0x08800000, 0, 0);
	geSgxWalkerRun(w);

	CHECK(frame.ndraws == 2, "draws %u", frame.ndraws);
	if (frame.ndraws == 2) {
		GeSgxDraw *d = &fdraws[1];
		GeSgxVertex *cv = &fverts[d->base];
		/* (0.25, 0.75) of a 512 x 256 texture: texels (128, 192) of a 256 x 128 buffer */
		CHECK(d->fbtex == 0x88000 + 1 && feq(cv->u, 0.5f) && feq(cv->v, 1.5f), "fb texture %X generated uv %f %f", d->fbtex, cv->u, cv->v);
	}
}

static void testTransferAlias(void) {
	resetFrame();
	L l = { 0x08800000 };
	emit(&l, C(0xB2, 0x088000));
	emit(&l, C(0xB3, 512));
	emit(&l, C(0xB4, 0x000000));
	emit(&l, C(0xB5, 512));
	emit(&l, C(0xEE, (271 << 10) | 479));
	emit(&l, C(0xEA, 1));
	emit(&l, C(0xEE, (15 << 10) | 15));   // small copy: not an alias
	emit(&l, C(0xEA, 1));
	emit(&l, C(0x0C, 0));
	/* VRAM addresses: upper bits 0x04 */
	w32(0x08800004, C(0xB3, (0x04 << 16) | 512));
	w32(0x0880000C, C(0xB5, (0x04 << 16) | 512));
	geSgxWalkerList(w, 1, 0x08800000, 0, 0);
	geSgxWalkerRun(w);
	CHECK(w->naliases == 1 && w->aliases[0].src == 0x88000 && w->aliases[0].dst == 0, "aliases %d src %x dst %x", w->naliases, w->naliases ? w->aliases[0].src : 0, w->naliases ? w->aliases[0].dst : 0);
}

/* ---- texture conversion ---- */

static void swizzle(uint8_t *dst, const uint8_t *src, uint32_t rowbytes, uint32_t rows) {
	uint32_t bxc = rowbytes / 16, by, bx, line;
	for (by = 0; by < rows / 8; by++)
		for (bx = 0; bx < bxc; bx++)
			for (line = 0; line < 8; line++)
				memcpy(dst + (by * bxc + bx) * 128 + line * 16, src + (by * 8 + line) * rowbytes + bx * 16, 16);
}

static uint32_t rgbaT(uint32_t r, uint32_t g, uint32_t b, uint32_t a) {
	return r | (g << 8) | (b << 16) | (a << 24);
}

static void testIndexed(void) {
	GeSgxTexDesc d;
	memset(&d, 0, sizeof(d));
	d.format = GE_TFMT_CLUT4; d.w = 4; d.h = 2; d.bufw = 4;
	uint8_t src[4] = { 0x21, 0x43, 0x65, 0x87 }; /* row 0: 1 2 3 4, row 1: 5 6 7 8 */
	uint8_t out[16];
	CHECK(geSgxTexIndexed(GE_TFMT_CLUT4) && geSgxTexIndexed(GE_TFMT_CLUT8) && !geSgxTexIndexed(GE_TFMT_CLUT16), "indexed formats");
	CHECK(geSgxTexConvertIndices(&d, src, 2, out, 8) && out[0] == 1 && out[1] == 2 && out[3] == 4 && out[8] == 5 && out[11] == 8, "clut4 indices %u %u %u %u", out[0], out[1], out[8], out[11]);
	static uint32_t clut[512], pal[256];
	int i;
	for (i = 0; i < 512; i++)
		clut[i] = 0xFF000000u | (uint32_t)i;
	/* shift 1, mask 0x0F, start 2 (entries 32..): index i -> ((i >> 1) & 15) | 32 */
	d.format = GE_TFMT_CLUT8;
	d.clut_format = 3 | (1u << 2) | (0x0Fu << 8) | (2u << 16);
	geSgxTexPalette(&d, clut, pal);
	CHECK(pal[0] == (0xFF000000u | 32) && pal[5] == (0xFF000000u | 34) && pal[255] == (0xFF000000u | 47), "palette %08x %08x %08x", pal[0], pal[5], pal[255]);
}

static void testTexconv(void) {
	static uint8_t lin[64 * 16 * 4], swz[64 * 16 * 4], back[64 * 16 * 4], out[64 * 16 * 4];
	uint32_t i;
	for (i = 0; i < sizeof(lin); i++)
		lin[i] = (uint8_t)(i * 7 + (i >> 8));
	swizzle(swz, lin, 64 * 4, 16);
	geSgxUnswizzle(back, swz, 64 * 4, 16);
	CHECK(memcmp(back, lin, sizeof(lin)) == 0, "unswizzle round trip");

	CHECK(geSgxTexRowBytes(GE_TFMT_CLUT4, 32) == 16 && geSgxTexRowBytes(GE_TFMT_8888, 32) == 128 && geSgxTexRowBytes(GE_TFMT_DXT1, 32) == 64 &&
	      geSgxTexRowBytes(GE_TFMT_DXT5, 32) == 128 && geSgxTexRowBytes(11, 32) == 0, "row bytes");
	{
		/* DXT1 8x4: two blocks. Block 0: c1 = pure red 0xF800 > c2 = pure blue 0x001F, indices row 0 = 0,1,2,3 */
		uint8_t dxt[16] = { 0xE4, 0x00, 0x55, 0xFF, 0x00, 0xF8, 0x1F, 0x00,
		                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 }; /* block 1: c1 == c2 == 0, index 0 -> black opaque */
		static uint32_t px[8 * 4];
		GeSgxTexDesc dd;
		memset(&dd, 0, sizeof(dd));
		dd.format = GE_TFMT_DXT1; dd.w = 8; dd.h = 4; dd.bufw = 8;
		CHECK(geSgxTexRows(&dd) == 1, "dxt rows %u", geSgxTexRows(&dd));
		CHECK(geSgxTexConvert(&dd, dxt, 16, (uint8_t *)px, 8 * 4, NULL), "dxt1 convert");
		/* blue in the low bits of the PSP 565: 0xF800 is red -> 0xFF0000F8 */
		CHECK(px[0] == 0xFF0000F8u && px[1] == 0xFFF80000u, "dxt1 colours %08x %08x", px[0], px[1]);
		CHECK(px[2] == rgbaT(0xA5, 0, 0x52, 0xFF) && px[3] == rgbaT(0x52, 0, 0xA5, 0xFF), "dxt1 mix %08x %08x", px[2], px[3]);
		/* rows 1..3: indices all 0, all 1, all 3; block 1 (c1 == c2 == 0): opaque black */
		CHECK(px[8] == 0xFF0000F8u && px[16] == 0xFFF80000u && px[24] == px[3] && px[4] == 0xFF000000u && px[15] == 0xFF000000u,
		      "dxt1 rows %08x %08x %08x %08x", px[8], px[16], px[24], px[4]);
		/* DXT5: alpha 255 / 0 with 6 interpolated, index bits 3 per texel; DXT3: 4 bit alpha */
		uint8_t d5[16] = { 0, 0, 0, 0, 0xFF, 0xFF, 0, 0, 0x08, 0, 0, 0, 0, 0, 0xFF, 0x00 }; /* texel 0 idx 0, texel 1 idx 1 */
		dd.format = GE_TFMT_DXT5; dd.w = 4;
		CHECK(geSgxTexConvert(&dd, d5, 16, (uint8_t *)px, 4 * 4, NULL) && (px[0] >> 24) == 0xFF && (px[1] >> 24) == 0 && (px[0] & 0xFFFFFF) == 0xF8FCF8, "dxt5 %08x %08x", px[0], px[1]);
		uint8_t d3[16] = { 0, 0, 0, 0, 0xFF, 0xFF, 0, 0, 0x21, 0x00, 0, 0, 0, 0, 0, 0xF0 };
		dd.format = GE_TFMT_DXT3;
		CHECK(geSgxTexConvert(&dd, d3, 16, (uint8_t *)px, 4 * 4, NULL) && (px[0] >> 24) == 0x11 && (px[1] >> 24) == 0x22 && (px[15] >> 24) == 0xFF, "dxt3 %08x %08x %08x", px[0], px[1], px[15]);
	}

	GeSgxTexDesc d;
	memset(&d, 0, sizeof(d));
	d.w = 4; d.h = 2; d.bufw = 8; d.format = GE_TFMT_5650;
	CHECK(geSgxTexOutStride(&d) == 16, "stride %u", geSgxTexOutStride(&d));
	memset(out, 0xAA, sizeof(out));
	CHECK(geSgxTexConvert(&d, lin, 16, out, 16, NULL) && memcmp(out, lin, 8) == 0 && out[8] == 0 && memcmp(out + 16, lin + 16, 8) == 0, "16 bit rows copied, padding cleared");

	/* CLUT: 5551 entries, CLUT8 with shift 1, mask 0x0F, start 1 (base 16) */
	uint8_t clutraw[64];
	for (i = 0; i < 32; i++) {
		uint16_t c = (uint16_t)(0x8000 | i); // red = i
		clutraw[i * 2] = (uint8_t)c;
		clutraw[i * 2 + 1] = (uint8_t)(c >> 8);
	}
	static uint32_t clut[GE_SGX_CLUT_ENTRIES];
	geSgxClutDecode(1, clutraw, 64, clut);
	CHECK(clut[3] == geSgxDecodeColor16(1, 0x8003) && (clut[3] >> 24) == 0xFF && clut[40] == 0, "clut decode %08x", clut[3]);

	uint8_t idx[8] = { 0, 2, 4, 6, 0x1E, 0x20, 0xFF, 0x08 };
	memset(&d, 0, sizeof(d));
	d.w = 8; d.h = 1; d.bufw = 8; d.format = GE_TFMT_CLUT8;
	d.clut_format = 1 | (1 << 2) | (0x0F << 8) | (1 << 16);
	CHECK(geSgxTexConvert(&d, idx, 8, out, 32, clut), "clut8 convert");
	uint32_t o[8];
	memcpy(o, out, 32);
	/* (raw >> 1) & 15 | 16 */
	CHECK(o[0] == clut[16] && o[1] == clut[17] && o[4] == clut[31] && o[5] == clut[16] && o[6] == clut[31] && o[7] == clut[20], "clut8 index %08x %08x", o[0], o[7]);

	/* CLUT4: low nibble first */
	uint8_t nib[2] = { 0x21, 0x43 };
	memset(&d, 0, sizeof(d));
	d.w = 4; d.h = 1; d.bufw = 4; d.format = GE_TFMT_CLUT4;
	d.clut_format = 1 | (0xFF << 8);
	CHECK(geSgxTexConvert(&d, nib, 2, out, 32, clut), "clut4 convert");
	memcpy(o, out, 16);
	CHECK(o[0] == clut[1] && o[1] == clut[2] && o[2] == clut[3] && o[3] == clut[4], "clut4 order %08x %08x", o[0], o[1]);

	CHECK(geSgxDecodeColor16(0, 0x07E0) == 0xFF00FF00u && geSgxDecodeColor16(2, 0x8F00) == 0x88FF0000u, "colour decode %08x %08x", geSgxDecodeColor16(0, 0x07E0), geSgxDecodeColor16(2, 0x8F00));
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
	testBuildSprites();
	testBuildOther();
	testBuildClearAndClut();
	testIndexed();
	testBuild3D();
	testLightingSkinning();
	testGpuTransform();
	testGpuDirect();
	testFbTexture();
	testFbTextureGenerated();
	testTransferAlias();
	testTexconv();

	printf("ge_sgx_test: %d checks, %d failures (page fetches %d)\n", checks, failures, page_calls);
	free(ram);
	free(vram);
	return failures ? 1 : 0;
}
