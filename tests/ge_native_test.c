/*
	Host unit tests for the native resolution GE list patcher
	(cef/systemctrl/ge_native_list.c). PSP RAM and VRAM are emulated with host
	buffers; run with: make -C tests
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ge_native.h"

#ifdef GE_NATIVE_TRACE
#include <stdarg.h>
static char trace_out[1 << 20];
static size_t trace_n;
void geNativeTrace(const char *fmt, ...) {
	va_list ap;
	va_start(ap, fmt);
	trace_n += vsnprintf(trace_out + trace_n, sizeof(trace_out) - trace_n, fmt, ap);
	va_end(ap);
}
#endif

#define RAM_BASE  0x08000000u
#define RAM_SIZE  0x04000000u
#define VRAM_BASE 0x04000000u
#define VRAM_SIZE 0x00200000u

static u8 *ram, *vram;
static int failures = 0, checks = 0;

void *ge_host_ptr(u32 addr) {
	addr &= 0x0FFFFFFF;
	if (addr >= RAM_BASE && addr < RAM_BASE + RAM_SIZE) return ram + (addr - RAM_BASE);
	if (addr >= VRAM_BASE && addr < VRAM_BASE + VRAM_SIZE) return vram + (addr - VRAM_BASE);
	fprintf(stderr, "FATAL: patcher dereferenced unmapped address %08x\n", addr);
	abort();
}

#define CHECK(cond, ...) do { checks++; if (!(cond)) { failures++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static u32 *W(u32 addr) { return (u32 *)ge_host_ptr(addr); }
static s16 *H(u32 addr) { return (s16 *)ge_host_ptr(addr); }
static float *F(u32 addr) { return (float *)ge_host_ptr(addr); }

static u32 f24(float f) { union { float f; u32 i; } t; t.f = f; return t.i >> 8; }
static float unf24(u32 d) { union { float f; u32 i; } t; t.i = d << 8; return t.f; }
static u32 C(u32 cmd, u32 data) { return (cmd << 24) | (data & 0xFFFFFF); }
static u32 data_of(u32 addr) { return *W(addr) & 0xFFFFFF; }

/* Simple list writer */
typedef struct { u32 start, pos; } L;
static void emit(L *l, u32 op) { *W(l->pos) = op; l->pos += 4; }
static u32 at(L *l) { return l->pos; }

/* Standard sceGu setup for a 480x272 screen. Returns addresses of commands. */
typedef struct { u32 fbp, fbw, fmt, zbp, zbw, vxs, vys, vxc, vyc, ofx, ofy, sc1, sc2, rg1, rg2; } Setup;
static Setup emitSetup(L *l, u32 fbp, u32 pitch, float sx, float sy, float cx, float cy, u32 ox, u32 oy) {
	Setup s;
	s.fmt = at(l); emit(l, C(GE_CMD_FRAMEBUFPIXFORMAT, 3));
	s.fbp = at(l); emit(l, C(GE_CMD_FRAMEBUFPTR, fbp & 0xFFFFFF));
	s.fbw = at(l); emit(l, C(GE_CMD_FRAMEBUFWIDTH, ((fbp >> 24) << 16) | pitch));
	s.zbp = at(l); emit(l, C(GE_CMD_ZBUFPTR, 0x110000));
	s.zbw = at(l); emit(l, C(GE_CMD_ZBUFWIDTH, 512));
	s.vxs = at(l); emit(l, C(GE_CMD_VIEWPORTXSCALE, f24(sx)));
	s.vys = at(l); emit(l, C(GE_CMD_VIEWPORTYSCALE, f24(sy)));
	s.vxc = at(l); emit(l, C(GE_CMD_VIEWPORTXCENTER, f24(cx)));
	s.vyc = at(l); emit(l, C(GE_CMD_VIEWPORTYCENTER, f24(cy)));
	s.ofx = at(l); emit(l, C(GE_CMD_OFFSETX, ox << 4));
	s.ofy = at(l); emit(l, C(GE_CMD_OFFSETY, oy << 4));
	s.sc1 = at(l); emit(l, C(GE_CMD_SCISSOR1, 0));
	s.sc2 = at(l); emit(l, C(GE_CMD_SCISSOR2, (271 << 10) | 479));
	s.rg1 = at(l); emit(l, C(GE_CMD_REGION1, 0));
	s.rg2 = at(l); emit(l, C(GE_CMD_REGION2, (271 << 10) | 479));
	return s;
}

static void emitVaddr(L *l, u32 addr) {
	emit(l, C(GE_CMD_BASE, (addr >> 8) & 0x0F0000));
	emit(l, C(GE_CMD_VADDR, addr & 0xFFFFFF));
}

static void emitFinish(L *l) {
	emit(l, C(GE_CMD_FINISH, 0));
	emit(l, C(GE_CMD_END, 0));
}

/* through mode, 16-bit position + 16-bit texcoord: vertex = u,v,x,y,z (10 bytes, no padding) */
#define VT_THROUGH_T16_P16 (GE_VTYPE_THROUGH | (2 << GE_VTYPE_TC_SHIFT) | (2 << GE_VTYPE_POS_SHIFT))
/* through mode, 8888 colour + 16-bit position: c(4) x y z (10) -> aligned to 12 with 2 padding bytes */
#define VT_THROUGH_C8888_P16 (GE_VTYPE_THROUGH | (7 << GE_VTYPE_COL_SHIFT) | (2 << GE_VTYPE_POS_SHIFT))
/* through mode float: 8888 colour + float pos = 4 + 12 = 16, no padding */
#define VT_THROUGH_C8888_PF (GE_VTYPE_THROUGH | (7 << GE_VTYPE_COL_SHIFT) | (3 << GE_VTYPE_POS_SHIFT))
/* through mode: 8-bit tc (2) + 16-bit pos: tc at 0, pos aligned to 2 -> 2+6 = 8 no padding... use 565 colour + float pos */
#define VT_THROUGH_C565_PF (GE_VTYPE_THROUGH | (4 << GE_VTYPE_COL_SHIFT) | (3 << GE_VTYPE_POS_SHIFT))

static void reset_mem(void) {
	memset(ram, 0, RAM_SIZE);
	memset(vram, 0, VRAM_SIZE);
	geNativeListInit();
}

/* --------------------------------------------------------------------- */

static void test_basic_fullscreen(void) {
	reset_mem();
	L l = { 0x08900000, 0x08900000 };
	Setup s = emitSetup(&l, 0x000000, 512, 240, -136, 2048, 2048, 2048 - 240, 2048 - 136);

	/* sprite vertices, colour 8888 + s16 pos, 12-byte stride with padding at 10 */
	u32 vb = 0x08A00000;
	s16 pts[4][3] = { { 0, 0, 0 }, { 480, 272, 0 }, { 100, 50, 0 }, { 200, 150, 0 } };
	for (int i = 0; i < 4; i++) {
		*W(vb + i * 12) = 0xFFFFFFFF;
		for (int k = 0; k < 3; k++) *H(vb + i * 12 + 4 + k * 2) = pts[i][k];
	}
	emit(&l, C(GE_CMD_VERTEXTYPE, VT_THROUGH_C8888_P16));
	emitVaddr(&l, vb);
	u32 prim = at(&l); emit(&l, C(GE_CMD_PRIM, (6 << 16) | 4));
	emitFinish(&l);

	geNativeListReset();
	geNativeListPatch(l.start, 0);

	CHECK(data_of(s.fmt) == GE_FORMAT_565, "pixel format -> 565");
	CHECK(data_of(s.fbp) == 0, "fbp -> vram 0");
	CHECK(data_of(s.fbw) == ((0x04 << 16) | 960), "fbw -> 960 (%06x)", data_of(s.fbw));
	CHECK(data_of(s.zbp) == 0x100000, "zbp");
	CHECK(data_of(s.zbw) == ((0x04 << 16) | 960), "zbw");
	CHECK(unf24(data_of(s.vxs)) == 480, "vx scale %f", unf24(data_of(s.vxs)));
	CHECK(unf24(data_of(s.vys)) == -272, "vy scale %f", unf24(data_of(s.vys)));
	CHECK(unf24(data_of(s.vxc)) == 2048, "vx center");
	CHECK(unf24(data_of(s.vyc)) == 2048, "vy center");
	CHECK(data_of(s.ofx) == (u32)((2048 - 480) << 4), "offset x %x", data_of(s.ofx));
	CHECK(data_of(s.ofy) == (u32)((2048 - 272) << 4), "offset y %x", data_of(s.ofy));
	CHECK(data_of(s.sc2) == ((543 << 10) | 959), "scissor end");
	CHECK(data_of(s.rg2) == ((543 << 10) | 959), "region end");
	CHECK(data_of(s.sc1) == 0 && data_of(s.rg1) == 0, "start corners");
	CHECK((*W(prim) >> 24) == GE_CMD_PRIM, "prim kept");

	s16 exp[4][2] = { { 0, 0 }, { 960, 544 }, { 200, 100 }, { 400, 300 } };
	for (int i = 0; i < 4; i++) {
		CHECK(*H(vb + i * 12 + 4) == exp[i][0] && *H(vb + i * 12 + 6) == exp[i][1],
		      "vertex %d = %d,%d", i, *H(vb + i * 12 + 4), *H(vb + i * 12 + 6));
	}
	CHECK(geNativeListHasDraws(), "has draws");
	CHECK(geNativeListTakeFrameDraws() == 1, "frame draws");
	CHECK(geNativeListTakeFrameDraws() == 0, "frame draws cleared");

	/* Same list executed again (static display list): nothing may be scaled twice */
	geNativeListReset();
	geNativeListPatch(l.start, 0);
	CHECK(unf24(data_of(s.vxs)) == 480, "2nd pass vx scale %f", unf24(data_of(s.vxs)));
	CHECK(data_of(s.ofx) == (u32)((2048 - 480) << 4), "2nd pass offset x");
	CHECK(data_of(s.sc2) == ((543 << 10) | 959), "2nd pass scissor");
	CHECK(data_of(s.fbw) == ((0x04 << 16) | 960), "2nd pass fbw");
	for (int i = 0; i < 4; i++)
		CHECK(*H(vb + i * 12 + 4) == exp[i][0] && *H(vb + i * 12 + 6) == exp[i][1], "2nd pass vertex %d", i);

	/* Game rebuilds the list in place with its own values: scaled once again */
	l.pos = l.start;
	emitSetup(&l, 0x088000, 512, 240, -136, 2048, 2048, 2048 - 240, 2048 - 136);
	geNativeListReset();
	geNativeListPatch(l.start, 0);
	CHECK(unf24(data_of(s.vxs)) == 480, "rebuilt vx scale");
	CHECK(data_of(s.ofy) == (u32)((2048 - 272) << 4), "rebuilt offset y");
}

/* The screen mapping must be exactly doubled for any viewport */
static void test_subviewport(void) {
	reset_mem();
	L l = { 0x08900000, 0x08900000 };
	/* Top-right quarter viewport: 240x136 at (240,0) */
	float sx = 120, sy = -68, cx = 2048 + 120, cy = 2048 - 68;
	u32 ox = 2048 - 240, oy = 2048 - 136;
	Setup s = emitSetup(&l, 0, 512, sx, sy, cx, cy, ox, oy);
	*W(s.sc1) = C(GE_CMD_SCISSOR1, (0 << 10) | 240);
	*W(s.sc2) = C(GE_CMD_SCISSOR2, (135 << 10) | 479);
	emitFinish(&l);

	geNativeListReset();
	geNativeListPatch(l.start, 0);

	float nsx = unf24(data_of(s.vxs)), nsy = unf24(data_of(s.vys));
	float ncx = unf24(data_of(s.vxc)), ncy = unf24(data_of(s.vyc));
	float nox = (data_of(s.ofx) & 0xFFFF) / 16.0f, noy = (data_of(s.ofy) & 0xFFFF) / 16.0f;

	float worst = 0;
	for (float v = -1; v <= 1.0001f; v += 0.25f) {
		float old_x = v * sx + cx - ox, old_y = v * sy + cy - oy;
		float new_x = v * nsx + ncx - nox, new_y = v * nsy + ncy - noy;
		worst = fmaxf(worst, fabsf(new_x - 2 * old_x));
		worst = fmaxf(worst, fabsf(new_y - 2 * old_y));
	}
	CHECK(worst < 0.01f, "sub-viewport mapping doubled (max error %f)", worst);
	CHECK(data_of(s.sc1) == ((0 << 10) | 480), "scissor start doubled %x", data_of(s.sc1));
	CHECK(data_of(s.sc2) == ((271 << 10) | 959), "scissor end doubled %x", data_of(s.sc2));
}

/* Draws into a non-display render target are disabled; draws sampling
   another framebuffer are disabled and re-enabled on a later pass. */
static void test_offscreen_and_restore(void) {
	reset_mem();

	/* Shared sub-list with one draw, called from two parent lists */
	u32 sub = 0x08920000;
	L ls = { sub, sub };
	u32 vb = 0x08A00000;
	for (int i = 0; i < 2; i++) { *H(vb + i * 10 + 4) = 10 * (i + 1); *H(vb + i * 10 + 6) = 5 * (i + 1); }
	emit(&ls, C(GE_CMD_TEXTUREMAPENABLE, 1));
	emit(&ls, C(GE_CMD_VERTEXTYPE, VT_THROUGH_T16_P16));
	emitVaddr(&ls, vb);
	u32 prim = at(&ls); emit(&ls, C(GE_CMD_PRIM, (6 << 16) | 2));
	emit(&ls, C(GE_CMD_RET, 0));
	u32 prim_op = *W(prim);

	/* Parent 1: render to display fb, but texture = an offscreen framebuffer */
	L l1 = { 0x08900000, 0x08900000 };
	/* offscreen target 0x154000 pitch 256 */
	emit(&l1, C(GE_CMD_FRAMEBUFPTR, 0x154000));
	u32 off_fbw = at(&l1); emit(&l1, C(GE_CMD_FRAMEBUFWIDTH, 256));
	u32 off_prim = at(&l1); emit(&l1, C(GE_CMD_PRIM, (6 << 16) | 0));
	emit(&l1, C(GE_CMD_FRAMEBUFPTR, 0x000000));
	emit(&l1, C(GE_CMD_FRAMEBUFWIDTH, 512));
	emit(&l1, C(GE_CMD_TEXADDR0, 0x154000));
	emit(&l1, C(GE_CMD_TEXBUFWIDTH0, (0x0A << 16) | 256)); /* fake vram 0x0A154000 */
	emit(&l1, C(GE_CMD_BASE, (sub >> 8) & 0x0F0000));
	emit(&l1, C(GE_CMD_CALL, sub & 0xFFFFFF));
	emitFinish(&l1);

	geNativeListReset();
	geNativeListPatch(l1.start, 0);
	CHECK(*W(off_prim) == ((GE_CMD_NOP << 24) | GE_NATIVE_NOP_MARKER), "offscreen draw disabled (%08x)", *W(off_prim));
	CHECK(data_of(off_fbw) == ((0x04 << 16) | 0), "offscreen target redirected to scratch (%06x)", data_of(off_fbw));
	CHECK(*W(prim) == ((GE_CMD_NOP << 24) | GE_NATIVE_NOP_MARKER), "draw sampling offscreen fb disabled");
	CHECK(*H(vb + 4) == 10, "vertices of disabled draw untouched");

	/* Parent 2: same sub-list with a normal texture -> draw must come back */
	L l2 = { 0x08910000, 0x08910000 };
	emit(&l2, C(GE_CMD_FRAMEBUFPTR, 0x000000));
	emit(&l2, C(GE_CMD_FRAMEBUFWIDTH, 512));
	emit(&l2, C(GE_CMD_TEXADDR0, 0x200000));
	emit(&l2, C(GE_CMD_TEXBUFWIDTH0, (0x09 << 16) | 64)); /* texture in RAM 0x09200000 */
	emit(&l2, C(GE_CMD_BASE, (sub >> 8) & 0x0F0000));
	emit(&l2, C(GE_CMD_CALL, sub & 0xFFFFFF));
	emitFinish(&l2);

	geNativeListTakeFrameDraws();
	geNativeListReset();
	geNativeListPatch(l2.start, 0);
	CHECK(*W(prim) == prim_op, "draw restored on later pass (%08x)", *W(prim));
	CHECK(*H(vb + 4) == 20 && *H(vb + 6) == 10, "restored draw vertices scaled (%d,%d)", *H(vb + 4), *H(vb + 6));
	CHECK(geNativeListTakeFrameDraws() == 1, "restored draw counted");

	/* A genuine NOP written by the game where our marker was must stay a NOP */
	*W(prim) = 0;
	geNativeListReset();
	geNativeListPatch(l2.start, 0);
	CHECK(*W(prim) == 0, "genuine NOP left alone");
}

static void test_control_flow(void) {
	reset_mem();
	/* JUMP + CALL with OFFSETADDR, nested call/ret, signal jump */
	L l = { 0x08900000, 0x08900000 };
	u32 target = 0x08940000;
	emit(&l, C(GE_CMD_OFFSETADDR, 0x000000));
	emit(&l, C(GE_CMD_BASE, (target >> 8) & 0x0F0000));
	emit(&l, C(GE_CMD_JUMP, target & 0xFFFFFF));
	u32 skipped = at(&l); emit(&l, C(GE_CMD_VIEWPORTXSCALE, f24(240)));
	emitFinish(&l);

	L t = { target, target };
	u32 sub1 = 0x08950000, sub2 = 0x08960000;
	emit(&t, C(GE_CMD_BASE, (sub1 >> 8) & 0x0F0000));
	emit(&t, C(GE_CMD_CALL, sub1 & 0xFFFFFF));
	u32 after_call = at(&t); emit(&t, C(GE_CMD_OFFSETX, (2048 - 240) << 4));
	/* signal jump to sub2 (behaviour 0x10, address split in signal/end data) */
	emit(&t, C(GE_CMD_SIGNAL, (GE_SIGNAL_JUMP << 16) | (sub2 >> 16)));
	emit(&t, C(GE_CMD_END, sub2 & 0xFFFF));
	u32 dead = at(&t); emit(&t, C(GE_CMD_OFFSETY, (2048 - 136) << 4));
	emitFinish(&t);

	L s1 = { sub1, sub1 };
	u32 in_sub1 = at(&s1); emit(&s1, C(GE_CMD_VIEWPORTYSCALE, f24(-136)));
	emit(&s1, C(GE_CMD_RET, 0));

	L s2 = { sub2, sub2 };
	u32 in_sub2 = at(&s2); emit(&s2, C(GE_CMD_SCISSOR2, (271 << 10) | 479));
	emitFinish(&s2);

	geNativeListReset();
	geNativeListPatch(l.start, 0);
	CHECK(unf24(data_of(skipped)) == 240, "command after JUMP not visited");
	CHECK(unf24(data_of(in_sub1)) == -272, "CALL target patched");
	CHECK(data_of(after_call) == (u32)((2048 - 480) << 4), "RET returns after CALL");
	CHECK(data_of(in_sub2) == ((543 << 10) | 959), "SIGNAL jump followed");
	CHECK(data_of(dead) == (u32)((2048 - 136) << 4), "code after SIGNAL jump not visited");
}

static void test_robustness(void) {
	reset_mem();
	/* Endless loop */
	L l = { 0x08900000, 0x08900000 };
	emit(&l, C(GE_CMD_BASE, (l.start >> 8) & 0x0F0000));
	emit(&l, C(GE_CMD_JUMP, l.start & 0xFFFFFF));
	geNativeListReset();
	geNativeListPatch(l.start, 0);
	CHECK(1, "endless jump loop terminates");

	/* Jump to unmapped memory */
	L l2 = { 0x08910000, 0x08910000 };
	emit(&l2, C(GE_CMD_BASE, 0));
	emit(&l2, C(GE_CMD_JUMP, 0x001000));
	geNativeListReset();
	geNativeListPatch(l2.start, 0);
	CHECK(1, "jump to unmapped address stops (no crash)");

	/* Vertex buffer pointing outside memory */
	L l3 = { 0x08920000, 0x08920000 };
	emit(&l3, C(GE_CMD_VERTEXTYPE, VT_THROUGH_C8888_P16));
	emit(&l3, C(GE_CMD_BASE, 0x0F0000));
	emit(&l3, C(GE_CMD_VADDR, 0xFFFFF0));
	emit(&l3, C(GE_CMD_PRIM, (6 << 16) | 100));
	emitFinish(&l3);
	geNativeListReset();
	geNativeListPatch(l3.start, 0);
	CHECK(1, "out of range vertices skipped (no crash)");

	/* List running into the end of RAM without END */
	u32 tail = RAM_BASE + RAM_SIZE - 8;
	*W(tail) = C(GE_CMD_NOP, 0);
	*W(tail + 4) = C(GE_CMD_NOP, 0);
	geNativeListReset();
	geNativeListPatch(tail, 0);
	CHECK(1, "list running off RAM stops");

	/* Stall address stops traversal */
	reset_mem();
	L l4 = { 0x08930000, 0x08930000 };
	u32 a = at(&l4); emit(&l4, C(GE_CMD_VIEWPORTXSCALE, f24(240)));
	u32 stall = at(&l4);
	u32 b = at(&l4); emit(&l4, C(GE_CMD_VIEWPORTXSCALE, f24(240)));
	emitFinish(&l4);
	geNativeListReset();
	geNativeListPatch(l4.start, stall);
	CHECK(unf24(data_of(a)) == 480 && unf24(data_of(b)) == 240, "stall respected");
	geNativeListPatch(stall, 0);
	CHECK(unf24(data_of(b)) == 480, "continue after stall update");

	/* Original-value map overflow: many tracked commands */
	reset_mem();
	L l5 = { 0x08940000, 0x08940000 };
	for (int i = 0; i < 40000; i++) emit(&l5, C(GE_CMD_VIEWPORTXSCALE, f24(240)));
	emitFinish(&l5);
	geNativeListReset();
	geNativeListPatch(l5.start, 0);
	int ok = 1;
	for (int i = 0; i < 40000; i++) if (unf24(data_of(l5.start + 4 * i)) != 480) ok = 0;
	CHECK(ok, "40000 tracked commands scaled once");

	/* A static list that runs every frame keeps its originals while the game
	   streams lots of one-shot lists through fresh addresses */
	reset_mem();
	L st = { 0x08950000, 0x08950000 };
	for (int i = 0; i < 200; i++) emit(&st, C(GE_CMD_VIEWPORTXSCALE, f24(240)));
	emitFinish(&st);
	for (int frame = 0; frame < 300; frame++) {
		geNativeListReset();
		geNativeListPatch(st.start, 0);
		L dyn = { 0x09000000 + frame * 0x8000, 0x09000000 + frame * 0x8000 };
		for (int i = 0; i < 1000; i++) emit(&dyn, C(GE_CMD_OFFSETX, (2048 - 240) << 4));
		emitFinish(&dyn);
		geNativeListReset();
		geNativeListPatch(dyn.start, 0);
		geNativeListTakeFrameDraws(); /* display flip */
	}
	ok = 1;
	for (int i = 0; i < 200; i++) if (unf24(data_of(st.start + 4 * i)) != 480) ok = 0;
	CHECK(ok, "static list survives 300k streamed commands without double scaling");
}

static void test_indexed_and_float(void) {
	reset_mem();
	L l = { 0x08900000, 0x08900000 };
	emit(&l, C(GE_CMD_FRAMEBUFPTR, 0));
	emit(&l, C(GE_CMD_FRAMEBUFWIDTH, 512));

	/* float positions, 565 colour (2 bytes) + pad to 4 + 12 = 16 bytes, padding at 2 */
	u32 vb = 0x08A10000;
	float fp[3][2] = { { 1.5f, 2.0f }, { 480, 272 }, { -10, 100 } };
	for (int i = 0; i < 3; i++) { *F(vb + i * 16 + 4) = fp[i][0]; *F(vb + i * 16 + 8) = fp[i][1]; }
	u32 ib = 0x08A20000;
	u16 idx[4] = { 2, 0, 1, 2 };
	memcpy(ge_host_ptr(ib), idx, sizeof(idx));
	emit(&l, C(GE_CMD_VERTEXTYPE, VT_THROUGH_C565_PF | GE_VTYPE_IDX_16BIT));
	emitVaddr(&l, vb);
	emit(&l, C(GE_CMD_BASE, (ib >> 8) & 0x0F0000));
	emit(&l, C(GE_CMD_IADDR, ib & 0xFFFFFF));
	emit(&l, C(GE_CMD_PRIM, (3 << 16) | 4));
	emitFinish(&l);

	geNativeListReset();
	geNativeListPatch(l.start, 0);
	float exp[3][2] = { { 3, 4 }, { 960, 544 }, { -20, 200 } };
	for (int i = 0; i < 3; i++)
		CHECK(*F(vb + i * 16 + 4) == exp[i][0] && *F(vb + i * 16 + 8) == exp[i][1],
		      "indexed float vertex %d = %f,%f", i, *F(vb + i * 16 + 4), *F(vb + i * 16 + 8));

	geNativeListReset();
	geNativeListPatch(l.start, 0);
	for (int i = 0; i < 3; i++)
		CHECK(*F(vb + i * 16 + 4) == exp[i][0], "indexed float vertex %d not scaled twice", i);
}

/* Vertex layouts without padding, single sprites, repeated passes */
static void test_vertex_fingerprint(void) {
	int ok;
	reset_mem();
	u32 vb = 0x08A30000; /* tc16 + pos16 = 10 bytes, no padding */
	*H(vb + 4) = 100; *H(vb + 6) = 50; *H(vb + 14) = 300; *H(vb + 16) = 200;
	L l = { 0x08900000, 0x08900000 };
	emit(&l, C(GE_CMD_FRAMEBUFPTR, 0));
	emit(&l, C(GE_CMD_FRAMEBUFWIDTH, 512));
	emit(&l, C(GE_CMD_VERTEXTYPE, VT_THROUGH_T16_P16));
	emitVaddr(&l, vb);
	emit(&l, C(GE_CMD_PRIM, (6 << 16) | 2));
	/* same buffer drawn a second time in the same list */
	emitVaddr(&l, vb);
	emit(&l, C(GE_CMD_PRIM, (6 << 16) | 2));
	emitFinish(&l);

	for (int pass = 0; pass < 5; pass++) {
		geNativeListReset();
		geNativeListPatch(l.start, 0);
		geNativeListTakeFrameDraws();
	}
	CHECK(*H(vb + 4) == 200 && *H(vb + 6) == 100 && *H(vb + 14) == 600 && *H(vb + 16) == 400,
	      "unpadded sprite scaled exactly once over 5 passes x 2 draws (%d,%d %d,%d)",
	      *H(vb + 4), *H(vb + 6), *H(vb + 14), *H(vb + 16));

	/* game rewrites the vertices with new originals -> scaled again */
	*H(vb + 4) = 10; *H(vb + 6) = 20; *H(vb + 14) = 30; *H(vb + 16) = 40;
	geNativeListReset();
	geNativeListPatch(l.start, 0);
	CHECK(*H(vb + 4) == 20 && *H(vb + 6) == 40 && *H(vb + 14) == 60 && *H(vb + 16) == 80, "rewritten vertices scaled");

	/* game rewrites only one coordinate */
	*H(vb + 14) = 30;
	geNativeListReset();
	geNativeListPatch(l.start, 0);
	CHECK(*H(vb + 14) == 60, "partially rewritten buffer detected (%d)", *H(vb + 14));

	/* the same buffer drawn with 4 vertices, then 2, then 4 again */
	u32 vb4 = 0x08A40000;
	for (int i = 0; i < 4; i++) { *H(vb4 + i * 10 + 4) = 10 * (i + 1); *H(vb4 + i * 10 + 6) = 1 + i; }
	L l4 = { 0x08960000, 0x08960000 };
	emit(&l4, C(GE_CMD_FRAMEBUFPTR, 0));
	emit(&l4, C(GE_CMD_FRAMEBUFWIDTH, 512));
	emit(&l4, C(GE_CMD_VERTEXTYPE, VT_THROUGH_T16_P16));
	emitVaddr(&l4, vb4); emit(&l4, C(GE_CMD_PRIM, (6 << 16) | 2));
	emitVaddr(&l4, vb4); emit(&l4, C(GE_CMD_PRIM, (6 << 16) | 4));
	emitVaddr(&l4, vb4); emit(&l4, C(GE_CMD_PRIM, (6 << 16) | 2));
	emitFinish(&l4);
	for (int pass = 0; pass < 3; pass++) {
		geNativeListReset();
		geNativeListPatch(l4.start, 0);
	}
	ok = 1;
	for (int i = 0; i < 4; i++) if (*H(vb4 + i * 10 + 4) != 20 * (i + 1) || *H(vb4 + i * 10 + 6) != 2 * (1 + i)) ok = 0;
	CHECK(ok, "prefix draws of one buffer scaled exactly once (%d %d %d %d)",
	      *H(vb4 + 4), *H(vb4 + 14), *H(vb4 + 24), *H(vb4 + 34));

	/* values outside the guard band are left alone */
	*H(vb + 4) = 3000; *H(vb + 6) = -3000;
	geNativeListReset();
	geNativeListPatch(l.start, 0);
	CHECK(*H(vb + 4) == 3000 && *H(vb + 6) == -3000, "out of range positions unchanged");
}

#ifdef GE_NATIVE_TRACE
static void test_trace(void) {
	reset_mem();
	L l = { 0x08900000, 0x08900000 };
	emitSetup(&l, 0x000000, 512, 240, -136, 2048, 2048, 2048 - 240, 2048 - 136);
	emit(&l, C(GE_CMD_CLEARMODE, 1));
	emit(&l, C(GE_CMD_VERTEXTYPE, VT_THROUGH_C8888_P16));
	emitVaddr(&l, 0x08A00000);
	emit(&l, C(GE_CMD_PRIM, (6 << 16) | 2));
	emitFinish(&l);

	trace_n = 0;
	geNativeTraceOn = 1;
	geNativeListReset();
	geNativeListPatch(l.start, 0);
	geNativeTraceOn = 0;

	CHECK(strstr(trace_out, "L 08900000") != NULL, "trace: list start");
	CHECK(strstr(trace_out, "F ") && strstr(trace_out, "-> hires"), "trace: framebuffer decision");
	CHECK(strstr(trace_out, "V 42 ") != NULL, "trace: viewport rewrite");
	CHECK(strstr(trace_out, "M ") != NULL, "trace: clear mode");
	CHECK(strstr(trace_out, "clr 1") && strstr(trace_out, " on\n"), "trace: draw line");
	CHECK(strstr(trace_out, "C finish") != NULL, "trace: finish");

	/* tracing off: nothing recorded */
	trace_n = 0;
	trace_out[0] = 0;
	geNativeListReset();
	geNativeListPatch(l.start, 0);
	CHECK(trace_n == 0, "trace off records nothing");
}
#endif

static void test_copy_list(void) {
	u32 list[16];
	int n = geNativeListBuildCopy(list);
	CHECK(n == 10, "copy list length");
	CHECK(list[0] == C(GE_CMD_TRANSFERSRC, 0x000000), "src");
	CHECK(list[1] == C(GE_CMD_TRANSFERSRCW, (0x04 << 16) | 960), "src width %08x", list[1]);
	CHECK(list[3] == C(GE_CMD_TRANSFERDST, 0x400000), "dst");
	CHECK(list[4] == C(GE_CMD_TRANSFERDSTW, (0x0A << 16) | 960), "dst width %08x", list[4]);
	CHECK(list[6] == C(GE_CMD_TRANSFERSIZE, (543 << 10) | 959), "size");
	CHECK(list[7] == C(GE_CMD_TRANSFERSTART, 0), "16 bpp transfer");
	CHECK(list[8] >> 24 == GE_CMD_FINISH && list[9] >> 24 == GE_CMD_END, "finish/end");
}

static void test_memory_layout(void) {
	CHECK(GE_NATIVE_VRAM_DRAW + GE_NATIVE_PITCH * GE_NATIVE_HEIGHT * 2 <= GE_NATIVE_VRAM_DEPTH, "colour buffer fits before depth");
	CHECK(GE_NATIVE_VRAM_DEPTH + GE_NATIVE_PITCH * GE_NATIVE_HEIGHT * 2 <= GE_NATIVE_VRAM_DUMMY, "depth fits before scratch");
	CHECK(GE_NATIVE_VRAM_DUMMY + 0x1000 <= VRAM_BASE + VRAM_SIZE, "scratch inside VRAM");
	CHECK(GE_NATIVE_FAKE_VRAM + GE_NATIVE_FAKE_VRAM_SIZE <= GE_NATIVE_DISPLAY_BUFFER, "fake vram before display buffer");
	CHECK(GE_NATIVE_DISPLAY_BUFFER + GE_NATIVE_PITCH * GE_NATIVE_HEIGHT * 2 <= GE_NATIVE_ORIG_MAP, "display buffer before map");
	CHECK(GE_NATIVE_ORIG_MAP + GE_NATIVE_ORIG_MAP_ENTRIES * 16 <= GE_NATIVE_VTX_MAP, "orig map before vertex map");
	CHECK(GE_NATIVE_VTX_MAP + GE_NATIVE_VTX_MAP_ENTRIES * 20 <= GE_NATIVE_COPY_LIST, "vertex map before copy list");
	CHECK(GE_NATIVE_COPY_LIST + 0x100 <= 0x0B000000, "inside the region cleared at boot");
}

int main(void) {
	ram = calloc(1, RAM_SIZE);
	vram = calloc(1, VRAM_SIZE);

	test_memory_layout();
	test_basic_fullscreen();
	test_subviewport();
	test_offscreen_and_restore();
	test_control_flow();
	test_robustness();
	test_indexed_and_float();
	test_vertex_fingerprint();
	test_copy_list();
#ifdef GE_NATIVE_TRACE
	test_trace();
#endif

	printf("%d checks, %d failures\n", checks, failures);
	return failures ? 1 : 0;
}
