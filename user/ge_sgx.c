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
	SGX renderer, Vita side: reads the GE events that systemctrl writes into
	the ring in PSP RAM (see ge_sgx_shared.h), walks the game's display lists
	with the walker (ge_sgx_walk.c) and hands each finished frame to the draw
	thread (AdrenalineDraw in menu.c), which renders it with the GXM context.

	PSP RAM is read through the cacheable mapping of ScePspemuConvertAddress:
	a page is invalidated the first time it is read after a new event, so
	data that the game wrote since (a list extended up to a new stall, reused
	vertex buffers) is never taken from a stale cache line.
*/

#include <psp2/io/fcntl.h>
#include <psp2/kernel/dmac.h>
#include <psp2/kernel/processmgr.h>
#include <psp2/kernel/sysmem.h>
#include <psp2/kernel/cpu.h>
#include <psp2/kernel/threadmgr.h>

#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "main.h"
#include "ge_sgx.h"
#include "ge_sgx_pool.h"
#include "ge_sgx_tex.h"
#include "ge_sgx_render.h"
#include "../ge_sgx_shared.h"
#include "ge_sgx_texconv.h"

// Test: every 3D draw transformed on the CPU (the GPU transform's vertex programs not used)
#define GE_SGX_TEST_CPU_TRANSFORM 0


#define LOG_PATH "ux0:data/adrenaline_sgx_log.txt"
#define LOG_INTERVAL_US (5 * 1000 * 1000)

#define EVENTS_PER_BATCH 256
#define IDLE_DELAY_US 300
#define IDLE_DELAY_SIGNALS_US 100

#define RAM_START  0x08000000u
#define VRAM_START 0x04000000u
#define VRAM_SIZE  0x00200000u

// Copies of PSP memory blocks for the walker, direct mapped
#define BLOCK_CACHE_ENTRIES 2048

#define RING_EVENTS_OFFSET ((uint32_t)offsetof(GeSgxRing, events))

// Start request from the Kermit thread
static volatile uint32_t start_seq = 0;
static volatile uint32_t start_ring = 0;
static volatile uint32_t start_session = 0;
static volatile uint32_t start_entries = 0;

static SceUID thid = -1;

static GeSgxWalker walker;

/*
	ScePspemu maps the PSP RAM uncached (see the benchmark): every load is a
	trip to memory, ~125 ns per word. The walker therefore reads copies of
	GE_SGX_BLOCK_SIZE byte blocks, made with burst loads (memcpy), kept for
	one event (the game may change the memory between events).
*/
static uint32_t epoch = 1;
static const uint8_t *ram_map = NULL;  // PSP RAM 0x08000000 in our address space
static const uint8_t *vram_map = NULL; // PSP VRAM 0x04000000 (the PSP side's copy of it)
static uint32_t vram_writes = 0, vram_write_bytes = 0; // GE transfers into it done here (vramCopyFn)
// pspemu maps PSP VRAM uncached (a USER_RW_UNCACHE block): the CPU reads it as it is
// The VRAM pages read so far, for the PSP side to copy only those (GeSgxVramWant)
static uint32_t vram_want[GE_SGX_VRAM_PAGES / 32];
static int vram_want_dirty = 0;

// Pages wanted only for the PSP's picture of an asked trace (framebuffers): not wanted again after it
static uint32_t picture_pages[GE_SGX_VRAM_PAGES / 32];

/*
	The flip a page was last read at: pages not read for VRAM_UNWANT_FLIPS
	are not wanted any more (the PSP side copied every page ever read at
	every flip: 1.2 MB, 3 ms of the game's time a frame in The 3rd Birthday)
*/
#define VRAM_UNWANT_FLIPS 180
static uint16_t vram_used[GE_SGX_VRAM_PAGES];
static uint16_t vram_flips = 0;

static inline void wantVram(uint32_t off, uint32_t size) {
	uint32_t p = off / GE_SGX_VRAM_PAGE, last = (off + (size ? size : 1) - 1) / GE_SGX_VRAM_PAGE;
	if (last >= GE_SGX_VRAM_PAGES)
		last = GE_SGX_VRAM_PAGES - 1;
	for (; p <= last; p++) {
		uint32_t bit = 1u << (p % 32);
		vram_used[p] = vram_flips;
		picture_pages[p / 32] &= ~bit; // wanted for itself now
		if (!(vram_want[p / 32] & bit)) {
			vram_want[p / 32] |= bit;
			vram_want_dirty = 1;
		}
	}
}
static uint32_t *block_tag = NULL;     // block address | 1 (valid)
static uint32_t *block_epoch = NULL;
static uint8_t *block_data = NULL;
static SceUID block_mem = -1;
static uint32_t block_copies = 0; // blocks copied
static uint32_t block_calls = 0;  // block lookups by the walker

static int initBlockCache(void) {
	if (block_mem >= 0)
		return 1;

	uint32_t size = BLOCK_CACHE_ENTRIES * (GE_SGX_BLOCK_SIZE + 8);
	size = (size + 0xFFF) & ~0xFFF;
	block_mem = sceKernelAllocMemBlock("SceGeSgxBlocks", SCE_KERNEL_MEMBLOCK_TYPE_USER_RW, size, NULL);
	if (block_mem < 0)
		return 0;

	void *base = NULL;
	sceKernelGetMemBlockBase(block_mem, &base);
	memset(base, 0, size);
	block_data = (uint8_t *)base;
	block_tag = (uint32_t *)(block_data + BLOCK_CACHE_ENTRIES * GE_SGX_BLOCK_SIZE);
	block_epoch = block_tag + BLOCK_CACHE_ENTRIES;
	return 1;
}

/*
	The commands the GE did between two signals, as the PSP copied them
	before the game's handler wrote over them (GE_SGX_EVENT_SIGNAL's shadow):
	put over the blocks the walker reads while it walks up to that signal
*/
#define MAX_OVERLAYS 96
static struct {
	uint32_t addr, len; // PSP address (no cache bits), bytes
	const uint8_t *data;
} overlays[MAX_OVERLAYS];
static uint32_t noverlays = 0;
static uint32_t overlays_used = 0, overlays_stale = 0;

static inline void applyOverlay(uint8_t *dst, uint32_t block) {
	uint32_t k;
	for (k = 0; k < noverlays; k++) {
		uint32_t a = overlays[k].addr, n = overlays[k].len;
		if (a >= block + GE_SGX_BLOCK_SIZE || a + n <= block)
			continue;
		uint32_t from = a > block ? a : block;
		uint32_t to = a + n < block + GE_SGX_BLOCK_SIZE ? a + n : block + GE_SGX_BLOCK_SIZE;
		memcpy(dst + (from - block), overlays[k].data + (from - a), to - from);
	}
}

// The cached blocks of the overlays' memory read again (the others kept: a new epoch copied every block anew, at every signal)
static void overlayBlocksStale(void) {
	uint32_t k;
	for (k = 0; k < noverlays; k++) {
		uint32_t b = overlays[k].addr & ~(uint32_t)(GE_SGX_BLOCK_SIZE - 1);
		for (; b < overlays[k].addr + overlays[k].len; b += GE_SGX_BLOCK_SIZE) {
			uint32_t i = (b / GE_SGX_BLOCK_SIZE) & (BLOCK_CACHE_ENTRIES - 1);
			if (block_tag[i] == (b | 1))
				block_tag[i] = 0;
		}
	}
}

/*
	A copy record (the PSP side's captureSegment): the event's seq, the
	number of pieces, then each piece's address, words and words
*/
static int takeOverlays(const uint32_t *rec, uint32_t words, uint32_t seq) {
	if (words < 2 || rec[0] != seq)
		return 0;
	uint32_t pieces = rec[1], pos = 2, k;
	noverlays = 0;
	for (k = 0; k < pieces && noverlays < MAX_OVERLAYS && pos + 2 <= words; k++) {
		uint32_t addr = rec[pos], n = rec[pos + 1];
		if (pos + 2 + n > words)
			break;
		overlays[noverlays].addr = addr & 0x0FFFFFFCu;
		overlays[noverlays].len = n * 4;
		overlays[noverlays].data = (const uint8_t *)(rec + pos + 2);
		noverlays++;
		pos += 2 + n;
	}
	return noverlays > 0;
}

static const void *blockFn(void *user, uint32_t block) {
	const uint8_t *src;

	if (block >= RAM_START && block < RAM_START + PSP_RAM_SIZE) {
		src = ram_map + (block - RAM_START);
	} else if (block >= VRAM_START && block < VRAM_START + 0x00800000u) {
		// VRAM and its mirrors
		src = vram_map + (block & (VRAM_SIZE - 1));
		wantVram(block & (VRAM_SIZE - 1), GE_SGX_BLOCK_SIZE);
	} else {
		return NULL;
	}

	block_calls++;

	uint32_t i = (block / GE_SGX_BLOCK_SIZE) & (BLOCK_CACHE_ENTRIES - 1);
	uint8_t *dst = block_data + i * GE_SGX_BLOCK_SIZE;

	if (block_tag[i] != (block | 1) || block_epoch[i] != epoch) {
		memcpy(dst, src, GE_SGX_BLOCK_SIZE);
		applyOverlay(dst, block);
		block_tag[i] = block | 1;
		block_epoch[i] = epoch;
		block_copies++;
	}

	return dst;
}

uint32_t geSgxPspSpan(uint32_t addr) {
	addr &= 0x0FFFFFFF;
	if (addr >= RAM_START && addr < RAM_START + PSP_RAM_SIZE)
		return RAM_START + PSP_RAM_SIZE - addr;
	if (addr >= VRAM_START && addr < VRAM_START + 0x00800000u)
		return VRAM_SIZE - (addr & (VRAM_SIZE - 1));
	return 0;
}

const uint8_t *geSgxPspPtr(uint32_t addr, uint32_t size) {
	addr &= 0x0FFFFFFF;
	if (!ram_map || !vram_map)
		return NULL;
	if (addr >= RAM_START && addr < RAM_START + PSP_RAM_SIZE) {
		if (size > RAM_START + PSP_RAM_SIZE - addr)
			return NULL;
		return ram_map + (addr - RAM_START);
	}
	if (addr >= VRAM_START && addr < VRAM_START + 0x00800000u) {
		uint32_t off = addr & (VRAM_SIZE - 1);
		if (size > VRAM_SIZE - off)
			return NULL;
		wantVram(off, size);
		return vram_map + off;
	}
	return NULL;
}

/*
	Diagnostics: is there anything in VRAM where we read it? Every 300th
	frame the non-zero words of the framebuffers
	(0 - 0x88000), of the texture area (0xD0000 - 0x1C0000) and of all of it
	are counted, and the memory blocks of other addresses that may be VRAM are
	listed (only listed: they are not read).
*/
static uint32_t nonZero(const uint8_t *base, uint32_t from, uint32_t to) {
	const uint32_t *p = (const uint32_t *)(base + from);
	uint32_t i, n = 0;
	for (i = 0; i < (to - from) / 4; i++)
		n += p[i] != 0;
	return n;
}

static void blockInfo(const char *what, const void *addr) {
	SceKernelMemBlockInfo info;
	memset(&info, 0, sizeof(info));
	info.size = sizeof(info);
	int r = sceKernelGetMemBlockInfoByAddr((void *)addr, &info);
	GeSgxLog("GeSgx VRAM probe: %s %08X: %s block %08X size %X type %08X\n", what, (unsigned)(uintptr_t)addr,
		r < 0 ? "no" : "in", (unsigned)(uintptr_t)info.mappedBase, (unsigned)info.mappedSize, (unsigned)info.type);
}

static void probeVram(void) {
	GeSgxLog("GeSgx VRAM probe: non-zero words at %08X: framebuffers %u of %u, textures %u of %u, all %u of %u\n",
		(unsigned)(uintptr_t)vram_map, (unsigned)nonZero(vram_map, 0, 0x88000), 0x88000u / 4,
		(unsigned)nonZero(vram_map, 0xD0000, 0x1C0000), (0x1C0000u - 0xD0000u) / 4, (unsigned)nonZero(vram_map, 0, VRAM_SIZE), VRAM_SIZE / 4);
	static int listed = 0;
	if (!listed) {
		listed = 1;
		blockInfo("VRAM (ConvertAddress 0x04000000)", vram_map);
		blockInfo("ConvertAddress 0x44000000", (const void *)ScePspemuConvertAddress(0x44000000u, 0, 0));
		blockInfo("ConvertAddress 0x84000000", (const void *)ScePspemuConvertAddress(0x84000000u, 0, 0));
		blockInfo("ConvertAddress 0x04200000", (const void *)ScePspemuConvertAddress(0x04200000u, 0, 0));
		blockInfo("pspemu display 0x74000000", (const void *)0x74000000u);
		blockInfo("RAM 0x08000000", ram_map);
	}
}

static uint32_t clockUs(void) {
	return sceKernelGetProcessTimeLow();
}

static uint32_t clock_ticks_per_us = 1; // the walker's clock is in microseconds

static void clockInit(GeSgxWalker *w) {
	w->clock_fn = clockUs;
	w->clock_every = 32; // a system call (2.4 us): one primitive in 32 is timed
	clock_ticks_per_us = 1;
}

// Diagnostics: a file of the asked trace's dump (the walker's textures)
static void dumpWrite(const char *name, const void *data, uint32_t bytes) {
	char path[128];
	sceIoMkdir("ux0:data/adrenaline/sgx_dump", 0777);
	snprintf(path, sizeof(path), "ux0:data/adrenaline/sgx_dump/%.64s", name);
	SceUID fd = sceIoOpen(path, SCE_O_WRONLY | SCE_O_CREAT | SCE_O_TRUNC, 0777);
	if (fd >= 0) {
		sceIoWrite(fd, data, bytes);
		sceIoClose(fd);
	}
}

/*
	The textures of an asked trace are kept in memory and written when the
	trace ends: written while the lists were walked, the walk took hundreds
	of ms an event, the ring overflowed and the traced frame was lost (The
	3rd Birthday's, twice)
*/
#define DUMP_BUF_BYTES (1024 * 1024)
#define DUMP_ITEMS 48
static SceUID dump_mem = -1;
static uint8_t *dump_buf = NULL;
static uint32_t dump_used = 0, dump_n = 0;
static struct { char name[64]; uint32_t off, len; } dump_items[DUMP_ITEMS];

static void dumpFn(const char *name, const void *data, uint32_t bytes) {
	if (!dump_buf && dump_mem < 0) {
		dump_mem = sceKernelAllocMemBlock("SceGeSgxDump", SCE_KERNEL_MEMBLOCK_TYPE_USER_RW, DUMP_BUF_BYTES, NULL);
		if (dump_mem >= 0)
			sceKernelGetMemBlockBase(dump_mem, (void **)&dump_buf);
	}
	if (!dump_buf || dump_n >= DUMP_ITEMS || dump_used + bytes > DUMP_BUF_BYTES)
		return; // dropped (written at once, they lost the frame)
	memcpy(dump_buf + dump_used, data, bytes);
	strncpy(dump_items[dump_n].name, name, sizeof(dump_items[0].name) - 1);
	dump_items[dump_n].name[sizeof(dump_items[0].name) - 1] = '\0';
	dump_items[dump_n].off = dump_used;
	dump_items[dump_n].len = bytes;
	dump_n++;
	dump_used += bytes;
}

static void dumpFlush(void) {
	uint32_t i;
	for (i = 0; i < dump_n; i++)
		dumpWrite(dump_items[i].name, dump_buf + dump_items[i].off, dump_items[i].len);
	dump_n = 0;
	dump_used = 0;
	if (dump_mem >= 0) {
		sceKernelFreeMemBlock(dump_mem);
		dump_mem = -1;
		dump_buf = NULL;
	}
}

/*
	A GE transfer into VRAM on the Vita side's copy (the window, which the
	flip's copy from the PSP fills too): the textures it brings are there
	for this frame's draws. Source anywhere (RAM, VRAM), rows may overlap.
*/
static void vramCopyFn(uint32_t dst, uint32_t src, uint32_t row_bytes, uint32_t rows, uint32_t dst_stride, uint32_t src_stride) {
	if (!vram_map || !rows || !row_bytes)
		return;
	dst &= 0x0FFFFFFFu;
	src &= 0x0FFFFFFFu;
	uint32_t span = (rows - 1) * dst_stride + row_bytes, sspan = (rows - 1) * src_stride + row_bytes;
	if (dst < VRAM_START || dst >= VRAM_START + 0x00800000u)
		return;
	uint32_t off = dst & (VRAM_SIZE - 1);
	if (off + span > VRAM_SIZE)
		return;
	const uint8_t *s = geSgxPspPtr(src, sspan);
	if (!s)
		return;
	uint8_t *d = (uint8_t *)vram_map + off;
	uint32_t r;
	if (dst_stride == row_bytes && src_stride == row_bytes) {
		memmove(d, s, span);
	} else {
		for (r = 0; r < rows; r++)
			memmove(d + r * dst_stride, s + r * src_stride, row_bytes);
	}
	wantVram(off, span);
	geSgxTexMemoryWritten(dst, span);
	vram_writes++;
	vram_write_bytes += span;
	epoch++; // the walker's blocks of it are read again
}

// Vertices of big primitives: one DMA (about 80 us a call, then 2 ns/byte against 11 for a copy)
static int bulkFn(uint32_t addr, void *dst, uint32_t size) {
	const uint8_t *src = geSgxPspPtr(addr, size);
	if (!src)
		return 0;
	sceDmacMemcpy(dst, src, size);
	return 1;
}

/*
	Frames for the GXM renderer: the walker builds one while it walks, and
	publishes it at the display flip; the draw thread renders the newest
	published one. A frame is built again only when the GPU is done with it
	(some draw thread loops after it was rendered).
*/
#define FRAME_SLOTS 3
#define FRAME_VERTICES 16384      // CPU transformed (2D and what the GPU transform cannot do)
#define FRAME_INDICES (3 * 65536)
#define FRAME_UNIFORMS (512 * 1024)
#define FRAME_RAW (1280 * 1024)   // PSP vertex data for the GPU transform
#define FRAME_DRAWS 8192
#define GPU_DONE_TICKS 3

enum {
	SLOT_FREE,
	SLOT_BUILDING,
	SLOT_READY,
	SLOT_RENDERING,
	SLOT_RENDERED,
};

typedef struct {
	GeSgxFrameData data;
	volatile int state;
	volatile uint32_t seq;
	volatile uint32_t rendered_tick;
	uint32_t fb;
	GeSgxAlias aliases[GE_SGX_MAX_ALIASES];
	int naliases;
} FrameSlot;

static FrameSlot slots[FRAME_SLOTS];
static int slots_ready = 0;
static FrameSlot *building = NULL;
static uint32_t frame_seq_counter = 0;
static volatile uint32_t draw_tick = 0;
static uint32_t frames_dropped = 0, frames_published = 0, frames_skipped = 0;

/*
	A frame's vertices, indices and uniforms are written in cached memory
	(staging, one for the frame being built: writes and the read-modify-
	writes of merged indices are slow in uncached memory) and copied into the
	slot's GPU memory when it is published. The raw arena (PSP vertex data
	for the GPU transform) is filled by DMA straight into GPU memory.
*/
#define OFF_VERTICES 0u
#define OFF_INDICES (OFF_VERTICES + FRAME_VERTICES * (uint32_t)sizeof(GeSgxVertex))
#define OFF_UNIFORMS (OFF_INDICES + FRAME_INDICES * 2u)
#define OFF_RAW (OFF_UNIFORMS + FRAME_UNIFORMS)
#define STAGING_SIZE OFF_RAW

static uint8_t *staging = NULL;
static uint32_t direct_size = 0; // PSP RAM mapped for the GPU, from 0x08800000

static int initSlots(void) {
	if (slots_ready)
		return 1;
	SceUID sm = sceKernelAllocMemBlock("SceGeSgxStaging", SCE_KERNEL_MEMBLOCK_TYPE_USER_RW, (STAGING_SIZE + 0xFFF) & ~0xFFFu, NULL);
	if (sm < 0)
		return 0;
	sceKernelGetMemBlockBase(sm, (void **)&staging);

	int i;
	for (i = 0; i < FRAME_SLOTS; i++) {
		uint32_t vsize = (OFF_RAW + FRAME_RAW + 0xFFF) & ~0xFFFu;
		uint32_t dsize = (FRAME_DRAWS * sizeof(GeSgxDraw) + 0xFFF) & ~0xFFFu;
		SceUID vm = sceKernelAllocMemBlock("SceGeSgxVertices", SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE, vsize, NULL);
		SceUID dm = sceKernelAllocMemBlock("SceGeSgxDraws", SCE_KERNEL_MEMBLOCK_TYPE_USER_RW, dsize, NULL);
		if (vm < 0 || dm < 0)
			return 0;
		uint8_t *vb = NULL;
		void *db = NULL;
		sceKernelGetMemBlockBase(vm, (void **)&vb);
		sceKernelGetMemBlockBase(dm, &db);
		if (sceGxmMapMemory(vb, vsize, SCE_GXM_MEMORY_ATTRIB_READ) < 0)
			return 0;
		memset(&slots[i], 0, sizeof(slots[i]));
		GeSgxFrameData *f = &slots[i].data;
		f->vertices = (GeSgxVertex *)(staging + OFF_VERTICES);
		f->max_vertices = FRAME_VERTICES;
		f->indices = (uint16_t *)(staging + OFF_INDICES);
		f->max_indices = FRAME_INDICES;
		f->uniforms = staging + OFF_UNIFORMS;
		f->max_uniforms = FRAME_UNIFORMS;
		f->raw = vb + OFF_RAW;
		f->max_raw = FRAME_RAW;
		f->direct = direct_size ? ram_map + 0x00800000u : NULL;
		f->vertices_gpu = (const GeSgxVertex *)(vb + OFF_VERTICES);
		f->indices_gpu = (const uint16_t *)(vb + OFF_INDICES);
		f->uniforms_gpu = vb + OFF_UNIFORMS;
		f->draws = (GeSgxDraw *)db;
		f->max_draws = FRAME_DRAWS;
		slots[i].state = SLOT_FREE;
	}
	slots_ready = 1;
	return 1;
}

// Staging to the GPU copy: one DMA for the big parts
static void toGpu(const void *gpu, const void *src, uint32_t bytes) {
	if (!bytes)
		return;
	if (bytes >= 4096)
		sceDmacMemcpy((void *)gpu, src, bytes);
	else
		memcpy((void *)gpu, src, bytes);
}

// Walker thread: a slot to build the next frame in (NULL: none free)
/*
	Diagnostics (traced frames): a list's top level as it is when the walker
	is done with it and again at the flip. A game whose signal handlers
	write commands into the list as the PSP's GE reaches them (The 3rd
	Birthday's HUD is in none of the draws walked) changes it in between.
*/
typedef struct { uint32_t words, prims, calls, jumps, signals, nops, sum; } ListScan;

static void scanList(uint32_t start, uint32_t end, ListScan *s, int log_calls) {
	memset(s, 0, sizeof(*s));
	start &= 0x0FFFFFFC;
	end &= 0x0FFFFFFC;
	if (end <= start || end - start > 1024 * 1024)
		return;
	const uint32_t *p = (const uint32_t *)geSgxPspPtr(start, end - start);
	if (!p)
		return;
	uint32_t i, n = (end - start) / 4, logged = 0;
	for (i = 0; i < n; i++) {
		uint32_t op = p[i], cmd = op >> 24;
		s->words++;
		s->sum = s->sum * 31u + op;
		if (cmd == 0x04) s->prims++;
		else if (cmd == 0x0A) {
			s->calls++;
			if (log_calls && logged < 24) {
				logged++;
				GeSgxLog("GeSgx T: list scan: call at %X to %06X\n", (unsigned)(start + i * 4), (unsigned)(op & 0xFFFFFF));
			}
		}
		else if (cmd == 0x08) s->jumps++;
		else if (cmd == 0x0E) s->signals++;
		else if (op == 0) s->nops++;
	}
}

static ListScan scan_done;
static uint32_t scan_done_count = 0, scan_start = 0, scan_end = 0;
// The list's words when the walker was done, to list what changed by the flip
#define SCAN_COPY_WORDS 16384
static uint32_t scan_copy[SCAN_COPY_WORDS];
static uint32_t scan_copy_n = 0;

static void subsNote(void);

static void noteListDone(void) {
	if (!walker.trace || walker.done_count == scan_done_count)
		return;
	scan_done_count = walker.done_count;
	scan_start = walker.done_start;
	scan_end = walker.done_end + 4;
	scanList(scan_start, scan_end, &scan_done, 0);
	scan_copy_n = 0;
	uint32_t bytes = scan_end - scan_start;
	const uint32_t *p = (scan_end > scan_start && bytes <= SCAN_COPY_WORDS * 4) ? (const uint32_t *)geSgxPspPtr(scan_start, bytes) : NULL;
	if (p) {
		scan_copy_n = bytes / 4;
		memcpy(scan_copy, p, bytes);
		subsNote();
	}
}

// The lists to their end: held ones after handler signals too (the GE is done with them)
static void walkAll(void) {
	geSgxWalkerRelease(&walker);
	geSgxWalkerRun(&walker);
	noteListDone();
}

/*
	The sublists the list calls (its top level's CALLs), each one's first
	256 words hashed when the walker is done and at the flip: a sublist the
	game fills as the GE reaches a signal before it (its handler) changes
*/
#define SCAN_SUBS 96
static uint32_t sub_addr[SCAN_SUBS], sub_hash[SCAN_SUBS], sub_words[SCAN_SUBS], sub_first[SCAN_SUBS][4], sub_n = 0;

static uint32_t subHash(uint32_t addr, uint32_t *words_out) {
	const uint32_t *p = (const uint32_t *)geSgxPspPtr(addr, 1024);
	if (!p)
		return 0;
	uint32_t h = 0, i;
	for (i = 0; i < 256; i++) {
		h = h * 31u + p[i];
		if ((p[i] >> 24) == 0x0B) // RET: the sublist's end
			break;
	}
	if (words_out)
		*words_out = i + 1;
	return h;
}

static void subsNote(void) {
	uint32_t k;
	sub_n = 0;
	for (k = 0; k < scan_copy_n && sub_n < SCAN_SUBS; k++) {
		uint32_t op = scan_copy[k];
		if ((op >> 24) != 0x0A)
			continue;
		uint32_t base = 0x08000000u; // BASE assumed at 08 (the top level's own region)
		uint32_t a = base | (op & 0x00FFFFFCu);
		uint32_t j;
		for (j = 0; j < sub_n && sub_addr[j] != a; j++)
			;
		if (j < sub_n)
			continue;
		sub_addr[sub_n] = a;
		sub_hash[sub_n] = subHash(a, &sub_words[sub_n]);
		const uint32_t *q = (const uint32_t *)geSgxPspPtr(a, 16);
		for (j = 0; j < 4; j++)
			sub_first[sub_n][j] = q ? q[j] : 0;
		sub_n++;
	}
}

static void subsDiff(void) {
	uint32_t k, changed = 0;
	for (k = 0; k < sub_n; k++) {
		uint32_t words = 0, h = subHash(sub_addr[k], &words);
		if (h != sub_hash[k]) {
			changed++;
			const uint32_t *p = (const uint32_t *)geSgxPspPtr(sub_addr[k], 32);
			GeSgxLog("GeSgx T: list scan: sublist %X changed (%u words, %u when walked: %08X %08X %08X %08X): %08X %08X %08X %08X %08X %08X %08X %08X\n",
				(unsigned)sub_addr[k], (unsigned)words, (unsigned)sub_words[k], (unsigned)sub_first[k][0], (unsigned)sub_first[k][1],
				(unsigned)sub_first[k][2], (unsigned)sub_first[k][3], p ? (unsigned)p[0] : 0u, p ? (unsigned)p[1] : 0u, p ? (unsigned)p[2] : 0u, p ? (unsigned)p[3] : 0u,
				p ? (unsigned)p[4] : 0u, p ? (unsigned)p[5] : 0u, p ? (unsigned)p[6] : 0u, p ? (unsigned)p[7] : 0u);
		}
	}
	GeSgxLog("GeSgx T: list scan: %u of %u sublists changed\n", (unsigned)changed, (unsigned)sub_n);
}

static void scanDiff(void) {
	subsDiff();
	const uint32_t *p = scan_copy_n ? (const uint32_t *)geSgxPspPtr(scan_start, scan_copy_n * 4) : NULL;
	uint32_t k, diffs = 0;
	for (k = 0; p && k < scan_copy_n; k++) {
		if (p[k] != scan_copy[k]) {
			if (diffs < 40)
				GeSgxLog("GeSgx T: list scan: changed at %X: %08X -> %08X\n", (unsigned)(scan_start + k * 4), (unsigned)scan_copy[k], (unsigned)p[k]);
			diffs++;
		}
	}
	GeSgxLog("GeSgx T: list scan: %u words changed\n", (unsigned)diffs);
}

static void acquireSlot(void) {
	int i;
	building = NULL;
	walker.frame = NULL;
	if (!slots_ready)
		return;
	for (i = 0; i < FRAME_SLOTS; i++) {
		FrameSlot *s = &slots[i];
		int st = s->state;
		if (st == SLOT_FREE || (st == SLOT_RENDERED && draw_tick - s->rendered_tick >= GPU_DONE_TICKS)) {
			s->state = SLOT_BUILDING;
			s->data.nvertices = 0;
			s->data.nindices = 0;
			s->data.nuniforms = 0;
			s->data.nraw = 0;
			s->data.ndraws = 0;
			s->data.nluts = 0;
			s->data.draws_3d = 0;
			s->data.prims_gpu = 0;
			memset(s->data.cpu_reason, 0, sizeof(s->data.cpu_reason));
			s->data.vertices_built = 0;
			s->data.state_builds = 0;
			s->data.us_state = s->data.us_texture = s->data.us_vertices = s->data.us_indices = 0;
			s->data.us_vindex = s->data.us_vuni = s->data.us_clut = 0;
			s->data.skipped_other = 0;
			s->data.skipped_range = 0;
			s->data.cull_off = 0;
			s->data.asked = 0;
			building = s;
			walker.frame = &s->data;
			walker.gpu_transform = !ge_sgx_gpu_failed && !GE_SGX_TEST_CPU_TRANSFORM;
			walker.ngpu_bad = ge_sgx_gpu_nbad;
			for (int b = 0; b < walker.ngpu_bad; b++)
				walker.gpu_bad[b] = ge_sgx_gpu_bad[b];
			return;
		}
	}
	frames_dropped++;
}

/*
	The framebuffers' pages, for the PSP's picture of an asked trace: the PSP
	side copies only the VRAM pages read here (textures), not the
	framebuffers this renderer draws itself (the picture was black). Wanted
	from the ask to the picture, then not any more.
*/
static struct { uint32_t fb, fbw, psm; } flipped[2] = { { 0xFFFFFFFFu, 0, 0 }, { 0xFFFFFFFFu, 0, 0 } };

static void wantPictures(void) {
	int k;
	for (k = 0; k < 2; k++) {
		if (flipped[k].fb == 0xFFFFFFFFu || !flipped[k].fbw)
			continue;
		uint32_t bytes = flipped[k].fbw * 272 * (flipped[k].psm == 3 ? 4 : 2);
		uint32_t p = flipped[k].fb / GE_SGX_VRAM_PAGE, last = (flipped[k].fb + bytes - 1) / GE_SGX_VRAM_PAGE;
		for (; p <= last && p < GE_SGX_VRAM_PAGES; p++) {
			uint32_t bit = 1u << (p % 32);
			if (!(vram_want[p / 32] & bit)) {
				vram_want[p / 32] |= bit;
				picture_pages[p / 32] |= bit;
				vram_want_dirty = 1;
			}
		}
	}
}

static void unwantPictures(void) {
	uint32_t i;
	for (i = 0; i < GE_SGX_VRAM_PAGES / 32; i++) {
		if (picture_pages[i]) {
			vram_want[i] &= ~picture_pages[i];
			picture_pages[i] = 0;
			vram_want_dirty = 1;
		}
	}
}

// Walker thread, at a display flip of fb
/*
	Asked (L + SELECT): the PSP's own picture of the traced frame, as its GE
	drew it into VRAM (psp_<ask>_<fb>.bmp, 480x272): what the game should
	look like, to compare with this renderer's dump of the same frame
*/
static void dumpPspPicture(uint32_t fb, uint32_t fbw, uint32_t psm) {
	static uint32_t asks = 0;
	static uint8_t bmp[54 + 480 * 272 * 4];
	if (fb >= VRAM_SIZE || fbw == 0 || fbw > 1024 || psm > 3)
		return;
	uint32_t bpp = psm == 3 ? 4 : 2;
	const uint8_t *src = geSgxPspPtr(VRAM_START + fb, fbw * 272 * bpp);
	if (!src)
		return;
	memset(bmp, 0, 54);
	uint32_t size = sizeof(bmp), v;
	bmp[0] = 'B'; bmp[1] = 'M';
	memcpy(bmp + 2, &size, 4);
	v = 54; memcpy(bmp + 10, &v, 4);
	v = 40; memcpy(bmp + 14, &v, 4);
	int32_t iw = 480, ih = -272;
	memcpy(bmp + 18, &iw, 4);
	memcpy(bmp + 22, &ih, 4);
	uint16_t s16 = 1; memcpy(bmp + 26, &s16, 2);
	s16 = 32; memcpy(bmp + 28, &s16, 2);
	uint32_t x, y;
	for (y = 0; y < 272; y++) {
		uint8_t *o = bmp + 54 + y * 480 * 4;
		for (x = 0; x < 480; x++) {
			uint32_t c; // 0xAABBGGRR
			if (bpp == 4) {
				memcpy(&c, src + (y * fbw + x) * 4, 4);
			} else {
				uint16_t s;
				memcpy(&s, src + (y * fbw + x) * 2, 2);
				c = geSgxDecodeColor16(psm, s);
			}
			o[x * 4 + 0] = (uint8_t)(c >> 16);
			o[x * 4 + 1] = (uint8_t)(c >> 8);
			o[x * 4 + 2] = (uint8_t)c;
			o[x * 4 + 3] = 0xFF;
		}
	}
	char name[64];
	snprintf(name, sizeof(name), "psp_%u_%X.bmp", (unsigned)++asks, (unsigned)fb);
	dumpWrite(name, bmp, sizeof(bmp));
	GeSgxLog("GeSgx: the PSP's picture of the traced frame: %s (fb %X width %u format %u)\n", name, (unsigned)fb, (unsigned)fbw, (unsigned)psm);
}

// An asked trace (L + SELECT) is waiting for its frame
static int trace_asked = 0;

static void publishSlot(uint32_t top, uint32_t fbw, uint32_t psm) {
	uint32_t addr = top & 0x0FFFFFFF;
	uint32_t fb = (addr >= VRAM_START && addr < VRAM_START + 0x00800000u) ? (addr & (VRAM_SIZE - 1)) : 0xFFFFFFFFu;
	if (fb != 0xFFFFFFFFu && fb != flipped[0].fb) {
		flipped[1] = flipped[0];
		flipped[0].fb = fb;
	}
	if (fb != 0xFFFFFFFFu) {
		flipped[0].fbw = fbw;
		flipped[0].psm = psm;
	}

	/*
		Diagnostics: a frame of the race traced into the log, about 10 s, 40 s and
		80 s into it. The lines wait in memory until the frame is done: written
		if it has many draws (the race), started again with the next one if not.
	*/
	// Diagnostics: VRAM's contents now and then (every 300th frame)
	static uint32_t probe_frames = 0;
	if (building && (++probe_frames % 300) == 0)
		probeVram();

	static uint32_t big_frames = 0, trace_tries = 0;
	/*
		An asked trace ends with the next frame walked, built or not (a walker
		behind walks frames without building them: the trace never ended)
	*/
	int walked_only = !(building && walker.frame);
	if (walker.trace && ((building && walker.frame) || trace_asked)) {
		uint32_t nd = walked_only ? 0 : building->data.ndraws;
		if (nd >= 1000 || (trace_asked && (nd || (walked_only && walker.trace_lines > 1)))) {
			// The renderer traces the same frame (once: a frame can be rendered again)
			static uint32_t asked_seq = 0;
			if (trace_asked && !walked_only)
				building->data.asked = ++asked_seq;
			if (walked_only)
				GeSgxLog("GeSgx T: (the walker was behind: this frame was walked, not built)\n");
			trace_asked = 0;
			walker.dump = 0;
			uint32_t len;
			const char *text = geSgxBuildTraceText(&len);
			GeSgxLog("GeSgx T: begin (%u lines)\n", (unsigned)walker.trace_lines);
			GeSgxLogText(text, len);
			GeSgxLog("GeSgx T: end, %u draws, shown fb %X, %u aliases\n", (unsigned)nd, (unsigned)fb,
				(unsigned)walker.naliases);
			dumpFlush();
			if (fb != 0xFFFFFFFFu)
				dumpPspPicture(fb, fbw, psm);
			unwantPictures();
			walker.trace = 0;
		} else if (++trace_tries >= 60) {
			GeSgxLog("GeSgx T: no frame with 1000 draws\n");
			walker.trace = 0;
			walker.dump = 0;
			dumpFlush();
			unwantPictures();
		}
		geSgxBuildTraceClear(&walker);
	}
	// Asked (L + SELECT): the next frame built, whatever its size
	// (over an automatic trace still waiting for a frame of 1000 draws: it kept the ask waiting)
	if (ge_sgx_trace_walker) {
		ge_sgx_trace_walker = 0;
		static int dump_asks = 0;
		trace_asked = 1;
		walker.trace = 1;
		walker.dump = ++dump_asks; // each ask its own textures written
		wantPictures();
		trace_tries = 0;
		geSgxBuildTraceClear(&walker);
	}
	if (building && walker.frame && building->data.ndraws >= 300) {
		big_frames++;
		if (walker.log_fn && !walker.trace && (big_frames == 300 || big_frames == 1200 || big_frames == 2400)) {
			walker.trace = 1;
			trace_tries = 0;
			geSgxBuildTraceClear(&walker);
		}
	}

	// Walked but not built (the walker was behind): nothing to show, the slot is free again
	if (building && !walker.frame) {
		building->state = SLOT_FREE;
		building = NULL;
	}

	if (building) {
		GeSgxFrameData *f = &building->data;
		geSgxBuildFinish(&walker);
		toGpu(f->vertices_gpu, f->vertices, f->nvertices * sizeof(GeSgxVertex));
		toGpu(f->indices_gpu, f->indices, f->nindices * sizeof(uint16_t));
		toGpu(f->uniforms_gpu, f->uniforms, f->nuniforms);
		building->fb = fb;
		building->naliases = walker.naliases;
		memcpy(building->aliases, walker.aliases, sizeof(building->aliases));
		building->seq = ++frame_seq_counter;
		__sync_synchronize();
		building->state = SLOT_READY;
		frames_published++;
	}
	walker.naliases = 0;
	geSgxTexFrameDone();
	acquireSlot();
}

int GeSgxTakeFrame(GeSgxFrameRef *ref) {
	int i, best = -1;
	uint32_t best_seq = 0;
	for (i = 0; i < FRAME_SLOTS; i++) {
		if (slots[i].state == SLOT_READY && (best < 0 || slots[i].seq > best_seq)) {
			best = i;
			best_seq = slots[i].seq;
		}
	}
	if (best < 0)
		return 0;

	// Older published frames are not shown any more
	for (i = 0; i < FRAME_SLOTS; i++) {
		if (i != best && slots[i].state == SLOT_READY)
			slots[i].state = SLOT_FREE;
	}

	FrameSlot *s = &slots[best];
	__sync_synchronize();
	s->state = SLOT_RENDERING;
	ref->slot = best;
	ref->data = &s->data;
	ref->fb = s->fb;
	ref->aliases = s->aliases;
	ref->naliases = s->naliases;
	return 1;
}

void GeSgxFrameRendered(const GeSgxFrameRef *ref) {
	FrameSlot *s = &slots[ref->slot];
	s->rendered_tick = draw_tick;
	__sync_synchronize();
	s->state = SLOT_RENDERED;
}

/*
	The newest frame the GPU is done with: one rendered GPU_DONE_TICKS draw
	loops ago (the rule the slots are reused by). A texture may be freed
	once the GPU is done with the frame that was being built when it died:
	counting flips instead (KEEP_FRAMES) freed textures the GPU still read
	while the walker caught up, flips then come much faster than frames are
	drawn.
*/
volatile uint32_t ge_sgx_gpu_done_seq = 0;

uint32_t GeSgxBuildingSeq(void) {
	return frame_seq_counter + 1;
}

void GeSgxDrawTick(void) {
	draw_tick++;
	int i;
	for (i = 0; i < FRAME_SLOTS; i++) {
		FrameSlot *s = &slots[i];
		if (s->state == SLOT_RENDERED && draw_tick - s->rendered_tick >= GPU_DONE_TICKS && (int32_t)(s->seq - ge_sgx_gpu_done_seq) > 0)
			ge_sgx_gpu_done_seq = s->seq;
	}
}

// Published frame (seqlock: odd while it is written)
static volatile uint32_t frame_seq = 0;
static GeSgxFrame frame;
static volatile SceUInt64 last_flip_time = 0;

static uint32_t cdram_kb = 0, ram_kb = 0;

// Threads of the emulator and ours, to see which cores are busy
#define MAX_TRACKED_THREADS 48
static struct {
	SceUID uid;
	char name[32];
	SceUInt64 last_run;
	volatile int valid;
} tracked[MAX_TRACKED_THREADS];
static int n_tracked = 0; // slots taken

void GeSgxTrackThread(SceUID uid, const char *name) {
	if (uid < 0 || config.sgx_renderer == GE_SGX_MODE_OFF)
		return;
	// Threads may be created concurrently: take a slot atomically
	int i = __sync_fetch_and_add(&n_tracked, 1);
	if (i >= MAX_TRACKED_THREADS)
		return;
	tracked[i].uid = uid;
	strncpy(tracked[i].name, name, sizeof(tracked[i].name) - 1);
	tracked[i].name[sizeof(tracked[i].name) - 1] = '\0';
	tracked[i].last_run = 0;
	__sync_synchronize();
	tracked[i].valid = 1;
}

// CPU use of each thread since the last report, the busy ones only
static void reportThreads(SceUInt64 elapsed_us) {
	char line[480];
	int len = 0, i, n = n_tracked;
	if (n > MAX_TRACKED_THREADS)
		n = MAX_TRACKED_THREADS;

	len += snprintf(line + len, sizeof(line) - len, "GeSgx threads (%%cpu core mask prio):");
	for (i = 0; i < n; i++) {
		if (!tracked[i].valid)
			continue;
		SceKernelThreadInfo info;
		memset(&info, 0, sizeof(info));
		info.size = sizeof(info);
		if (sceKernelGetThreadInfo(tracked[i].uid, &info) < 0)
			continue;

		SceUInt64 run = info.runClocks;
		SceUInt64 delta = run - tracked[i].last_run;
		int first = tracked[i].last_run == 0;
		tracked[i].last_run = run;
		if (first || elapsed_us == 0)
			continue;

		uint32_t permille = (uint32_t)(delta * 1000 / elapsed_us);
		if (permille < 5)
			continue;

		int k = snprintf(line + len, sizeof(line) - len, " %s %u.%u %d %X %X;",
			tracked[i].name, (unsigned)(permille / 10), (unsigned)(permille % 10),
			(int)info.lastExecutedCpuId, (unsigned)info.currentCpuAffinityMask, (unsigned)info.currentPriority);
		if (k < 0 || len + k >= (int)sizeof(line) - 2) {
			GeSgxLog("%s\n", line);
			len = snprintf(line, sizeof(line), "GeSgx threads (cont):");
			k = snprintf(line + len, sizeof(line) - len, " %s %u.%u %d %X %X;",
				tracked[i].name, (unsigned)(permille / 10), (unsigned)(permille % 10),
				(int)info.lastExecutedCpuId, (unsigned)info.currentCpuAffinityMask, (unsigned)info.currentPriority);
			if (k < 0)
				continue;
		}
		len += k;
	}
	GeSgxLog("%s\n", line);
}

static void writeAck(uint32_t session, uint32_t read) {
	uint8_t *block = (uint8_t *)ScePspemuConvertAddress(ADRENALINE_ADDRESS, KERMIT_OUTPUT_MODE, ADRENALINE_SIZE);
	if (!block)
		return;
	GeSgxAck *ack = (GeSgxAck *)(block + (GE_SGX_ACK_ADDRESS - ADRENALINE_ADDRESS));
	ack->session = session;
	ack->read = read;
	ScePspemuWritebackCache(ack, sizeof(GeSgxAck));
	if (vram_want_dirty) {
		vram_want_dirty = 0;
		GeSgxVramWant *want = (GeSgxVramWant *)(block + (GE_SGX_VRAM_WANT_ADDRESS - ADRENALINE_ADDRESS));
		memcpy((void *)want->bits, vram_want, sizeof(vram_want));
		ScePspemuWritebackCache(want, sizeof(GeSgxVramWant));
	}
}

void GeSgxLogText(const char *text, uint32_t len) {
	if (!len)
		return;
	SceUID fd = sceIoOpen(LOG_PATH, SCE_O_WRONLY | SCE_O_CREAT | SCE_O_APPEND, 0777);
	if (fd >= 0) {
		sceIoWrite(fd, text, len);
		sceIoClose(fd);
	}
}

void GeSgxLog(const char *fmt, ...) {
	char buf[512];
	va_list ap;
	va_start(ap, fmt);
	int n = vsnprintf(buf, sizeof(buf), fmt, ap);
	va_end(ap);
	if (n <= 0)
		return;
	if (n >= (int)sizeof(buf))
		n = sizeof(buf) - 1;

	SceUID fd = sceIoOpen(LOG_PATH, SCE_O_WRONLY | SCE_O_CREAT | SCE_O_APPEND, 0777);
	if (fd >= 0) {
		sceIoWrite(fd, buf, n);
		sceIoClose(fd);
	}
}

// How much memory the renderer could still get, in 1 MB blocks (64 MB at most:
// the blocks are taken from the emulator for a moment)
/*
	Free memory of the process, without allocating it (probing took all of
	it for a moment: an allocation of the emulator at that moment failed).
*/
uint32_t GeSgxFreeKB(SceKernelMemBlockType type) {
	SceKernelFreeMemorySizeInfo info;
	memset(&info, 0, sizeof(info));
	info.size = sizeof(info);
	if (sceKernelGetFreeMemorySize(&info) < 0)
		return 0;
	if (type == SCE_KERNEL_MEMBLOCK_TYPE_USER_CDRAM_RW)
		return (uint32_t)info.size_cdram / 1024;
	return (uint32_t)info.size_user / 1024;
}

static void publish(const GeSgxRect *rects, int nrects, const GeSgxStats *stats, const GeSgxFrame *rates) {
	frame_seq++;
	__sync_synchronize();

	frame.flips++;
	frame.nrects = nrects;
	memcpy(frame.rects, rects, sizeof(GeSgxRect) * nrects);
	frame.stats = *stats;
	frame.fps = rates->fps;
	frame.events = rates->events;
	frame.walk_us = rates->walk_us;
	frame.walk_max_us = rates->walk_max_us;
	frame.invalidates = rates->invalidates;
	frame.resyncs = rates->resyncs;
	frame.sync_waits = rates->sync_waits;
	frame.sync_timeouts = rates->sync_timeouts;
	frame.sync_max_us = rates->sync_max_us;
	frame.cdram_kb = cdram_kb;
	frame.ram_kb = ram_kb;

	__sync_synchronize();
	frame_seq++;
}

int GeSgxGetFrame(GeSgxFrame *out) {
	// Never wait for the writer: a few tries, then take the copy as it is
	// (at worst statistics of two frames mixed)
	int tries;
	for (tries = 0; tries < 4; tries++) {
		uint32_t seq = frame_seq;
		__sync_synchronize();
		memcpy(out, &frame, sizeof(GeSgxFrame));
		__sync_synchronize();
		if (!(seq & 1) && seq == frame_seq)
			break;
	}

	if (out->nrects < 0 || out->nrects > GE_SGX_FRAME_RECTS)
		out->nrects = 0;

	if (out->flips == 0)
		return 0;
	return (sceKernelGetProcessTimeWide() - last_flip_time) < 500 * 1000;
}

static uint32_t sumWords(const volatile uint32_t *p, int n) {
	uint32_t sum = 0;
	int i;
	for (i = 0; i < n; i++)
		sum += p[i];
	return sum;
}

/*
	Cost of the ways to read PSP RAM, for the choice of the renderer's
	memory access. 64 pages of 4 KB of the game's memory, times in us.
*/
static void benchmarkMemory(void) {
	enum { PAGES = 64, PAGE = 4096, WORDS = PAGE / 4 };
	const uint32_t first = 0x88800000u;
	uint32_t sum = 0;
	uint32_t *buf = NULL;
	SceUID buf_id = sceKernelAllocMemBlock("SceGeSgxBench", SCE_KERNEL_MEMBLOCK_TYPE_USER_RW, PAGES * PAGE, NULL);
	if (buf_id < 0)
		return;
	sceKernelGetMemBlockBase(buf_id, (void **)&buf);
	int i;
	SceUInt64 t;

	const uint8_t *cached = (const uint8_t *)ScePspemuConvertAddress(first, KERMIT_INPUT_MODE, PAGES * PAGE);
	const uint8_t *plain = (const uint8_t *)ScePspemuConvertAddress(first, 0, 0);
	GeSgxLog("GeSgx bench: cached map %08X, plain map %08X, CONVERT_ADDRESS %08X\n",
		(unsigned)(uintptr_t)cached, (unsigned)(uintptr_t)plain, (unsigned)CONVERT_ADDRESS(first));
	if (!cached || !plain) {
		sceKernelFreeMemBlock(buf_id);
		return;
	}

	// the same memory?
	GeSgxLog("GeSgx bench: same data %d\n", memcmp(cached, plain, 256) == 0);

	t = sceKernelGetProcessTimeWide();
	for (i = 0; i < 1000; i++)
		sceKernelGetProcessTimeWide();
	uint32_t t_clock = (uint32_t)(sceKernelGetProcessTimeWide() - t);

	// invalidate only, one call per page
	t = sceKernelGetProcessTimeWide();
	for (i = 0; i < PAGES; i++)
		ScePspemuConvertAddress(first + i * PAGE, KERMIT_INPUT_MODE, PAGE);
	uint32_t t_inval_page = (uint32_t)(sceKernelGetProcessTimeWide() - t);

	// invalidate the 256 KB in one call
	t = sceKernelGetProcessTimeWide();
	ScePspemuConvertAddress(first, KERMIT_INPUT_MODE, PAGES * PAGE);
	uint32_t t_inval_all = (uint32_t)(sceKernelGetProcessTimeWide() - t);

	// read after invalidation (cache misses)
	ScePspemuConvertAddress(first, KERMIT_INPUT_MODE, PAGES * PAGE);
	t = sceKernelGetProcessTimeWide();
	sum += sumWords((const volatile uint32_t *)cached, PAGES * WORDS);
	uint32_t t_read_cold = (uint32_t)(sceKernelGetProcessTimeWide() - t);

	// read again (cache hits as far as 256 KB stay in the cache)
	t = sceKernelGetProcessTimeWide();
	sum += sumWords((const volatile uint32_t *)cached, PAGES * WORDS);
	uint32_t t_read_warm = (uint32_t)(sceKernelGetProcessTimeWide() - t);

	// one page, again and again: L1 hits
	t = sceKernelGetProcessTimeWide();
	for (i = 0; i < PAGES; i++)
		sum += sumWords((const volatile uint32_t *)cached, WORDS);
	uint32_t t_read_l1 = (uint32_t)(sceKernelGetProcessTimeWide() - t);

	// through the other mapping (uncached?)
	t = sceKernelGetProcessTimeWide();
	sum += sumWords((const volatile uint32_t *)plain, PAGES * WORDS);
	uint32_t t_read_plain = (uint32_t)(sceKernelGetProcessTimeWide() - t);

	// DMA into our buffer: one copy per page, then all at once
	t = sceKernelGetProcessTimeWide();
	for (i = 0; i < PAGES; i++)
		sceDmacMemcpy(&buf[i * WORDS], cached + i * PAGE, PAGE);
	uint32_t t_dma_page = (uint32_t)(sceKernelGetProcessTimeWide() - t);

	t = sceKernelGetProcessTimeWide();
	sceDmacMemcpy(buf, cached, PAGES * PAGE);
	uint32_t t_dma_all = (uint32_t)(sceKernelGetProcessTimeWide() - t);

	t = sceKernelGetProcessTimeWide();
	sum += sumWords(buf, PAGES * WORDS);
	uint32_t t_read_buf = (uint32_t)(sceKernelGetProcessTimeWide() - t);

	// memcpy with the CPU from the invalidated mapping
	ScePspemuConvertAddress(first, KERMIT_INPUT_MODE, PAGES * PAGE);
	t = sceKernelGetProcessTimeWide();
	memcpy(buf, cached, PAGES * PAGE);
	uint32_t t_memcpy = (uint32_t)(sceKernelGetProcessTimeWide() - t);

	// memcpy in blocks of 64, 256, 1024 bytes (the walker's way)
	uint32_t t_blocks[3];
	static const int block_sizes[3] = { 64, 256, 1024 };
	int k;
	for (k = 0; k < 3; k++) {
		int bs = block_sizes[k], n = PAGES * PAGE / bs;
		t = sceKernelGetProcessTimeWide();
		for (i = 0; i < n; i++)
			memcpy((uint8_t *)buf + (i & 15) * bs, cached + i * bs, bs);
		t_blocks[k] = (uint32_t)(sceKernelGetProcessTimeWide() - t);
	}

	GeSgxLog("GeSgx bench (us, 256 KB in blocks of 64/256/1024): %u %u %u\n",
		(unsigned)t_blocks[0], (unsigned)t_blocks[1], (unsigned)t_blocks[2]);

	GeSgxLog("GeSgx bench (us, 64 x 4 KB): clock x1000 %u | inval per page %u, all %u | read cold %u, warm %u, L1 %u, plain map %u | dma per page %u, all %u, read copy %u | memcpy %u (sum %08X)\n",
		(unsigned)t_clock, (unsigned)t_inval_page, (unsigned)t_inval_all,
		(unsigned)t_read_cold, (unsigned)t_read_warm, (unsigned)t_read_l1, (unsigned)t_read_plain,
		(unsigned)t_dma_page, (unsigned)t_dma_all, (unsigned)t_read_buf, (unsigned)t_memcpy, (unsigned)sum);

	sceKernelFreeMemBlock(buf_id);
}

static int GeSgxThread(SceSize args, void *argp) {
	uint32_t seen_seq = 0;
	uint32_t ring_addr = 0, session = 0, mask = 0, entries = 0;
	uint32_t shadow_addr = 0, shadow_size = 0;
	uint32_t read = 0, last_lost = 0;
	uint32_t logged_types = 0, slow_logs = 0, bad_logged = 0, bad_logs = 0;
	uint32_t frame_stats_range = 0, frame_stats_cull_off = 0;
	uint32_t frame_stats_3d = 0, frame_stats_other = 0, frame_stats_draws = 0, frame_stats_verts = 0, frame_stats_indices = 0;
	uint32_t frame_stats_gpu = 0, frame_stats_raw = 0, frame_stats_uni = 0, frame_stats_cpu[3] = { 0 };
	uint32_t built_vertices = 0, built_states = 0, built_frames = 0, built_us[7] = { 0 }; // per log interval
	uint32_t interval_walk_us = 0; // all of the walker's time in it (the build included)
	SceUInt64 session_start = 0;

	// Per second
	SceUInt64 second_start = sceKernelGetProcessTimeWide();
	SceUInt64 log_start = second_start;
	SceUInt64 thread_report = second_start;
	uint32_t flips = 0, events = 0, walk_us = 0, walk_max_us = 0, fetches_start = block_copies;
	uint32_t calls_start = block_calls;
	GeSgxFrame rates;
	memset(&rates, 0, sizeof(rates));

	GeSgxStats frame_stats;
	GeSgxRect rects[GE_SGX_FRAME_RECTS];

	cdram_kb = GeSgxFreeKB(SCE_KERNEL_MEMBLOCK_TYPE_USER_CDRAM_RW);
	ram_kb = GeSgxFreeKB(SCE_KERNEL_MEMBLOCK_TYPE_USER_RW);
	GeSgxLog("GeSgx: free CDRAM %u KB, free RAM %u KB\n", (unsigned)cdram_kb, (unsigned)ram_kb);

	benchmarkMemory();

	ram_map = (const uint8_t *)ScePspemuConvertAddress(0x80000000u | RAM_START, 0, 0);
	// pspemu's window for VRAM, which the PSP side keeps filled (see GE_SGX_VRAM_SHADOW)
	vram_map = (const uint8_t *)ScePspemuConvertAddress(VRAM_START, 0, 0);
	if (!ram_map || !vram_map || !initBlockCache()) {
		GeSgxLog("GeSgx: no memory map (%08X %08X) or block cache\n", (unsigned)(uintptr_t)ram_map, (unsigned)(uintptr_t)vram_map);
		return 0;
	}
	GeSgxLog("GeSgx: RAM map %08X, VRAM map %08X\n", (unsigned)(uintptr_t)ram_map, (unsigned)(uintptr_t)vram_map);

	/*
		The GPU reads PSP RAM where it is: vertex data is not copied. The
		user part (from 0x08800000), as much as maps: 56, 24 or 8 MB.
	*/
	if (!direct_size) {
		static const uint32_t sizes[] = { 56u << 20, 24u << 20, 8u << 20 };
		int res = -1;
		unsigned k;
		for (k = 0; k < sizeof(sizes) / sizeof(sizes[0]); k++) {
			res = sceGxmMapMemory((void *)(ram_map + 0x00800000u), sizes[k], SCE_GXM_MEMORY_ATTRIB_READ);
			if (res >= 0) {
				direct_size = sizes[k];
				break;
			}
		}
		GeSgxLog("GeSgx: GPU map of PSP RAM: %u MB (%08X)\n", (unsigned)(direct_size >> 20), (unsigned)res);
	}

	geSgxWalkerInit(&walker, blockFn, NULL);

	if (config.sgx_renderer >= GE_SGX_MODE_DRAW) {
		geSgxPoolInit(); // before the texture cache, which takes most of it
		int slots_ok = initSlots();
		int tex_ok = geSgxTexInit();
		GeSgxLog("GeSgx: frames %d, texture cache %d, free RAM now %u KB\n", slots_ok, tex_ok,
			(unsigned)GeSgxFreeKB(SCE_KERNEL_MEMBLOCK_TYPE_USER_RW));
		if (slots_ok && tex_ok) {
			walker.texture_fn = geSgxTexLookup;
			walker.bulk_fn = bulkFn;
			walker.vram_copy_fn = vramCopyFn;
			walker.dump_fn = dumpFn;
			clockInit(&walker);
			walker.log_fn = GeSgxLog;
			walker.gpu_transform = !GE_SGX_TEST_CPU_TRANSFORM;
			GeSgxLog("GeSgx: transform on the %s\n", GE_SGX_TEST_CPU_TRANSFORM ? "CPU (test)" : "GPU");
			walker.direct_start = RAM_START + 0x00800000u;
			walker.direct_size = direct_size;
			acquireSlot();
		}
	}

	int walk_late = 0, synced_since_flip = 0;
	while (1) {
		if (start_seq != seen_seq) {
			seen_seq = start_seq;
			ring_addr = start_ring;
			session = start_session;
			entries = start_entries;
			mask = entries - 1;
			read = 0;
			last_lost = 0;
			geSgxWalkerReset(&walker);
			// A new game: the programs of the one before given back to the shader patcher
			if (session > 1)
				ge_sgx_render_flush = 1;
			geSgxBuildReset();
			if (walker.texture_fn) {
				geSgxTexReset();
				if (building) {
					building->data.ndraws = 0;
					building->data.nluts = 0;
					building->data.nvertices = 0;
					building->data.nindices = 0;
					building->data.nuniforms = 0;
					building->data.nraw = 0;
				}
			}
			rates.resyncs = 0;
			walk_late = 0;
			synced_since_flip = 0;
			logged_types = 0;
			slow_logs = 0;
			session_start = sceKernelGetProcessTimeWide();
			GeSgxLog("GeSgx: ring %08X, %u entries, session %u\n", (unsigned)ring_addr, (unsigned)entries, (unsigned)session);
			{
				const GeSgxRing *r0 = (const GeSgxRing *)ScePspemuConvertAddress(ring_addr, KERMIT_INPUT_MODE, RING_EVENTS_OFFSET);
				shadow_addr = r0 ? r0->shadow : 0;
				shadow_size = r0 ? r0->shadow_size : 0;
				noverlays = 0;
				GeSgxLog("GeSgx: copies of the lists at the signals: %08X, %u KB\n", (unsigned)shadow_addr, (unsigned)(shadow_size / 1024));
			}
		}

		if (!ring_addr) {
			sceKernelDelayThread(10 * 1000);
			continue;
		}

		const GeSgxRing *ring = (const GeSgxRing *)ScePspemuConvertAddress(ring_addr, KERMIT_INPUT_MODE, RING_EVENTS_OFFSET);
		if (!ring || ring->magic != GE_SGX_MAGIC || ring->session != session) {
			// The game has exited: the memory belongs to someone else now
			sceKernelDelayThread(10 * 1000);
			continue;
		}

		uint32_t write = ring->write;
		uint32_t lost = ring->lost;
		rates.sync_waits = ring->sync_waits;
		rates.sync_timeouts = ring->sync_timeouts;
		rates.sync_max_us = ring->sync_max_us;
		rates.vram_copies = ring->vram_copies;
		rates.vram_copy_us = ring->vram_copy_us;
		rates.vram_copy_max_us = ring->vram_copy_max_us;

		if (lost != last_lost || (write - read) > entries) {
			// Events are missing: start over from the next ones
			last_lost = lost;
			read = write;
			geSgxWalkerResync(&walker);
			rates.resyncs++;
			writeAck(session, read);
			continue;
		}

		if (write == read) {
			// The PSP side may wait in a signal handler for the walk: looked at more often then
			sceKernelDelayThread(walker.hold_signals ? IDLE_DELAY_SIGNALS_US : IDLE_DELAY_US);
			goto per_second;
		}

		uint32_t n = write - read;
		if (n > EVENTS_PER_BATCH)
			n = EVENTS_PER_BATCH;

		// Invalidate the events, in 64 byte lines (two events)
		uint32_t first = read & mask & ~1u;
		uint32_t last = ((read + n - 1) & mask) | 1u;
		if (first <= last) {
			ScePspemuConvertAddress(ring_addr + RING_EVENTS_OFFSET + first * sizeof(GeSgxEvent), KERMIT_INPUT_MODE, (last - first + 1) * sizeof(GeSgxEvent));
		} else {
			// wraps around
			ScePspemuConvertAddress(ring_addr + RING_EVENTS_OFFSET + first * sizeof(GeSgxEvent), KERMIT_INPUT_MODE, (entries - first) * sizeof(GeSgxEvent));
			ScePspemuConvertAddress(ring_addr + RING_EVENTS_OFFSET, KERMIT_INPUT_MODE, (last + 1) * sizeof(GeSgxEvent));
		}

		uint32_t i;
		int gap = 0;
		for (i = 0; i < n; i++) {
			const GeSgxEvent *e = &ring->events[(read + i) & mask];
			if (e->seq != read + i) {
				gap = 1;
				break;
			}

			SceUInt64 t0 = sceKernelGetProcessTimeWide();
			epoch++;

			if (e->type < 32 && !(logged_types & (1u << e->type))) {
				logged_types |= 1u << e->type;
				GeSgxLog("GeSgx: first event %u: %08X %08X %08X %08X (walker: %d lists)\n",
					(unsigned)e->type, (unsigned)e->a, (unsigned)e->b, (unsigned)e->c, (unsigned)e->d, walker.count);
			}

			switch (e->type) {
				/*
					Lists are walked late, when the game waits for the GE
					(GE_SGX_EVENT_SYNC): the GE is done, the memory is what it
					read. Walked as they came, the walk ran ahead of the GE and
					read what the game writes after queuing them: The 3rd
					Birthday's signal handler writes the matrices of its HUD,
					fog and particles after the signals (1852 list words changed
					between the walk and the flip; off the screen with the old
					ones); Black Rock Shooter draws its glyphs into its text
					texture after queuing the draws that use them (empty there
					when it was read: letters in pieces). A game that never
					waits (no sync event yet in this game) has its lists walked
					as they come.
				*/
				case GE_SGX_EVENT_LIST:
					geSgxWalkerList(&walker, (int)e->a, e->b, e->c, (int)e->d);
					if (!walk_late) {
						geSgxWalkerRun(&walker);
						noteListDone();
					}
					break;
				case GE_SGX_EVENT_STALL:
					geSgxWalkerStall(&walker, (int)e->a, e->b);
					if (!walk_late) {
						geSgxWalkerRun(&walker);
						noteListDone();
					}
					break;
				case GE_SGX_EVENT_SYNC:
					if (!walk_late)
						GeSgxLog("GeSgx: the game waits for the GE: lists walked at its waits from now on\n");
					walk_late = 1;
					synced_since_flip = 1;
					walkAll();
					break;
				case GE_SGX_EVENT_SIGNAL:
					// The GE at a signal, its handler about to run: the lists up to it (the PSP side waits for this)
					if (!walker.hold_signals)
						GeSgxLog("GeSgx: the GE's signal handlers reported: lists walked up to each signal before its handler runs\n");
					{
						/*
							The commands since the signal before, as the GE read them
							(the game's handler writes over them after): over the
							memory while the walk gets to this signal
						*/
						uint32_t words = e->d >> 16, off = e->d & 0xFFFF;
						if (words && shadow_addr && (off + words) * 4 <= shadow_size) {
							const uint32_t *rec = (const uint32_t *)ScePspemuConvertAddress(shadow_addr + off * 4, KERMIT_INPUT_MODE, words * 4);
							if (rec && takeOverlays(rec, words, e->seq)) {
								overlays_used++;
								overlayBlocksStale(); // read again, with them
							} else {
								overlays_stale++;
							}
						}
					}
					if (e->c)
						geSgxWalkerFinish(&walker, e->b);
					else
						geSgxWalkerSignal(&walker, e->b);
					noteListDone();
					if (noverlays) {
						overlayBlocksStale(); // the memory as it is again
						noverlays = 0;
					}
					// Read up to here: the PSP side waits for it in the GE's interrupt
					writeAck(session, read + i + 1);
					break;
				case GE_SGX_EVENT_REGS:
					geSgxWalkerHardwareReg(&walker, e->a, e->b);
					geSgxWalkerHardwareReg(&walker, e->c, e->d);
					break;
				case GE_SGX_EVENT_STATE:
					// Read with the GE idle: the lists it has done first (queued, not walked yet)
					if (walker.count)
						walkAll();
					geSgxWalkerHardwareState(&walker, e->a, e->b);
					geSgxWalkerHardwareState(&walker, e->c, e->d);
					break;
				case GE_SGX_EVENT_DEQUEUE:
					geSgxWalkerDequeue(&walker, (int)e->a);
					geSgxWalkerRun(&walker);
					noteListDone();
					break;
				case GE_SGX_EVENT_FLIP: {
					// Pages long unread: not copied any more
					if ((++vram_flips & 31) == 0) {
						uint32_t p;
						for (p = 0; p < GE_SGX_VRAM_PAGES; p++) {
							uint32_t bit = 1u << (p % 32);
							if ((vram_want[p / 32] & bit) && !(picture_pages[p / 32] & bit) &&
							    (uint16_t)(vram_flips - vram_used[p]) > VRAM_UNWANT_FLIPS) {
								vram_want[p / 32] &= ~bit;
								vram_want_dirty = 1;
							}
						}
					}
					// No wait for the GE in this frame: its lists walked now (the ones after a wait are the next frame's)
					if ((!walk_late || !synced_since_flip) && walker.count)
						walkAll();
					synced_since_flip = 0;
					/*
						Asked traces only: an automatic one waiting for a frame of
						1000 draws wrote these lines at every flip (up to 600
						frames at 2 to 7 frames a second, The 3rd Birthday and
						Black Rock Shooter)
					*/
					if (walker.trace && trace_asked && scan_end) {
						ListScan now;
						scanList(scan_start, scan_end, &now, 1);
						GeSgxLog("GeSgx T: list scan %X-%X: walker's top level prims %u calls %u signals %u | when done %u words, prims %u calls %u jumps %u signals %u nops %u sum %08X | "
							"at the flip prims %u calls %u jumps %u signals %u nops %u sum %08X\n", (unsigned)scan_start, (unsigned)scan_end,
							(unsigned)walker.done_prims, (unsigned)walker.done_calls, (unsigned)walker.done_signals,
							(unsigned)scan_done.words, (unsigned)scan_done.prims, (unsigned)scan_done.calls, (unsigned)scan_done.jumps,
							(unsigned)scan_done.signals, (unsigned)scan_done.nops, (unsigned)scan_done.sum, (unsigned)now.prims, (unsigned)now.calls,
							(unsigned)now.jumps, (unsigned)now.signals, (unsigned)now.nops, (unsigned)now.sum);
						scanDiff();
						scan_end = 0;
					}
					geSgxBuildTraceNote(&walker, "flip: %X (%d lists queued)", (unsigned)e->a, walker.count);
					for (int q = 0; q < walker.count && q < 4; q++)
						geSgxBuildTraceNote(&walker, "flip: list qid %X from %X at %X, stall %X, depth %d", (unsigned)walker.lists[q].qid,
							(unsigned)walker.lists[q].start, (unsigned)walker.lists[q].pc, (unsigned)walker.lists[q].stall, walker.lists[q].depth);
					int nrects = geSgxWalkerFlip(&walker, e->a, rects, GE_SGX_FRAME_RECTS, &frame_stats);
					if (walker.texture_fn) {
						if (building && walker.frame) {
							frame_stats_3d = building->data.draws_3d;
							frame_stats_gpu = building->data.prims_gpu;
							frame_stats_raw = building->data.nraw;
							frame_stats_uni = building->data.nuniforms;
							memcpy(frame_stats_cpu, building->data.cpu_reason, sizeof(frame_stats_cpu));
							frame_stats_indices = building->data.nindices;
							frame_stats_other = building->data.skipped_other;
							frame_stats_range = building->data.skipped_range;
							frame_stats_cull_off = building->data.cull_off;
							frame_stats_draws = building->data.ndraws;
							frame_stats_verts = building->data.nvertices;
							built_vertices += building->data.vertices_built;
							built_states += building->data.state_builds;
							// The walker's clock: microseconds or cycles
							built_us[0] += building->data.us_state / clock_ticks_per_us;
							built_us[1] += building->data.us_texture / clock_ticks_per_us;
							built_us[2] += building->data.us_vertices / clock_ticks_per_us;
							built_us[3] += building->data.us_indices / clock_ticks_per_us;
							built_us[4] += building->data.us_vindex / clock_ticks_per_us;
							built_us[5] += building->data.us_vuni / clock_ticks_per_us;
							built_us[6] += building->data.us_clut / clock_ticks_per_us;
							built_frames++;
						}
						publishSlot(e->a, e->b, e->c);

						/*
							Behind: the next frame is already queued (another flip waits in
							the ring). Its lists are walked to stay in step, but not built,
							so the walker catches up instead of falling further behind.
						*/
						int behind = (write - read) > n;
						uint32_t j;
						for (j = i + 1; j < n && !behind; j++) {
							if (ring->events[(read + j) & mask].type == GE_SGX_EVENT_FLIP)
								behind = 1;
						}
						if (behind && walker.frame) {
							walker.frame = NULL;
							frames_skipped++;
						}
					}
					publish(rects, nrects, &frame_stats, &rates);
					last_flip_time = sceKernelGetProcessTimeWide();
					flips++;
					break;
				}
				default:
					break;
			}

			uint32_t dt = (uint32_t)(sceKernelGetProcessTimeWide() - t0);
			if (dt > 20 * 1000 && slow_logs < 20) {
				slow_logs++;
				GeSgxLog("GeSgx: slow event %u: %u us, walker %d lists, commands %u, bad %u\n",
					(unsigned)e->type, (unsigned)dt, walker.count, (unsigned)walker.stats.commands, (unsigned)walker.stats.bad);
			}
			walk_us += dt;
			interval_walk_us += dt;
			if (dt > walk_max_us)
				walk_max_us = dt;
		}

		if (walker.last_bad.count != bad_logged && bad_logs < 10) {
			bad_logged = walker.last_bad.count;
			bad_logs++;
			uint32_t op = 0;
			const uint8_t *b = blockFn(NULL, walker.last_bad.pc & ~(uint32_t)(GE_SGX_BLOCK_SIZE - 1));
			if (b)
				memcpy(&op, b + (walker.last_bad.pc & (GE_SGX_BLOCK_SIZE - 1) & ~3u), 4);
			GeSgxLog("GeSgx: bad list %d: start %08X stopped at %08X (op %08X), call depth %u\n",
				walker.last_bad.qid, (unsigned)walker.last_bad.start, (unsigned)walker.last_bad.pc,
				(unsigned)op, (unsigned)walker.last_bad.depth);
		}

		read += i;
		events += i;
		if (gap) {
			read = write;
			geSgxWalkerResync(&walker);
			rates.resyncs++;
		}
		writeAck(session, read);

per_second:;
		SceUInt64 now = sceKernelGetProcessTimeWide();
		if (now - second_start >= 1000 * 1000) {
			rates.fps = flips;
			rates.events = events;
			rates.walk_us = walk_us;
			rates.walk_max_us = walk_max_us;
			rates.invalidates = block_copies - fetches_start;

			if ((flips || events) && (now - log_start >= LOG_INTERVAL_US || now - session_start < 30 * 1000 * 1000)) {
				log_start = now;
				GeSgxLog("GeSgx: t %u fps %u events %u walk %u us (max %u) blocks %u lookups %u | lists %u cmds %u prims %u verts %u clears %u signals %u held %u bad %u bbox %u/%u hw regs %u state %u contexts %u signal events %u unstalled %u | waits %u timeouts %u max %u us resyncs %u peeks %u held %u callbacks %u signal waits %u timeouts %u (%u us, max %u) skipped %u copies %u stale %u | VRAM copies %u (%u pages, %u us, max %u, how %u %s), transfers into it here %u (%u KB)\n",
					(unsigned)((now - session_start) / 1000000), (unsigned)rates.fps, (unsigned)rates.events,
					(unsigned)rates.walk_us, (unsigned)rates.walk_max_us,
					(unsigned)rates.invalidates, (unsigned)(block_calls - calls_start),
					(unsigned)frame_stats.lists, (unsigned)frame_stats.commands, (unsigned)frame_stats.prims, (unsigned)frame_stats.vertices,
					(unsigned)frame_stats.clears, (unsigned)frame_stats.signals, (unsigned)frame_stats.held, (unsigned)frame_stats.bad, (unsigned)frame_stats.bbox_visible, (unsigned)frame_stats.bbox_hidden, (unsigned)frame_stats.hw_regs, (unsigned)frame_stats.hw_state, (unsigned)frame_stats.contexts, (unsigned)frame_stats.signal_events, (unsigned)frame_stats.unstalled,
					(unsigned)rates.sync_waits, (unsigned)rates.sync_timeouts, (unsigned)rates.sync_max_us, (unsigned)rates.resyncs,
					(unsigned)ring->sync_peeks, (unsigned)ring->sync_held, (unsigned)ring->callbacks,
					(unsigned)ring->signal_waits, (unsigned)ring->signal_timeouts, (unsigned)ring->signal_wait_us, (unsigned)ring->signal_max_us, (unsigned)ring->signal_skipped, (unsigned)overlays_used, (unsigned)overlays_stale,
					(unsigned)rates.vram_copies, (unsigned)ring->vram_copy_pages, (unsigned)rates.vram_copy_us, (unsigned)rates.vram_copy_max_us, (unsigned)ring->vram_copy_how, ring->vram_copy_module,
					(unsigned)vram_writes, (unsigned)(vram_write_bytes / 1024));
			}

			if (walker.texture_fn && now - log_start < 1000) {
				GeSgxTexStats ts;
				geSgxTexStats(&ts);
				GeSgxLog("GeSgx build: draws %u verts %u indices %u 3d prims %u (gpu %u, %u KB copied vertices, %u KB uniforms; cpu: sprites %u layout %u full %u) skipped %u, past the depth range %u, culling off %u | interval: %u frames, vertices %u, states %u, us state %u texture %u (clut %u) vertices %u (index read %u, uniforms %u) indices %u, walk %u | published %u dropped %u behind %u | tex %u (%u KB) uploads %u (%u KB, %u us) checks %u failed %u\n",
					(unsigned)frame_stats_draws, (unsigned)frame_stats_verts, (unsigned)frame_stats_indices, (unsigned)frame_stats_3d,
					(unsigned)frame_stats_gpu, (unsigned)(frame_stats_raw / 1024), (unsigned)(frame_stats_uni / 1024),
					(unsigned)frame_stats_cpu[0], (unsigned)frame_stats_cpu[1], (unsigned)frame_stats_cpu[2], (unsigned)frame_stats_other, (unsigned)frame_stats_range, (unsigned)frame_stats_cull_off,
					(unsigned)built_frames, (unsigned)built_vertices, (unsigned)built_states,
					(unsigned)built_us[0], (unsigned)built_us[1], (unsigned)built_us[6], (unsigned)built_us[2], (unsigned)built_us[4],
					(unsigned)built_us[5], (unsigned)built_us[3], (unsigned)interval_walk_us,
					(unsigned)frames_published, (unsigned)frames_dropped, (unsigned)frames_skipped,
					(unsigned)ts.entries, (unsigned)ts.kb, (unsigned)ts.uploads, (unsigned)ts.upload_kb, (unsigned)ts.upload_us,
					(unsigned)ts.checks, (unsigned)ts.failed);
				built_vertices = built_states = built_frames = 0;
				memset(built_us, 0, sizeof(built_us));
				interval_walk_us = 0;
			}

			if (now - thread_report >= LOG_INTERVAL_US) {
				reportThreads(now - thread_report);
				thread_report = now;
			}

			second_start = now;
			flips = events = walk_us = walk_max_us = 0;
			fetches_start = block_copies;
			calls_start = block_calls;
		}
	}

	return 0;
}

int GeSgxStart(uint32_t ring_addr) {
	if (config.sgx_renderer == GE_SGX_MODE_OFF)
		return 0;

	const GeSgxRing *ring = (const GeSgxRing *)ScePspemuConvertAddress(ring_addr, KERMIT_INPUT_MODE, RING_EVENTS_OFFSET);
	if (!ring || ring->magic != GE_SGX_MAGIC || ring->version != GE_SGX_VERSION)
		return 0;

	uint32_t entries = ring->entries;
	if (entries < 2 || entries > GE_SGX_RING_ENTRIES || (entries & (entries - 1)))
		return 0;

	if (thid < 0) {
		/*
			Core 0: measured in Carbon, the emulator keeps core 1 for audio
			(~37%) and core 2 for the media engine (25-55%), core 0 has only
			ScePspemuMain and Msfs (~5-10%). Core 3 is busy with other work.
		*/
		thid = sceKernelCreateThread("AdrenalineGeSgx", GeSgxThread, 0x90, 0x4000, 0,
			SCE_KERNEL_CPU_MASK_USER_0, NULL);
		if (thid < 0)
			return 0;
		GeSgxTrackThread(thid, "AdrenalineGeSgx");
		if (sceKernelStartThread(thid, 0, NULL) < 0) {
			sceKernelDeleteThread(thid);
			thid = -1;
			return 0;
		}
	}

	// The PSP side may wait for us from now on (and copies no VRAM page before we read one)
	memset(vram_want, 0, sizeof(vram_want));
	memset(picture_pages, 0, sizeof(picture_pages));
	vram_want_dirty = 1;
	writeAck(ring->session, 0);

	start_ring = ring_addr;
	start_session = ring->session;
	start_entries = entries;
	__sync_synchronize();
	start_seq++;

	return 1;
}
