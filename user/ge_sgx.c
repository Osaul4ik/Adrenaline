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
#include "ge_sgx_tex.h"
#include "ge_sgx_render.h"
#include "../ge_sgx_shared.h"

#define LOG_PATH "ux0:data/adrenaline_sgx_log.txt"
#define LOG_INTERVAL_US (5 * 1000 * 1000)

#define EVENTS_PER_BATCH 256
#define IDLE_DELAY_US 300

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
static const uint8_t *vram_map = NULL; // PSP VRAM 0x04000000
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

static const void *blockFn(void *user, uint32_t block) {
	const uint8_t *src;

	if (block >= RAM_START && block < RAM_START + PSP_RAM_SIZE) {
		src = ram_map + (block - RAM_START);
	} else if (block >= VRAM_START && block < VRAM_START + 0x00800000u) {
		// VRAM and its mirrors
		src = vram_map + (block & (VRAM_SIZE - 1));
	} else {
		return NULL;
	}

	block_calls++;

	uint32_t i = (block / GE_SGX_BLOCK_SIZE) & (BLOCK_CACHE_ENTRIES - 1);
	uint8_t *dst = block_data + i * GE_SGX_BLOCK_SIZE;

	if (block_tag[i] != (block | 1) || block_epoch[i] != epoch) {
		memcpy(dst, src, GE_SGX_BLOCK_SIZE);
		block_tag[i] = block | 1;
		block_epoch[i] = epoch;
		block_copies++;
	}

	return dst;
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
		return vram_map + off;
	}
	return NULL;
}

static uint32_t clockUs(void) {
	return sceKernelGetProcessTimeLow();
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
			s->data.draws_3d = 0;
			s->data.prims_gpu = 0;
			memset(s->data.cpu_reason, 0, sizeof(s->data.cpu_reason));
			s->data.vertices_built = 0;
			s->data.state_builds = 0;
			s->data.us_state = s->data.us_texture = s->data.us_vertices = s->data.us_indices = 0;
			s->data.skipped_other = 0;
			building = s;
			walker.frame = &s->data;
			walker.gpu_transform = !ge_sgx_gpu_failed;
			walker.ngpu_bad = ge_sgx_gpu_nbad;
			for (int b = 0; b < walker.ngpu_bad; b++)
				walker.gpu_bad[b] = ge_sgx_gpu_bad[b];
			return;
		}
	}
	frames_dropped++;
}

// Walker thread, at a display flip of fb
static void publishSlot(uint32_t top) {
	uint32_t addr = top & 0x0FFFFFFF;
	uint32_t fb = (addr >= VRAM_START && addr < VRAM_START + 0x00800000u) ? (addr & (VRAM_SIZE - 1)) : 0xFFFFFFFFu;

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

void GeSgxDrawTick(void) {
	draw_tick++;
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
static uint32_t probeMemory(SceKernelMemBlockType type, int max_blocks) {
	SceUID blocks[256];
	int n = 0;
	if (max_blocks > 256)
		max_blocks = 256;
	while (n < max_blocks) {
		SceUID id = sceKernelAllocMemBlock("SceGeSgxProbe", type, 1024 * 1024, NULL);
		if (id < 0)
			break;
		blocks[n++] = id;
	}
	int i;
	for (i = 0; i < n; i++)
		sceKernelFreeMemBlock(blocks[i]);
	return (uint32_t)n * 1024;
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
	uint32_t read = 0, last_lost = 0;
	uint32_t logged_types = 0, slow_logs = 0, bad_logged = 0, bad_logs = 0;
	uint32_t frame_stats_3d = 0, frame_stats_other = 0, frame_stats_draws = 0, frame_stats_verts = 0, frame_stats_indices = 0;
	uint32_t frame_stats_gpu = 0, frame_stats_raw = 0, frame_stats_uni = 0, frame_stats_cpu[3] = { 0 };
	uint32_t built_vertices = 0, built_states = 0, built_frames = 0, built_us[4] = { 0 }; // per log interval
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

	cdram_kb = probeMemory(SCE_KERNEL_MEMBLOCK_TYPE_USER_CDRAM_RW, 64);
	ram_kb = probeMemory(SCE_KERNEL_MEMBLOCK_TYPE_USER_RW, 64);
	GeSgxLog("GeSgx: free CDRAM %u KB, free RAM %u KB\n", (unsigned)cdram_kb, (unsigned)ram_kb);

	benchmarkMemory();

	ram_map = (const uint8_t *)ScePspemuConvertAddress(0x80000000u | RAM_START, 0, 0);
	vram_map = (const uint8_t *)ScePspemuConvertAddress(VRAM_START, 0, 0);
	if (!ram_map || !vram_map || !initBlockCache()) {
		GeSgxLog("GeSgx: no memory map (%08X %08X) or block cache\n", (unsigned)(uintptr_t)ram_map, (unsigned)(uintptr_t)vram_map);
		return 0;
	}
	GeSgxLog("GeSgx: RAM map %08X, VRAM map %08X\n", (unsigned)(uintptr_t)ram_map, (unsigned)(uintptr_t)vram_map);

	// Could the GPU read PSP memory where it is (no vertex copies)? Probe only, for the log
	{
		void *probe = (void *)(ram_map + 0x00800000u);
		int map_res = sceGxmMapMemory(probe, 0x10000, SCE_GXM_MEMORY_ATTRIB_READ);
		int unmap_res = map_res >= 0 ? sceGxmUnmapMemory(probe) : 0;
		GeSgxLog("GeSgx: GPU map of PSP RAM: %08X (unmap %08X)\n", (unsigned)map_res, (unsigned)unmap_res);
	}

	geSgxWalkerInit(&walker, blockFn, NULL);

	if (config.sgx_renderer >= GE_SGX_MODE_DRAW) {
		int slots_ok = initSlots();
		int tex_ok = geSgxTexInit();
		GeSgxLog("GeSgx: frames %d, texture cache %d, free RAM now %u KB\n", slots_ok, tex_ok,
			(unsigned)probeMemory(SCE_KERNEL_MEMBLOCK_TYPE_USER_RW, 64));
		if (slots_ok && tex_ok) {
			walker.texture_fn = geSgxTexLookup;
			walker.bulk_fn = bulkFn;
			walker.clock_fn = clockUs;
			walker.gpu_transform = 1;
			acquireSlot();
		}
	}

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
			if (walker.texture_fn) {
				geSgxTexReset();
				if (building) {
					building->data.ndraws = 0;
					building->data.nvertices = 0;
					building->data.nindices = 0;
					building->data.nuniforms = 0;
					building->data.nraw = 0;
				}
			}
			rates.resyncs = 0;
			logged_types = 0;
			slow_logs = 0;
			session_start = sceKernelGetProcessTimeWide();
			GeSgxLog("GeSgx: ring %08X, %u entries, session %u\n", (unsigned)ring_addr, (unsigned)entries, (unsigned)session);
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

		if (lost != last_lost || (write - read) > entries) {
			// Events are missing: start over from the next ones
			last_lost = lost;
			read = write;
			geSgxWalkerReset(&walker);
			rates.resyncs++;
			writeAck(session, read);
			continue;
		}

		if (write == read) {
			sceKernelDelayThread(IDLE_DELAY_US);
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

			if (e->type < 8 && !(logged_types & (1u << e->type))) {
				logged_types |= 1u << e->type;
				GeSgxLog("GeSgx: first event %u: %08X %08X %08X %08X (walker: %d lists)\n",
					(unsigned)e->type, (unsigned)e->a, (unsigned)e->b, (unsigned)e->c, (unsigned)e->d, walker.count);
			}

			switch (e->type) {
				case GE_SGX_EVENT_LIST:
					geSgxWalkerList(&walker, (int)e->a, e->b, e->c, (int)e->d);
					geSgxWalkerRun(&walker);
					break;
				case GE_SGX_EVENT_STALL:
					geSgxWalkerStall(&walker, (int)e->a, e->b);
					geSgxWalkerRun(&walker);
					break;
				case GE_SGX_EVENT_DEQUEUE:
					geSgxWalkerDequeue(&walker, (int)e->a);
					geSgxWalkerRun(&walker);
					break;
				case GE_SGX_EVENT_FLIP: {
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
							frame_stats_draws = building->data.ndraws;
							frame_stats_verts = building->data.nvertices;
							built_vertices += building->data.vertices_built;
							built_states += building->data.state_builds;
							built_us[0] += building->data.us_state;
							built_us[1] += building->data.us_texture;
							built_us[2] += building->data.us_vertices;
							built_us[3] += building->data.us_indices;
							built_frames++;
						}
						publishSlot(e->a);

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
			geSgxWalkerReset(&walker);
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
				GeSgxLog("GeSgx: t %u fps %u events %u walk %u us (max %u) blocks %u lookups %u | lists %u cmds %u prims %u verts %u clears %u signals %u bad %u | waits %u timeouts %u max %u us resyncs %u\n",
					(unsigned)((now - session_start) / 1000000), (unsigned)rates.fps, (unsigned)rates.events,
					(unsigned)rates.walk_us, (unsigned)rates.walk_max_us,
					(unsigned)rates.invalidates, (unsigned)(block_calls - calls_start),
					(unsigned)frame_stats.lists, (unsigned)frame_stats.commands, (unsigned)frame_stats.prims, (unsigned)frame_stats.vertices,
					(unsigned)frame_stats.clears, (unsigned)frame_stats.signals, (unsigned)frame_stats.bad,
					(unsigned)rates.sync_waits, (unsigned)rates.sync_timeouts, (unsigned)rates.sync_max_us, (unsigned)rates.resyncs);
			}

			if (walker.texture_fn && now - log_start < 1000) {
				GeSgxTexStats ts;
				geSgxTexStats(&ts);
				GeSgxLog("GeSgx build: draws %u verts %u indices %u 3d prims %u (gpu %u, %u KB vertices, %u KB uniforms; cpu: sprites %u layout %u full %u) skipped %u | interval: %u frames, vertices %u, states %u, us state %u texture %u vertices %u indices %u | published %u dropped %u behind %u | tex %u (%u KB) uploads %u (%u KB, %u us) checks %u failed %u\n",
					(unsigned)frame_stats_draws, (unsigned)frame_stats_verts, (unsigned)frame_stats_indices, (unsigned)frame_stats_3d,
					(unsigned)frame_stats_gpu, (unsigned)(frame_stats_raw / 1024), (unsigned)(frame_stats_uni / 1024),
					(unsigned)frame_stats_cpu[0], (unsigned)frame_stats_cpu[1], (unsigned)frame_stats_cpu[2], (unsigned)frame_stats_other,
					(unsigned)built_frames, (unsigned)built_vertices, (unsigned)built_states,
					(unsigned)built_us[0], (unsigned)built_us[1], (unsigned)built_us[2], (unsigned)built_us[3],
					(unsigned)frames_published, (unsigned)frames_dropped, (unsigned)frames_skipped,
					(unsigned)ts.entries, (unsigned)ts.kb, (unsigned)ts.uploads, (unsigned)ts.upload_kb, (unsigned)ts.upload_us,
					(unsigned)ts.checks, (unsigned)ts.failed);
				built_vertices = built_states = built_frames = 0;
				memset(built_us, 0, sizeof(built_us));
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

	// The PSP side may wait for us from now on
	writeAck(ring->session, 0);

	start_ring = ring_addr;
	start_session = ring->session;
	start_entries = entries;
	__sync_synchronize();
	start_seq++;

	return 1;
}
