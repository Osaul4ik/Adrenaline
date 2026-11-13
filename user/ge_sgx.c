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
#include <psp2/kernel/processmgr.h>
#include <psp2/kernel/sysmem.h>
#include <psp2/kernel/threadmgr.h>

#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "main.h"
#include "ge_sgx.h"
#include "../ge_sgx_shared.h"

#define LOG_PATH "ux0:data/adrenaline_sgx_log.txt"
#define LOG_INTERVAL_US (5 * 1000 * 1000)

#define EVENTS_PER_BATCH 256
#define IDLE_DELAY_US 300

#define RAM_START  0x08000000u
#define RAM_PAGES  (PSP_RAM_SIZE / GE_SGX_PAGE_SIZE)
#define VRAM_START 0x04000000u
#define VRAM_SIZE  0x00200000u
#define VRAM_PAGES (VRAM_SIZE / GE_SGX_PAGE_SIZE)

#define RING_EVENTS_OFFSET ((uint32_t)offsetof(GeSgxRing, events))

// Start request from the Kermit thread
static volatile uint32_t start_seq = 0;
static volatile uint32_t start_ring = 0;
static volatile uint32_t start_session = 0;
static volatile uint32_t start_entries = 0;

static SceUID thid = -1;

static GeSgxWalker walker;

// Pages fetched since the last event
static uint32_t epoch = 1;
static uint32_t ram_epoch[RAM_PAGES];
static const void *ram_ptr[RAM_PAGES];
static uint32_t vram_epoch[VRAM_PAGES];
static const void *vram_ptr[VRAM_PAGES];
static uint32_t page_fetches = 0;

// Published frame (seqlock: odd while it is written)
static volatile uint32_t frame_seq = 0;
static GeSgxFrame frame;
static volatile SceUInt64 last_flip_time = 0;

static uint32_t cdram_kb = 0, ram_kb = 0;

static const void *pageFn(void *user, uint32_t page) {
	uint32_t *ep;
	const void **pp;
	uint32_t psp;

	if (page >= RAM_START && page < RAM_START + PSP_RAM_SIZE) {
		uint32_t i = (page - RAM_START) / GE_SGX_PAGE_SIZE;
		ep = &ram_epoch[i];
		pp = &ram_ptr[i];
		// Kernel address: ScePspemu maps all of the PSP RAM only for those
		psp = 0x80000000u | page;
	} else if (page >= VRAM_START && page < VRAM_START + 0x00800000u) {
		// VRAM and its mirrors
		uint32_t off = page & (VRAM_SIZE - 1);
		uint32_t i = off / GE_SGX_PAGE_SIZE;
		ep = &vram_epoch[i];
		pp = &vram_ptr[i];
		psp = VRAM_START | off;
	} else {
		return NULL;
	}

	if (*ep != epoch) {
		// Aligned address and size: cacheable mapping, invalidated
		const void *p = (const void *)ScePspemuConvertAddress(psp, KERMIT_INPUT_MODE, GE_SGX_PAGE_SIZE);
		if (!p)
			return NULL;
		*pp = p;
		*ep = epoch;
		page_fetches++;
	}

	return *pp;
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

static void logLine(const char *fmt, ...) {
	char buf[256];
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
	uint32_t seq;
	do {
		seq = frame_seq;
		__sync_synchronize();
		memcpy(out, &frame, sizeof(GeSgxFrame));
		__sync_synchronize();
	} while ((seq & 1) || seq != frame_seq);

	if (out->flips == 0)
		return 0;
	return (sceKernelGetProcessTimeWide() - last_flip_time) < 500 * 1000;
}

static int GeSgxThread(SceSize args, void *argp) {
	uint32_t seen_seq = 0;
	uint32_t ring_addr = 0, session = 0, mask = 0, entries = 0;
	uint32_t read = 0, last_lost = 0;

	// Per second
	SceUInt64 second_start = sceKernelGetProcessTimeWide();
	SceUInt64 log_start = second_start;
	uint32_t flips = 0, events = 0, walk_us = 0, walk_max_us = 0, fetches_start = page_fetches;
	GeSgxFrame rates;
	memset(&rates, 0, sizeof(rates));

	GeSgxStats frame_stats;
	GeSgxRect rects[GE_SGX_FRAME_RECTS];

	cdram_kb = probeMemory(SCE_KERNEL_MEMBLOCK_TYPE_USER_CDRAM_RW, 64);
	ram_kb = probeMemory(SCE_KERNEL_MEMBLOCK_TYPE_USER_RW, 64);
	logLine("GeSgx: free CDRAM %u KB, free RAM %u KB\n", (unsigned)cdram_kb, (unsigned)ram_kb);

	geSgxWalkerInit(&walker, pageFn, NULL);

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
			rates.resyncs = 0;
			logLine("GeSgx: ring %08X, %u entries, session %u\n", (unsigned)ring_addr, (unsigned)entries, (unsigned)session);
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
					publish(rects, nrects, &frame_stats, &rates);
					last_flip_time = sceKernelGetProcessTimeWide();
					flips++;
					break;
				}
				default:
					break;
			}

			uint32_t dt = (uint32_t)(sceKernelGetProcessTimeWide() - t0);
			walk_us += dt;
			if (dt > walk_max_us)
				walk_max_us = dt;
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
			rates.invalidates = page_fetches - fetches_start;

			if (now - log_start >= LOG_INTERVAL_US && flips) {
				log_start = now;
				logLine("GeSgx: fps %u events %u walk %u us (max %u) pages %u | lists %u cmds %u prims %u verts %u clears %u bad %u | waits %u timeouts %u max %u us resyncs %u\n",
					(unsigned)rates.fps, (unsigned)rates.events, (unsigned)rates.walk_us, (unsigned)rates.walk_max_us, (unsigned)rates.invalidates,
					(unsigned)frame_stats.lists, (unsigned)frame_stats.commands, (unsigned)frame_stats.prims, (unsigned)frame_stats.vertices,
					(unsigned)frame_stats.clears, (unsigned)frame_stats.bad,
					(unsigned)rates.sync_waits, (unsigned)rates.sync_timeouts, (unsigned)rates.sync_max_us, (unsigned)rates.resyncs);
			}

			second_start = now;
			flips = events = walk_us = walk_max_us = 0;
			fetches_start = page_fetches;
		}
	}

	return 0;
}

int GeSgxStart(uint32_t ring_addr) {
	if (!config.sgx_renderer)
		return 0;

	const GeSgxRing *ring = (const GeSgxRing *)ScePspemuConvertAddress(ring_addr, KERMIT_INPUT_MODE, RING_EVENTS_OFFSET);
	if (!ring || ring->magic != GE_SGX_MAGIC || ring->version != GE_SGX_VERSION)
		return 0;

	uint32_t entries = ring->entries;
	if (entries < 2 || entries > GE_SGX_RING_ENTRIES || (entries & (entries - 1)))
		return 0;

	if (thid < 0) {
		thid = sceKernelCreateThread("AdrenalineGeSgx", GeSgxThread, 0x60, 0x4000, 0, 0, NULL);
		if (thid < 0)
			return 0;
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
