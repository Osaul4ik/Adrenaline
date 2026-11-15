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
	SGX renderer: GPU memory from adrenaline_kernel, outside the emulator's
	budget (see kernel/gesgx_mem.c), for the texture cache and the depth
	buffer. A few physically contiguous blocks (the kernel gives 16 MB at
	most in one, none of CDRAM), each mapped for the GPU once and handed out
	in 4 KB pages. Without them everything is as before: memblocks of the
	emulator's own budget.

	Used from the walker thread (textures) and the draw thread (depth): a
	spinlock around the page map.
*/

#include <psp2/gxm.h>
#include <psp2/kernel/sysmem.h>

#include <string.h>

#include "ge_sgx.h"
#include "ge_sgx_pool.h"
#include "../ge_sgx_kernel.h"

#define PAGE 4096u
#define CHUNK_MAX_PAGES (16u * 1024 * 1024 / PAGE)
#define MAX_CHUNKS 4
// Not all there is: the system (the shell, the video decoder) shares the rest
#define POOL_MAX_BYTES (48u * 1024 * 1024)

typedef struct {
	uint8_t *base;
	uint32_t pages, free, rover;
	uint32_t used_map[CHUNK_MAX_PAGES / 32]; // bit set: page taken
} Chunk;

static Chunk chunks[MAX_CHUNKS];
static int nchunks = 0;
static uint32_t pool_pages = 0, free_pages = 0;
static int pool_cdram = 0;
static volatile int lock = 0;

static void take(void) {
	while (__sync_lock_test_and_set(&lock, 1))
		;
}

static void give(void) {
	__sync_lock_release(&lock);
}

static int tryChunk(SceKernelMemBlockType type, uint32_t mb) {
	uint32_t size = mb << 20;
	void *base = NULL;
	GeSgxKernelMemInfo info;
	memset(&info, 0, sizeof(info));
	int res = kuGeSgxAllocMemBlock(type, size, &base, &info);
	int map = -1;
	if (res >= 0 && base) {
		map = sceGxmMapMemory(base, size, SCE_GXM_MEMORY_ATTRIB_READ | SCE_GXM_MEMORY_ATTRIB_WRITE);
		if (map < 0)
			kuGeSgxFreeMemBlock(base);
	}
	GeSgxLog("GeSgx pool: %s %u MB: %s at %08X, how %d | kernel %08X/%08X, pa %08X err %08X, at pa %08X, mirror %08X, gpu map %08X\n",
		type == SCE_KERNEL_MEMBLOCK_TYPE_USER_CDRAM_RW ? "CDRAM" : "RAM", (unsigned)mb, res >= 0 && map >= 0 ? "ok" : "no",
		(unsigned)(uintptr_t)base, info.how, (unsigned)info.kernel_err, (unsigned)info.kernel_err2, (unsigned)info.paddr,
		(unsigned)info.pa_err, (unsigned)info.paddr_err, (unsigned)info.mirror_err, (unsigned)map);
	if (res < 0 || map < 0)
		return 0;
	Chunk *c = &chunks[nchunks++];
	memset(c, 0, sizeof(*c));
	c->base = (uint8_t *)base;
	c->pages = c->free = size / PAGE;
	pool_pages += c->pages;
	free_pages += c->pages;
	if (nchunks == 1)
		pool_cdram = type == SCE_KERNEL_MEMBLOCK_TYPE_USER_CDRAM_RW;
	return 1;
}

uint32_t geSgxPoolInit(void) {
	static int tried = 0;
	if (tried)
		return pool_pages * PAGE;
	tried = 1;
	// CDRAM (the GPU's own memory) if the kernel has some, then main RAM: 16 MB blocks while it gives them, then smaller
	static const uint32_t mb[] = { 16, 8, 4 };
	unsigned k;
	for (k = 0; k < sizeof(mb) / sizeof(mb[0]) && nchunks < MAX_CHUNKS && pool_pages * PAGE + (mb[k] << 20) <= POOL_MAX_BYTES; ) {
		if (!tryChunk(SCE_KERNEL_MEMBLOCK_TYPE_USER_CDRAM_RW, mb[k]))
			break;
	}
	for (k = 0; k < sizeof(mb) / sizeof(mb[0]); k++) {
		while (nchunks < MAX_CHUNKS && pool_pages * PAGE + (mb[k] << 20) <= POOL_MAX_BYTES &&
		       tryChunk(SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE, mb[k]))
			;
	}
	GeSgxLog("GeSgx pool: %u KB in %d blocks%s\n", (unsigned)(pool_pages * (PAGE / 1024)), nchunks,
		nchunks ? "" : ", the emulator's memory as before");
	return pool_pages * PAGE;
}

uint32_t geSgxPoolBytes(void) {
	return pool_pages * PAGE;
}

uint32_t geSgxPoolFreeKB(void) {
	return free_pages * (PAGE / 1024);
}

int geSgxPoolIsCdram(void) {
	return pool_cdram;
}

static inline int taken(const Chunk *c, uint32_t p) {
	return (c->used_map[p >> 5] >> (p & 31)) & 1;
}

static void mark(Chunk *c, uint32_t first, uint32_t n, int on) {
	uint32_t p;
	for (p = first; p < first + n; p++) {
		if (on)
			c->used_map[p >> 5] |= 1u << (p & 31);
		else
			c->used_map[p >> 5] &= ~(1u << (p & 31));
	}
}

// First run of n free pages from start, below end; c->pages if none
static uint32_t findRun(const Chunk *c, uint32_t start, uint32_t end, uint32_t n) {
	uint32_t p = start, run = 0;
	while (p < end) {
		// whole words taken are skipped
		if ((p & 31) == 0 && c->used_map[p >> 5] == 0xFFFFFFFFu) {
			p += 32;
			run = 0;
			continue;
		}
		if (taken(c, p)) {
			run = 0;
		} else if (++run == n) {
			return p + 1 - n;
		}
		p++;
	}
	return c->pages;
}

void *geSgxPoolAlloc(uint32_t bytes) {
	uint32_t n = (bytes + PAGE - 1) / PAGE;
	if (!nchunks || !n)
		return NULL;
	take();
	void *mem = NULL;
	int k;
	for (k = 0; k < nchunks && !mem; k++) {
		Chunk *c = &chunks[k];
		if (n > c->free)
			continue;
		// next fit: from where the last one ended, then from the start
		uint32_t first = findRun(c, c->rover, c->pages, n);
		if (first >= c->pages)
			first = findRun(c, 0, c->rover + n < c->pages ? c->rover + n : c->pages, n);
		if (first < c->pages) {
			mark(c, first, n, 1);
			c->free -= n;
			free_pages -= n;
			c->rover = first + n < c->pages ? first + n : 0;
			mem = c->base + first * PAGE;
		}
	}
	give();
	return mem;
}

void geSgxPoolFree(void *mem, uint32_t bytes) {
	uint32_t n = (bytes + PAGE - 1) / PAGE;
	int k;
	for (k = 0; k < nchunks; k++) {
		Chunk *c = &chunks[k];
		if ((uint8_t *)mem < c->base || (uint8_t *)mem >= c->base + c->pages * PAGE)
			continue;
		uint32_t first = (uint32_t)((uint8_t *)mem - c->base) / PAGE;
		if (first + n > c->pages)
			return;
		take();
		mark(c, first, n, 0);
		c->free += n;
		free_pages += n;
		give();
		return;
	}
}
