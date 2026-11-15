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
	SGX renderer: one big block of GPU memory from adrenaline_kernel, outside
	the emulator's budget (see kernel/gesgx_mem.c), for the texture cache and
	the depth buffer. Mapped for the GPU once, handed out in 4 KB pages.

	CDRAM first (the GPU's own memory), main RAM else; the biggest size that
	the kernel gives. Without it everything is as before: memblocks of the
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
#define MAX_PAGES (128u * 1024 * 1024 / PAGE)

static uint8_t *pool = NULL;
static uint32_t pool_pages = 0;
static uint32_t free_pages = 0;
static uint32_t rover = 0;
static int pool_cdram = 0;
static uint32_t used_map[MAX_PAGES / 32]; // bit set: page taken
static volatile int lock = 0;

static void take(void) {
	while (__sync_lock_test_and_set(&lock, 1))
		;
}

static void give(void) {
	__sync_lock_release(&lock);
}

static int tryPool(SceKernelMemBlockType type, uint32_t mb) {
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
	pool = (uint8_t *)base;
	pool_pages = size / PAGE;
	free_pages = pool_pages;
	pool_cdram = type == SCE_KERNEL_MEMBLOCK_TYPE_USER_CDRAM_RW;
	memset(used_map, 0, sizeof(used_map));
	return 1;
}

uint32_t geSgxPoolInit(void) {
	static int tried = 0;
	if (tried)
		return pool_pages * PAGE;
	tried = 1;
	// Not all there is: the system (the shell, the display) shares the rest
	static const uint32_t cdram_mb[] = { 48, 32, 16 };
	static const uint32_t ram_mb[] = { 48, 32, 16 };
	unsigned i;
	for (i = 0; i < sizeof(cdram_mb) / sizeof(cdram_mb[0]) && !pool; i++)
		tryPool(SCE_KERNEL_MEMBLOCK_TYPE_USER_CDRAM_RW, cdram_mb[i]);
	for (i = 0; i < sizeof(ram_mb) / sizeof(ram_mb[0]) && !pool; i++)
		tryPool(SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE, ram_mb[i]);
	if (!pool)
		GeSgxLog("GeSgx pool: none, the emulator's memory as before\n");
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

static inline int taken(uint32_t p) {
	return (used_map[p >> 5] >> (p & 31)) & 1;
}

static void mark(uint32_t first, uint32_t n, int on) {
	uint32_t p;
	for (p = first; p < first + n; p++) {
		if (on)
			used_map[p >> 5] |= 1u << (p & 31);
		else
			used_map[p >> 5] &= ~(1u << (p & 31));
	}
}

// First run of n free pages from start, below end; pool_pages if none
static uint32_t findRun(uint32_t start, uint32_t end, uint32_t n) {
	uint32_t p = start, run = 0;
	while (p < end) {
		// whole words taken are skipped
		if ((p & 31) == 0 && used_map[p >> 5] == 0xFFFFFFFFu) {
			p += 32;
			run = 0;
			continue;
		}
		if (taken(p)) {
			run = 0;
		} else if (++run == n) {
			return p + 1 - n;
		}
		p++;
	}
	return pool_pages;
}

void *geSgxPoolAlloc(uint32_t bytes) {
	uint32_t n = (bytes + PAGE - 1) / PAGE;
	if (!pool || !n)
		return NULL;
	take();
	void *mem = NULL;
	if (n <= free_pages) {
		// next fit: from where the last one ended, then from the start
		uint32_t first = findRun(rover, pool_pages, n);
		if (first >= pool_pages)
			first = findRun(0, rover + n < pool_pages ? rover + n : pool_pages, n);
		if (first < pool_pages) {
			mark(first, n, 1);
			free_pages -= n;
			rover = first + n < pool_pages ? first + n : 0;
			mem = pool + first * PAGE;
		}
	}
	give();
	return mem;
}

void geSgxPoolFree(void *mem, uint32_t bytes) {
	uint32_t n = (bytes + PAGE - 1) / PAGE;
	if (!pool || !mem || (uint8_t *)mem < pool)
		return;
	uint32_t first = (uint32_t)((uint8_t *)mem - pool) / PAGE;
	if (first + n > pool_pages)
		return;
	take();
	mark(first, n, 0);
	free_pages += n;
	give();
}
