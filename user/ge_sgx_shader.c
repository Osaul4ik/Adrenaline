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
	SGX renderer: Cg shaders generated from the GE state, compiled on the
	console with SceShaccCg (libshacccg.suprx, as installed for vitaGL
	ports), cached on the memory card.

	The compiler module is loaded only when a program is missing from the
	cache, in a thread of its own (it needs a deep stack), with its own heap
	(sceClibMspace on a memory block), and unloaded again when it has had
	nothing to do for a while. Its functions are looked up by NID after the
	load (taiGetModuleExportFunc): an import of SceShaccCg would stop
	adrenaline_user from loading where the library is missing.
*/

#include <psp2/io/fcntl.h>
#include <psp2/io/stat.h>
#include <psp2/kernel/clib.h>
#include <psp2/kernel/modulemgr.h>
#include <psp2/kernel/processmgr.h>
#include <psp2/kernel/sysmem.h>
#include <psp2/kernel/threadmgr.h>
#include <psp2/shacccg.h>
#include <taihen.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ge_sgx.h"
#include "ge_sgx_shader.h"
#include "ge_sgx_cg.h"
#include "ge_sgx_blend.h"

#define MAX_PROGRAMS 1024
#define QUEUE_SIZE 64 // power of two

#define COMPILER_HEAP_SIZE (6 * 1024 * 1024)
#define COMPILER_STACK_SIZE (4 * 1024 * 1024)
#define COMPILER_IDLE_US (10 * 1000 * 1000)


enum {
	STATE_NONE,
	STATE_QUEUED,
	STATE_READY,
	STATE_FAILED,
};

typedef struct {
	uint32_t key;
	volatile int state;
	SceGxmProgram *program;
} Program;

static Program programs[MAX_PROGRAMS];
static int nprograms = 0;

// Draw thread -> compiler thread
static volatile uint32_t queue[QUEUE_SIZE];
static volatile uint32_t queue_write = 0, queue_read = 0;
static SceUID queue_sema = -1;
static SceUID compiler_thid = -1;

static GeSgxShaderStats stats;

static SceUID heap_block = -1;
static SceClibMspace heap = NULL;
static int compile_requested = 0;

uint32_t geSgxShaderKey(const GeSgxDraw *d) {
	uint32_t key = 0;

	if (d->flags & GE_SGX_DRAW_CLEAR)
		return 1u << 10;

	if (d->flags & GE_SGX_DRAW_TEXTURED) {
		key |= 1;
		key |= (uint32_t)(d->tfunc & 7) << 1;
		if (d->flags & GE_SGX_DRAW_TCC)
			key |= 1 << 4;
	}
	if (d->flags & GE_SGX_DRAW_DOUBLE)
		key |= 1 << 5;
	// ALWAYS passes everything: no test
	if ((d->flags & GE_SGX_DRAW_ALPHATEST) && d->atest_func != 1)
		key |= (1 << 6) | ((uint32_t)(d->atest_func & 7) << 7);
	if (d->flags & GE_SGX_DRAW_FOG)
		key |= 1 << 11;
	if (d->flags & GE_SGX_DRAW_BLEND) {
		GeSgxBlend b;
		geSgxBlendSetup(d, &b);
		if (b.fix)
			key |= GE_SGX_FS_FIX;
	}

	return key;
}

static void *heapAlloc(unsigned int size) {
	return sceClibMspaceMalloc(heap, size);
}

static void heapFree(void *ptr) {
	sceClibMspaceFree(heap, ptr);
}

// The compiler module, loaded into this process by taiHEN
static SceUID loadModule(const char *path) {
	SceUID modid = taiLoadStartModuleForPid(sceKernelGetProcessId(), path, 0, NULL, 0);
	if (modid < 0) {
		GeSgxLog("GeSgx shader: taiHEN load %s: %08X\n", path, (unsigned)modid);
		modid = sceKernelLoadStartModule(path, 0, NULL, 0, NULL, NULL);
		if (modid < 0)
			GeSgxLog("GeSgx shader: load %s: %08X\n", path, (unsigned)modid);
	}
	return modid;
}

static int loadCompiler(void) {
	heap_block = sceKernelAllocMemBlock("SceGeSgxCgHeap", SCE_KERNEL_MEMBLOCK_TYPE_USER_RW, COMPILER_HEAP_SIZE, NULL);
	if (heap_block < 0)
		return heap_block;
	void *base = NULL;
	sceKernelGetMemBlockBase(heap_block, &base);
	heap = sceClibMspaceCreate(base, COMPILER_HEAP_SIZE);
	if (!heap) {
		sceKernelFreeMemBlock(heap_block);
		return -1;
	}

	int res = geSgxCgInit(loadModule, heapAlloc, heapFree);
	if (res != 1) {
		sceClibMspaceDestroy(heap);
		sceKernelFreeMemBlock(heap_block);
		heap = NULL;
		heap_block = -1;
	}
	return res;
}

static void unloadCompiler(void) {
	geSgxCgTerm();
	sceClibMspaceDestroy(heap);
	sceKernelFreeMemBlock(heap_block);
	heap = NULL;
	heap_block = -1;
}

static void compileLog(uint32_t key, int line, const char *msg) {
	GeSgxLog("GeSgx shader %04X: line %d: %s\n", (unsigned)key, line, msg);
}

static SceGxmProgram *compile(uint32_t key) {
	uint32_t size = 0;
	SceGxmProgram *program = geSgxCgCompile(key, &size, compileLog);
	if (program)
		geSgxCgSave(key, program, size);
	return program;
}

// The compiler cannot be loaded here: the Adrenaline app compiles at its next start
static void requestCompile(void) {
	sceIoMkdir("ux0:data/adrenaline", 0777);
	sceIoMkdir(GE_SGX_CG_DIR, 0777);
	SceUID fd = sceIoOpen(GE_SGX_CG_REQUEST, SCE_O_WRONLY | SCE_O_CREAT | SCE_O_TRUNC, 0777);
	if (fd >= 0) {
		sceIoWrite(fd, "1", 1);
		sceIoClose(fd);
	}
	GeSgxLog("GeSgx shader: shaders requested from the Adrenaline app (restart Adrenaline once)\n");
}

static Program *find(uint32_t key) {
	int i;
	for (i = 0; i < nprograms; i++) {
		if (programs[i].key == key)
			return &programs[i];
	}
	return NULL;
}

static int CompilerThread(SceSize args, void *argp) {
	while (1) {
		// Wait for work; unload the compiler after a while without
		SceUInt timeout = COMPILER_IDLE_US;
		int res = sceKernelWaitSema(queue_sema, 1, stats.compiler == 1 ? &timeout : NULL);
		if (res < 0) {
			if (stats.compiler == 1) {
				unloadCompiler();
				stats.compiler = 0;
				GeSgxLog("GeSgx shader: compiler unloaded\n");
			}
			continue;
		}

		while (queue_read != queue_write) {
			uint32_t key = queue[queue_read & (QUEUE_SIZE - 1)];
			Program *p = NULL;
			int i;
			for (i = 0; i < nprograms; i++) {
				if (programs[i].key == key) {
					p = &programs[i];
					break;
				}
			}

			if (p && p->state == STATE_QUEUED) {
				if (stats.compiler == 0) {
					int r = loadCompiler();
					stats.compiler = r == 1 ? 1 : r;
					GeSgxLog("GeSgx shader: compiler load %08X\n", (unsigned)r);
					if (r != 1 && !compile_requested) {
						compile_requested = 1;
						requestCompile();
					}
				}

				SceGxmProgram *program = NULL;
				if (stats.compiler == 1) {
					SceUInt64 t0 = sceKernelGetProcessTimeWide();
					program = compile(key);
					stats.compile_us = (uint32_t)(sceKernelGetProcessTimeWide() - t0);
					GeSgxLog("GeSgx shader %04X: %s in %u us\n", (unsigned)key, program ? "compiled" : "FAILED", (unsigned)stats.compile_us);
				}

				p->program = program;
				__sync_synchronize();
				p->state = program ? STATE_READY : STATE_FAILED;
				if (program) {
					stats.compiled++;
					stats.ready++;
				} else {
					stats.failed++;
				}
				stats.queued--;
			}

			__sync_synchronize();
			queue_read++;
		}
	}
	return 0;
}

static int startCompiler(void) {
	if (compiler_thid >= 0)
		return 1;
	queue_sema = sceKernelCreateSema("SceGeSgxShaderQueue", 0, 0, QUEUE_SIZE, NULL);
	if (queue_sema < 0)
		return 0;
	// Low priority, cores 0-2: the compiler takes a while and must not stall anything
	compiler_thid = sceKernelCreateThread("AdrenalineGeSgxCg", CompilerThread, 0xB0, COMPILER_STACK_SIZE, 0,
		SCE_KERNEL_CPU_MASK_USER_0 | SCE_KERNEL_CPU_MASK_USER_1 | SCE_KERNEL_CPU_MASK_USER_2, NULL);
	if (compiler_thid < 0)
		return 0;
	sceKernelStartThread(compiler_thid, 0, NULL);
	GeSgxTrackThread(compiler_thid, "AdrenalineGeSgxCg");
	return 1;
}

const SceGxmProgram *geSgxShaderGet(uint32_t key) {
	Program *p = find(key);

	if (p) {
		if (p->state == STATE_READY) {
			__sync_synchronize();
			return p->program;
		}
		return NULL;
	}

	if (nprograms >= MAX_PROGRAMS)
		return NULL;

	p = &programs[nprograms];
	p->key = key;
	p->program = geSgxCgLoadCached(key);
	if (p->program) {
		p->state = STATE_READY;
		nprograms++;
		stats.loaded++;
		stats.ready++;
		return p->program;
	}

	// The queue is full or the thread is not there: try again next time
	if (queue_write - queue_read >= QUEUE_SIZE || !startCompiler())
		return NULL;

	// Not in the cache: which ones (the app compiles every key there is)
	static int missing_logs = 0;
	if (missing_logs < 16) {
		missing_logs++;
		char path[96];
		geSgxCgPath(key, path, sizeof(path));
		GeSgxLog("GeSgx shader %04X: not in the cache (%s)\n", (unsigned)key, path);
	}

	p->state = STATE_QUEUED;
	__sync_synchronize();
	nprograms++;
	stats.queued++;

	queue[queue_write & (QUEUE_SIZE - 1)] = key;
	__sync_synchronize();
	queue_write++;
	sceKernelSignalSema(queue_sema, 1);
	return NULL;
}

int geSgxShaderFailed(uint32_t key) {
	Program *p = find(key);
	return p && p->state == STATE_FAILED;
}

void geSgxShaderStats(GeSgxShaderStats *out) {
	*out = stats;
}
