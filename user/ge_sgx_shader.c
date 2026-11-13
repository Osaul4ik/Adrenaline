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

// Bump when the generated sources change: older cached programs are not used
#define SHADER_VERSION 1

#define CACHE_DIR "ux0:data/adrenaline/sgx_shaders"

#define MAX_PROGRAMS 256
#define QUEUE_SIZE 64 // power of two

#define COMPILER_HEAP_SIZE (6 * 1024 * 1024)
#define COMPILER_STACK_SIZE (4 * 1024 * 1024)
#define COMPILER_IDLE_US (10 * 1000 * 1000)

#define SHACCCG_LIB_NID 0xA05BBEBB

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

// The SceShaccCg functions, looked up after the module is loaded
static struct {
	SceUID modid;
	SceUID heap_block;
	SceClibMspace heap;
	int (* InitializeCompileOptions)(SceShaccCgCompileOptions *options);
	const SceShaccCgCompileOutput *(* CompileProgram)(const SceShaccCgCompileOptions *options, const SceShaccCgCallbackList *callbacks, int unk);
	int (* SetDefaultAllocator)(void *(*malloc_cb)(unsigned int), void (*free_cb)(void *));
	void (* InitializeCallbackList)(SceShaccCgCallbackList *callbacks, SceShaccCgCallbackDefaults defaults);
	void (* DestroyCompileOutput)(const SceShaccCgCompileOutput *output);
	void (* ReleaseCompiler)(void);
} cg;

static SceShaccCgSourceFile source_file;

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

	return key;
}

static int sourceOf(uint32_t key, char *buf, int size) {
	if (key == GE_SGX_SHADER_VERTEX) {
		return snprintf(buf, size,
			"void main(\n"
			"	float3 aPosition,\n"
			"	float2 aTexcoord,\n"
			"	float4 aColor,\n"
			"	out float4 vPosition : POSITION,\n"
			"	out float2 vTexcoord : TEXCOORD0,\n"
			"	out float4 vColor : COLOR)\n"
			"{\n"
			"	vPosition = float4(aPosition.x * (2.0 / 960.0) - 1.0, 1.0 - aPosition.y * (2.0 / 544.0), aPosition.z, 1.0);\n"
			"	vTexcoord = aTexcoord;\n"
			"	vColor = aColor;\n"
			"}\n");
	}

	int clear = (key >> 10) & 1;
	int textured = key & 1;
	int tfunc = (key >> 1) & 7;
	int tcc = (key >> 4) & 1;
	int dbl = (key >> 5) & 1;
	int atest = (key >> 6) & 1;
	int afunc = (key >> 7) & 7;

	int n = snprintf(buf, size,
		"float4 main(float2 vTexcoord : TEXCOORD0, float4 vColor : COLOR%s%s%s) : COLOR\n"
		"{\n"
		"	float4 v = vColor;\n"
		"	float4 c = v;\n",
		textured ? ", uniform sampler2D uTexture" : "",
		(textured && tfunc == 2) ? ", uniform float4 uEnv" : "",
		atest ? ", uniform float uAlphaRef" : "");

	if (!clear && textured) {
		static const char *funcs[5] = {
			// modulate
			"	c.rgb = v.rgb * t.rgb;\n	c.a = %s;\n",
			// decal
			"	c.rgb = lerp(v.rgb, t.rgb, %s);\n	c.a = v.a;\n",
			// blend
			"	c.rgb = lerp(v.rgb, uEnv.rgb, t.rgb);\n	c.a = %s;\n",
			// replace
			"	c.rgb = t.rgb;\n	c.a = %s;\n",
			// add
			"	c.rgb = v.rgb + t.rgb;\n	c.a = %s;\n",
		};
		static const char *alphas[5][2] = {
			{ "v.a", "v.a * t.a" }, // modulate
			{ "1.0", "t.a" },       // decal: RGB texture is opaque
			{ "v.a", "v.a * t.a" }, // blend
			{ "v.a", "t.a" },       // replace
			{ "v.a", "v.a * t.a" }, // add
		};
		if (tfunc > 4)
			tfunc = 0;
		n += snprintf(buf + n, size - n, "	float4 t = tex2D(uTexture, vTexcoord);\n");
		n += snprintf(buf + n, size - n, funcs[tfunc], alphas[tfunc][tcc]);
	}

	if (dbl)
		n += snprintf(buf + n, size - n, "	c.rgb = c.rgb * 2.0;\n");

	if (atest) {
		static const char *tests[8] = {
			"false",                                   // never
			"true",                                    // always
			"abs(c.a - uAlphaRef) < (0.5 / 255.0)",    // equal
			"abs(c.a - uAlphaRef) >= (0.5 / 255.0)",   // not equal
			"c.a < uAlphaRef - (0.5 / 255.0)",         // less
			"c.a <= uAlphaRef + (0.5 / 255.0)",        // less or equal
			"c.a > uAlphaRef + (0.5 / 255.0)",         // greater
			"c.a >= uAlphaRef - (0.5 / 255.0)",        // greater or equal
		};
		n += snprintf(buf + n, size - n, "	if (!(%s))\n		discard;\n", tests[afunc]);
	}

	n += snprintf(buf + n, size - n, "	return saturate(c);\n}\n");
	return n;
}

static void cachePath(uint32_t key, char *path, int size) {
	snprintf(path, size, CACHE_DIR "/v%d_%04x.gxp", SHADER_VERSION, (unsigned)key);
}

static SceGxmProgram *loadCached(uint32_t key) {
	char path[96];
	cachePath(key, path, sizeof(path));

	SceUID fd = sceIoOpen(path, SCE_O_RDONLY, 0);
	if (fd < 0)
		return NULL;

	SceIoStat st;
	int size = 0;
	if (sceIoGetstatByFd(fd, &st) >= 0)
		size = (int)st.st_size;

	SceGxmProgram *program = NULL;
	if (size > 0 && size < 256 * 1024) {
		program = (SceGxmProgram *)malloc(size);
		if (program && (sceIoRead(fd, program, size) != size || sceGxmProgramCheck(program) < 0)) {
			free(program);
			program = NULL;
		}
	}
	sceIoClose(fd);
	return program;
}

static void saveCached(uint32_t key, const void *data, uint32_t size) {
	char path[96];
	sceIoMkdir("ux0:data/adrenaline", 0777);
	sceIoMkdir(CACHE_DIR, 0777);
	cachePath(key, path, sizeof(path));
	SceUID fd = sceIoOpen(path, SCE_O_WRONLY | SCE_O_CREAT | SCE_O_TRUNC, 0777);
	if (fd < 0)
		return;
	int ok = sceIoWrite(fd, data, size) == (int)size;
	sceIoClose(fd);
	if (!ok)
		sceIoRemove(path);
}

static void *heapAlloc(unsigned int size) {
	return sceClibMspaceMalloc(cg.heap, size);
}

static void heapFree(void *ptr) {
	sceClibMspaceFree(cg.heap, ptr);
}

static SceShaccCgSourceFile *openFile(const char *fileName, const SceShaccCgSourceLocation *includedFrom,
	const SceShaccCgCompileOptions *compileOptions, const char **errorString) {
	return &source_file;
}

static int findExport(uint32_t nid, void *func) {
	uintptr_t addr = 0;
	if (taiGetModuleExportFunc("SceShaccCg", SHACCCG_LIB_NID, nid, &addr) < 0 || !addr)
		return 0;
	*(uintptr_t *)func = addr;
	return 1;
}

static int loadCompiler(void) {
	static const char *paths[] = {
		"ur0:data/libshacccg.suprx",
		"ur0:data/external/libshacccg.suprx",
	};

	cg.heap_block = sceKernelAllocMemBlock("SceGeSgxCgHeap", SCE_KERNEL_MEMBLOCK_TYPE_USER_RW, COMPILER_HEAP_SIZE, NULL);
	if (cg.heap_block < 0)
		return cg.heap_block;
	void *base = NULL;
	sceKernelGetMemBlockBase(cg.heap_block, &base);
	cg.heap = sceClibMspaceCreate(base, COMPILER_HEAP_SIZE);
	if (!cg.heap) {
		sceKernelFreeMemBlock(cg.heap_block);
		return -1;
	}

	int i;
	cg.modid = -1;
	for (i = 0; i < 2 && cg.modid < 0; i++)
		cg.modid = sceKernelLoadStartModule(paths[i], 0, NULL, 0, NULL, NULL);
	if (cg.modid < 0) {
		int err = cg.modid;
		sceClibMspaceDestroy(cg.heap);
		sceKernelFreeMemBlock(cg.heap_block);
		return err;
	}

	if (!findExport(0x3B58AFA0, &cg.InitializeCompileOptions) ||
	    !findExport(0x66814F35, &cg.CompileProgram) ||
	    !findExport(0x6F01D573, &cg.SetDefaultAllocator) ||
	    !findExport(0xA8C2C1C8, &cg.InitializeCallbackList) ||
	    !findExport(0xAA82EF0C, &cg.DestroyCompileOutput) ||
	    !findExport(0x95F57A23, &cg.ReleaseCompiler)) {
		sceKernelStopUnloadModule(cg.modid, 0, NULL, 0, NULL, NULL);
		sceClibMspaceDestroy(cg.heap);
		sceKernelFreeMemBlock(cg.heap_block);
		return -2;
	}

	cg.SetDefaultAllocator(heapAlloc, heapFree);
	return 1;
}

static void unloadCompiler(void) {
	cg.ReleaseCompiler();
	sceKernelStopUnloadModule(cg.modid, 0, NULL, 0, NULL, NULL);
	sceClibMspaceDestroy(cg.heap);
	sceKernelFreeMemBlock(cg.heap_block);
	memset(&cg, 0, sizeof(cg));
}

static SceGxmProgram *compile(uint32_t key) {
	static char src[4096];
	int len = sourceOf(key, src, sizeof(src));
	if (len <= 0 || len >= (int)sizeof(src))
		return NULL;

	source_file.fileName = "<built-in>";
	source_file.text = src;
	source_file.size = len;

	SceShaccCgCallbackList callbacks;
	cg.InitializeCallbackList(&callbacks, SCE_SHACCCG_TRIVIAL);
	callbacks.openFile = openFile;

	SceShaccCgCompileOptions options;
	cg.InitializeCompileOptions(&options);
	options.mainSourceFile = source_file.fileName;
	options.targetProfile = key == GE_SGX_SHADER_VERTEX ? SCE_SHACCCG_PROFILE_VP : SCE_SHACCCG_PROFILE_FP;
	options.entryFunctionName = "main";
	options.useFx = 1;
	options.locale = SCE_SHACCCG_ENGLISH;
	options.optimizationLevel = 3;

	const SceShaccCgCompileOutput *out = cg.CompileProgram(&options, &callbacks, 0);
	SceGxmProgram *program = NULL;

	if (out && out->programData && out->programSize > 0) {
		program = (SceGxmProgram *)malloc(out->programSize);
		if (program) {
			memcpy(program, out->programData, out->programSize);
			if (sceGxmProgramCheck(program) < 0) {
				free(program);
				program = NULL;
			} else {
				saveCached(key, program, out->programSize);
			}
		}
	} else if (out) {
		int i;
		for (i = 0; i < out->diagnosticCount && i < 4; i++) {
			const SceShaccCgDiagnosticMessage *m = &out->diagnostics[i];
			GeSgxLog("GeSgx shader %04X: line %d: %s\n", (unsigned)key,
				m->location ? (int)m->location->lineNumber : -1, m->message ? m->message : "?");
		}
	}

	if (out)
		cg.DestroyCompileOutput(out);
	return program;
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
				if (stats.compiler != 1) {
					int r = loadCompiler();
					stats.compiler = r == 1 ? 1 : r;
					GeSgxLog("GeSgx shader: compiler load %08X\n", (unsigned)r);
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
	p->program = loadCached(key);
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

void geSgxShaderStats(GeSgxShaderStats *out) {
	*out = stats;
}
