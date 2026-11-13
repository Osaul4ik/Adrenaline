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
	SGX renderer: Cg shaders and their compiler. See ge_sgx_cg.h.

	The compiler's functions are looked up by NID once the module is loaded
	(taiGetModuleExportFunc): an import of SceShaccCg would stop the module
	or app from loading where libshacccg.suprx is missing.
*/

#include <psp2/io/fcntl.h>
#include <psp2/io/stat.h>
#include <psp2/kernel/modulemgr.h>
#include <psp2/shacccg.h>
#include <taihen.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ge_sgx_cg.h"

#define SHACCCG_LIB_NID 0xA05BBEBB

static struct {
	SceUID modid;
	int (* InitializeCompileOptions)(SceShaccCgCompileOptions *options);
	const SceShaccCgCompileOutput *(* CompileProgram)(const SceShaccCgCompileOptions *options, const SceShaccCgCallbackList *callbacks, int unk);
	int (* SetDefaultAllocator)(void *(*malloc_cb)(unsigned int), void (*free_cb)(void *));
	void (* InitializeCallbackList)(SceShaccCgCallbackList *callbacks, SceShaccCgCallbackDefaults defaults);
	void (* DestroyCompileOutput)(const SceShaccCgCompileOutput *output);
	void (* ReleaseCompiler)(void);
} cg;

static SceShaccCgSourceFile source_file;

int geSgxCgSource(uint32_t key, char *buf, int size) {
	if (key == GE_SGX_SHADER_VERTEX) {
		return snprintf(buf, size,
			// Pixels times w to clip space (see GeSgxVertex)
			"void main(\n"
			"	float4 aPosition,\n"
			"	float2 aTexcoord,\n"
			"	float4 aColor,\n"
			"	float aFog,\n"
			"	out float4 vPosition : POSITION,\n"
			"	out float2 vTexcoord : TEXCOORD0,\n"
			"	out float vFog : TEXCOORD1,\n"
			"	out float4 vColor : COLOR)\n"
			"{\n"
			"	float w = aPosition.w;\n"
			"	vPosition = float4(aPosition.x * (2.0 / 960.0) - w, w - aPosition.y * (2.0 / 544.0), aPosition.z, w);\n"
			"	vTexcoord = aTexcoord;\n"
			"	vFog = aFog;\n"
			"	vColor = aColor;\n"
			"}\n");
	}

	int clear = (key >> 10) & 1;
	int fog = (key >> 11) & 1;
	int textured = key & 1;
	int tfunc = (key >> 1) & 7;
	int tcc = (key >> 4) & 1;
	int dbl = (key >> 5) & 1;
	int atest = (key >> 6) & 1;
	int afunc = (key >> 7) & 7;

	int n = snprintf(buf, size,
		"float4 main(float2 vTexcoord : TEXCOORD0, float4 vColor : COLOR%s%s%s%s) : COLOR\n"
		"{\n"
		"	float4 v = vColor;\n"
		"	float4 c = v;\n",
		textured ? ", uniform sampler2D uTexture" : "",
		(textured && tfunc == 2) ? ", uniform float4 uEnv" : "",
		atest ? ", uniform float uAlphaRef" : "",
		fog ? ", float vFog : TEXCOORD1, uniform float4 uFogColor" : "");

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

	// Fog after texturing, on the colour only
	if (fog)
		n += snprintf(buf + n, size - n, "	c.rgb = lerp(uFogColor.rgb, saturate(c.rgb), vFog);\n");

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

int geSgxCgAllKeys(uint32_t *keys, int max) {
	int n = 0;
	uint32_t textured, tfunc, tcc, dbl, atest, fog;

#define ADD(k) do { if (n < max) keys[n] = (k); n++; } while (0)
	ADD(GE_SGX_SHADER_VERTEX);
	ADD(1u << 10); // clear
	// alpha test: off, or one of the functions but ALWAYS (= off)
	static const int8_t atests[8] = { -1, 0, 2, 3, 4, 5, 6, 7 };
	for (fog = 0; fog < 2; fog++)
	for (textured = 0; textured < 2; textured++) {
		for (tfunc = 0; tfunc < (textured ? 5u : 1u); tfunc++) {
			for (tcc = 0; tcc < (textured ? 2u : 1u); tcc++) {
				for (dbl = 0; dbl < 2; dbl++) {
					for (atest = 0; atest < 8; atest++) {
						uint32_t key = fog << 11;
						if (textured)
							key |= 1 | (tfunc << 1) | (tcc << 4);
						if (dbl)
							key |= 1 << 5;
						if (atests[atest] >= 0)
							key |= (1 << 6) | ((uint32_t)atests[atest] << 7);
						ADD(key);
					}
				}
			}
		}
	}
#undef ADD
	return n;
}

void geSgxCgPath(uint32_t key, char *path, int size) {
	snprintf(path, size, GE_SGX_CG_DIR "/v%d_%04x.gxp", GE_SGX_CG_VERSION, (unsigned)key);
}

SceGxmProgram *geSgxCgLoadCached(uint32_t key) {
	char path[96];
	geSgxCgPath(key, path, sizeof(path));

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

void geSgxCgSave(uint32_t key, const SceGxmProgram *program, uint32_t size) {
	char path[96];
	sceIoMkdir("ux0:data/adrenaline", 0777);
	sceIoMkdir(GE_SGX_CG_DIR, 0777);
	geSgxCgPath(key, path, sizeof(path));
	SceUID fd = sceIoOpen(path, SCE_O_WRONLY | SCE_O_CREAT | SCE_O_TRUNC, 0777);
	if (fd < 0)
		return;
	int ok = sceIoWrite(fd, program, size) == (int)size;
	sceIoClose(fd);
	if (!ok)
		sceIoRemove(path);
}

static int findExport(uint32_t nid, void *func) {
	uintptr_t addr = 0;
	if (taiGetModuleExportFunc("SceShaccCg", SHACCCG_LIB_NID, nid, &addr) < 0 || !addr)
		return 0;
	*(uintptr_t *)func = addr;
	return 1;
}

int geSgxCgInit(SceUID (* load_fn)(const char *path), void *(* alloc)(unsigned int), void (* release)(void *)) {
	static const char *paths[] = {
		"ur0:data/libshacccg.suprx",
		"ur0:data/external/libshacccg.suprx",
	};

	int i;
	cg.modid = -1;
	for (i = 0; i < 2 && cg.modid < 0; i++)
		cg.modid = load_fn(paths[i]);
	if (cg.modid < 0)
		return cg.modid;

	if (!findExport(0x3B58AFA0, &cg.InitializeCompileOptions) ||
	    !findExport(0x66814F35, &cg.CompileProgram) ||
	    !findExport(0x6F01D573, &cg.SetDefaultAllocator) ||
	    !findExport(0xA8C2C1C8, &cg.InitializeCallbackList) ||
	    !findExport(0xAA82EF0C, &cg.DestroyCompileOutput) ||
	    !findExport(0x95F57A23, &cg.ReleaseCompiler)) {
		sceKernelStopUnloadModule(cg.modid, 0, NULL, 0, NULL, NULL);
		return -2;
	}

	cg.SetDefaultAllocator(alloc, release);
	return 1;
}

void geSgxCgTerm(void) {
	if (cg.ReleaseCompiler)
		cg.ReleaseCompiler();
	if (cg.modid >= 0)
		sceKernelStopUnloadModule(cg.modid, 0, NULL, 0, NULL, NULL);
	memset(&cg, 0, sizeof(cg));
	cg.modid = -1;
}

static SceShaccCgSourceFile *openFile(const char *fileName, const SceShaccCgSourceLocation *includedFrom,
	const SceShaccCgCompileOptions *compileOptions, const char **errorString) {
	return &source_file;
}

SceGxmProgram *geSgxCgCompile(uint32_t key, uint32_t *size, void (* log)(uint32_t key, int line, const char *msg)) {
	static char src[4096];
	int len = geSgxCgSource(key, src, sizeof(src));
	if (len <= 0 || len >= (int)sizeof(src) || !cg.CompileProgram)
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
				*size = out->programSize;
			}
		}
	} else if (out && log) {
		int i;
		for (i = 0; i < out->diagnosticCount && i < 4; i++) {
			const SceShaccCgDiagnosticMessage *m = &out->diagnostics[i];
			log(key, m->location ? (int)m->location->lineNumber : -1, m->message ? m->message : "?");
		}
	}

	if (out)
		cg.DestroyCompileOutput(out);
	return program;
}
