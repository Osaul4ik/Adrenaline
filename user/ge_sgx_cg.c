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

// The GPU transform (see ge_sgx_cg.h): PPSSPP's transform and lighting in a vertex shader
static int vertexSource(uint32_t key, char *buf, int size) {
	int tc = key & GE_SGX_VS_TC, col = key & GE_SGX_VS_COLOR, nrm = key & GE_SGX_VS_NORMAL;
	int col16 = key & GE_SGX_VS_COLOR16;
	int w4 = key & (GE_SGX_VS_W4 | GE_SGX_VS_W8), w8 = key & GE_SGX_VS_W8, lit = key & GE_SGX_VS_LIGHT;
	int n = 0;

#define S(...) do { n += snprintf(buf + n, size > n ? size - n : 0, __VA_ARGS__); } while (0)
	S("void main(\n	float3 aPosition,\n");
	if (tc) S("	float2 aTexcoord,\n");
	if (col) S("	float4 aColor,\n");
	if (col16) S("	float aColor,\n"); // the 16 bit value
	if (nrm) S("	float3 aNormal,\n");
	if (w4) S("	float4 aWeight0,\n");
	if (w8) S("	float4 aWeight1,\n");
	S("	uniform float4 uC[%d] : BUFFER[0],\n	uniform float4 uO[%d] : BUFFER[1],\n", GE_SGX_UC_COUNT, GE_SGX_UO_COUNT);
	S("	out float4 vPosition : POSITION,\n	out float2 vTexcoord : TEXCOORD0,\n	out float vFog : TEXCOORD1,\n	out float4 vColor : COLOR)\n{\n");

	// Model space, with the bones
	S("	float3 p = aPosition * uO[%d].x;\n", GE_SGX_UO_SCALE);
	if (nrm)
		S("	float3 n = aNormal * uO[%d].y * uC[%d].w;\n", GE_SGX_UO_SCALE, GE_SGX_UC_FLAGS);
	else
		S("	float3 n = float3(0.0, 0.0, uC[%d].w);\n", GE_SGX_UC_FLAGS);
	if (w4) {
		S("	float4 w0 = aWeight0 * uO[%d].w * uO[%d];\n", GE_SGX_UO_SCALE, GE_SGX_UO_WMASK);
		if (w8)
			S("	float4 w1 = aWeight1 * uO[%d].w * uO[%d];\n", GE_SGX_UO_SCALE, GE_SGX_UO_WMASK + 1);
		S("	float3 sp = float3(0.0, 0.0, 0.0), sn = float3(0.0, 0.0, 0.0);\n");
		// Unrolled: constant indices only
		int i;
		for (i = 0; i < (w8 ? 8 : 4); i++) {
			int b = GE_SGX_UO_BONES + i * 4;
			const char *wt = i < 4 ? "w0" : "w1";
			S("	sp += (uO[%d].xyz * p.x + uO[%d].xyz * p.y + uO[%d].xyz * p.z + uO[%d].xyz) * %s.%c;\n", b, b + 1, b + 2, b + 3, wt, "xyzw"[i & 3]);
			S("	sn += (uO[%d].xyz * n.x + uO[%d].xyz * n.y + uO[%d].xyz * n.z) * %s.%c;\n", b, b + 1, b + 2, wt, "xyzw"[i & 3]);
		}
		S("	p = sp;\n	n = sn;\n");
	}

	// World, view, projection, the PSP viewport (pixels times w), depth clamped like the PSP
	S("	float3 wp = uO[0].xyz * p.x + uO[1].xyz * p.y + uO[2].xyz * p.z + uO[3].xyz;\n");
	S("	float3 wn = uO[0].xyz * n.x + uO[1].xyz * n.y + uO[2].xyz * n.z;\n");
	S("	float nl = dot(wn, wn);\n	wn = nl > 0.0 ? wn * rsqrt(nl) : float3(0.0, 0.0, 1.0);\n");
	S("	float3 vp = uC[%d].xyz * wp.x + uC[%d].xyz * wp.y + uC[%d].xyz * wp.z + uC[%d].xyz;\n",
		GE_SGX_UC_VIEW, GE_SGX_UC_VIEW + 1, GE_SGX_UC_VIEW + 2, GE_SGX_UC_VIEW + 3);
	S("	float4 c = uC[0] * vp.x + uC[1] * vp.y + uC[2] * vp.z + uC[3];\n");
	S("	float4 vpx = uC[%d], vpz = uC[%d];\n", GE_SGX_UC_VP, GE_SGX_UC_VPZ);
	S("	float x = vpx.x * c.x + vpx.y * c.w;\n	float y = vpx.z * c.y + vpx.w * c.w;\n");
	S("	float z = vpz.x * c.z + vpz.y * c.w;\n	if (c.w > 0.0)\n		z = clamp(z, 0.0, c.w);\n");
	S("	vPosition = float4(x * (2.0 / 960.0) - c.w, c.w - y * (2.0 / 544.0), z, c.w);\n");

	// Texture coordinates
	S("	float4 fl = uC[%d];\n", GE_SGX_UC_FLAGS);
	S("	float2 t = %s;\n", tc ? "aTexcoord * uO[4].z" : "float2(0.0, 0.0)");
	S("	if (fl.x < 0.5) {\n		vTexcoord = t * uO[%d].xy + uO[%d].zw;\n", GE_SGX_UO_UV, GE_SGX_UO_UV);
	S("	} else if (fl.x < 1.5) {\n");
	S("		float3 s = fl.y < 0.5 ? p : (fl.y < 1.5 ? float3(t, 0.0) : (fl.y < 2.5 ? normalize(n) : n));\n");
	S("		float3 q = uC[%d].xyz * s.x + uC[%d].xyz * s.y + uC[%d].xyz * s.z + uC[%d].xyz;\n",
		GE_SGX_UC_TGEN, GE_SGX_UC_TGEN + 1, GE_SGX_UC_TGEN + 2, GE_SGX_UC_TGEN + 3);
	S("		vTexcoord = q.xy / (abs(q.z) > 0.000001 ? q.z : 1.0);\n");
	S("	} else {\n		vTexcoord = (float2(dot(uC[%d].xyz, wn), dot(uC[%d].xyz, wn)) + 1.0) * 0.5;\n	}\n",
		GE_SGX_UC_ENVLS, GE_SGX_UC_ENVLS + 1);

	// Colour; 16 bit ones are decoded here (exact in float: integers below 65536)
	if (col16) {
		S("	float cv = aColor, cf = uC[%d].y;\n	float4 ic;\n", GE_SGX_UC_MATUPD);
		S("	if (cf < 0.5)\n		ic = float4(fmod(cv, 32.0) / 31.0, fmod(floor(cv / 32.0), 64.0) / 63.0, floor(cv / 2048.0) / 31.0, 1.0);\n");
		S("	else if (cf < 1.5)\n		ic = float4(fmod(cv, 32.0) / 31.0, fmod(floor(cv / 32.0), 32.0) / 31.0, fmod(floor(cv / 1024.0), 32.0) / 31.0, floor(cv / 32768.0));\n");
		S("	else\n		ic = float4(fmod(cv, 16.0), fmod(floor(cv / 16.0), 16.0), fmod(floor(cv / 256.0), 16.0), floor(cv / 4096.0)) / 15.0;\n");
	} else {
		S("	float4 ic = %s;\n", col ? "aColor" : "uC[11]");
	}
	if (lit) {
		S("	float3 ambC = lerp(uC[%d].rgb, ic.rgb, uC[%d].w);\n", GE_SGX_UC_MATAMB, GE_SGX_UC_MATDIF);
		S("	float ambA = lerp(uC[%d].a, ic.a, uC[%d].w);\n", GE_SGX_UC_MATAMB, GE_SGX_UC_MATDIF);
		S("	float3 difC = lerp(uC[%d].rgb, ic.rgb, uC[%d].w);\n", GE_SGX_UC_MATDIF, GE_SGX_UC_MATSPC);
		S("	float3 spcC = lerp(uC[%d].rgb, ic.rgb, uC[%d].x);\n", GE_SGX_UC_MATSPC, GE_SGX_UC_MATUPD);
		S("	float sexp = uC[%d].w;\n", GE_SGX_UC_EMISSIVE);
		S("	float3 sum = uC[%d].rgb * ambC + uC[%d].rgb;\n", GE_SGX_UC_AMBIENT, GE_SGX_UC_EMISSIVE);
		S("	float3 spec = float3(0.0, 0.0, 0.0);\n");
		int l;
		for (l = 0; l < 4; l++) {
			int o = GE_SGX_UC_LIGHT + l * GE_SGX_UC_LIGHT_SIZE;
			S("	{\n		float4 L0 = uC[%d];\n", o);
			S("		if (L0.w > -0.5) {\n");
			S("			float3 to = L0.xyz;\n			float dist = 1.0;\n");
			S("			if (L0.w > 0.5) {\n				to = L0.xyz - wp;\n				dist = length(to);\n				to = dist > 0.0 ? to / dist : to;\n			}\n");
			S("			float4 L1 = uC[%d], L2 = uC[%d];\n", o + 1, o + 2);
			S("			float d = max(dot(to, wn), 0.0);\n");
			S("			if (L1.w > 1.5)\n				d = d > 0.0 ? pow(d, sexp) : 0.0;\n");
			S("			float sc = 1.0;\n");
			S("			if (L0.w > 0.5) {\n				float den = L2.x + L2.y * dist + L2.z * dist * dist;\n				sc = den > 0.0 ? saturate(1.0 / den) : 1.0;\n");
			S("				if (L0.w > 1.5) {\n					float ang = dot(to, L1.xyz);\n					sc = ang >= uC[%d].x ? sc * (ang > 0.0 ? pow(ang, L2.w) : 0.0) : 0.0;\n				}\n			}\n", o + 3);
			S("			sum += (uC[%d].rgb * ambC + uC[%d].rgb * difC * d) * sc;\n", o + 4, o + 5);
			S("			if (L1.w > 0.5 && L1.w < 1.5) {\n				float3 h = normalize(to + float3(0.0, 0.0, 1.0));\n				float sd = dot(h, wn);\n");
			S("				if (sd > 0.0)\n					spec += uC[%d].rgb * spcC * (pow(sd, sexp) * sc);\n			}\n", o + 6);
			S("		}\n	}\n");
		}
		S("	vColor = float4(saturate(sum + spec), saturate(uC[%d].a * ambA));\n", GE_SGX_UC_AMBIENT);
	} else {
		S("	vColor = ic;\n");
	}

	// Fog from the view depth
	S("	vFog = fl.z > 0.5 ? saturate((vp.z + vpz.z) * vpz.w) : 1.0;\n}\n");
#undef S
	return n;
}

int geSgxCgSource(uint32_t key, char *buf, int size) {
	if ((key & 0xF000u) == GE_SGX_VS_BASE)
		return vertexSource(key, buf, size);
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
	// GPU transform: every attribute combination, with and without lighting
	{
		uint32_t v, wc, lit;
		for (lit = 0; lit < 2; lit++)
			for (wc = 0; wc < 3; wc++)
				for (v = 0; v < 8; v++) {
					uint32_t k = GE_SGX_VS_BASE | v | (wc == 1 ? GE_SGX_VS_W4 : wc == 2 ? GE_SGX_VS_W8 : 0) | (lit ? GE_SGX_VS_LIGHT : 0);
					ADD(k);
					if (v & GE_SGX_VS_COLOR)
						ADD((k & ~GE_SGX_VS_COLOR) | GE_SGX_VS_COLOR16);
				}
	}
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
	static char src[8192];
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
	options.targetProfile = geSgxCgIsVertex(key) ? SCE_SHACCCG_PROFILE_VP : SCE_SHACCCG_PROFILE_FP;
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
