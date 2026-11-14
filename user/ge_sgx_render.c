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
	SGX renderer: draws a frame's draw records with GXM, in the draw thread
	(the GXM context of vita2d). Its own shader patcher holds the programs
	made from the generated shaders (ge_sgx_shader.c): one vertex program,
	one fragment program per fragment shader and blend state.
*/

#include <psp2/kernel/processmgr.h>
#include <psp2/kernel/sysmem.h>

#include <stdlib.h>
#include <string.h>

#include <vita2d.h>

extern SceGxmContext *_vita2d_context;

#include "ge_sgx.h"
#include "ge_sgx_render.h"
#include "ge_sgx_shader.h"
#include "ge_sgx_tex.h"
#include "ge_sgx_blend.h"

#define PATCHER_BUFFER_SIZE (64 * 1024)
#define PATCHER_VERTEX_USSE_SIZE (64 * 1024)
#define PATCHER_FRAGMENT_USSE_SIZE (512 * 1024)

#define MAX_FRAGMENT_PROGRAMS 1024

typedef struct {
	uint32_t key;
	uint32_t blend;
	const SceGxmProgram *vgxp; // the vertex program it was made for
	SceGxmFragmentProgram *program;
	const SceGxmProgramParameter *alpha_ref;
	const SceGxmProgramParameter *env;
	const SceGxmProgramParameter *fog_color;
	const SceGxmProgramParameter *fix_c, *fix_a;
} FragmentProgram;

typedef struct {
	uint32_t key;
	SceGxmShaderPatcherId id;
	const SceGxmProgram *gxp;
} Registered;

static SceGxmShaderPatcher *patcher = NULL;
static SceGxmVertexProgram *vertex_program = NULL;
static const SceGxmProgram *vertex_gxp = NULL;
static FragmentProgram fragment_programs[MAX_FRAGMENT_PROGRAMS];
static int nfragment_programs = 0;
static Registered registered[MAX_FRAGMENT_PROGRAMS];
static int nregistered = 0;

static void *gpuAlloc(SceKernelMemBlockType type, uint32_t size, SceGxmMemoryAttribFlags attr) {
	size = (size + 0xFFF) & ~0xFFFu;
	SceUID uid = sceKernelAllocMemBlock("SceGeSgxGpu", type, size, NULL);
	if (uid < 0)
		return NULL;
	void *mem = NULL;
	sceKernelGetMemBlockBase(uid, &mem);
	if (attr && sceGxmMapMemory(mem, size, attr) < 0) {
		sceKernelFreeMemBlock(uid);
		return NULL;
	}
	return mem;
}

static void *patcherHostAlloc(void *user, SceSize size) {
	return malloc(size);
}

static void patcherHostFree(void *user, void *mem) {
	free(mem);
}

int geSgxRenderInit(void) {
	if (patcher)
		return 0;

	void *buffer = gpuAlloc(SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE, PATCHER_BUFFER_SIZE, SCE_GXM_MEMORY_ATTRIB_READ);
	void *vusse = gpuAlloc(SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE, PATCHER_VERTEX_USSE_SIZE, 0);
	void *fusse = gpuAlloc(SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE, PATCHER_FRAGMENT_USSE_SIZE, 0);
	if (!buffer || !vusse || !fusse)
		return -1;

	unsigned int vusse_offset = 0, fusse_offset = 0;
	if (sceGxmMapVertexUsseMemory(vusse, PATCHER_VERTEX_USSE_SIZE, &vusse_offset) < 0)
		return -2;
	if (sceGxmMapFragmentUsseMemory(fusse, PATCHER_FRAGMENT_USSE_SIZE, &fusse_offset) < 0)
		return -3;

	SceGxmShaderPatcherParams params;
	memset(&params, 0, sizeof(params));
	params.hostAllocCallback = patcherHostAlloc;
	params.hostFreeCallback = patcherHostFree;
	params.bufferMem = buffer;
	params.bufferMemSize = PATCHER_BUFFER_SIZE;
	params.vertexUsseMem = vusse;
	params.vertexUsseMemSize = PATCHER_VERTEX_USSE_SIZE;
	params.vertexUsseOffset = vusse_offset;
	params.fragmentUsseMem = fusse;
	params.fragmentUsseMemSize = PATCHER_FRAGMENT_USSE_SIZE;
	params.fragmentUsseOffset = fusse_offset;

	int res = sceGxmShaderPatcherCreate(&params, &patcher);
	if (res < 0) {
		patcher = NULL;
		return res;
	}
	return 0;
}

// The vertex program, once its shader is there
static int vertexProgram(void) {
	if (vertex_program)
		return 1;

	const SceGxmProgram *gxp = geSgxShaderGet(GE_SGX_SHADER_VERTEX);
	if (!gxp)
		return 0;

	SceGxmShaderPatcherId id;
	if (sceGxmShaderPatcherRegisterProgram(patcher, gxp, &id) < 0)
		return 0;

	const SceGxmProgramParameter *pos = sceGxmProgramFindParameterByName(gxp, "aPosition");
	const SceGxmProgramParameter *tc = sceGxmProgramFindParameterByName(gxp, "aTexcoord");
	const SceGxmProgramParameter *col = sceGxmProgramFindParameterByName(gxp, "aColor");
	const SceGxmProgramParameter *fog = sceGxmProgramFindParameterByName(gxp, "aFog");
	if (!pos || !tc || !col || !fog) {
		GeSgxLog("GeSgx render: vertex program parameters missing\n");
		return 0;
	}

	// GeSgxVertex
	SceGxmVertexAttribute attrs[4];
	memset(attrs, 0, sizeof(attrs));
	attrs[0].offset = 0;
	attrs[0].format = SCE_GXM_ATTRIBUTE_FORMAT_F32;
	attrs[0].componentCount = 4;
	attrs[0].regIndex = sceGxmProgramParameterGetResourceIndex(pos);
	attrs[1].offset = 16;
	attrs[1].format = SCE_GXM_ATTRIBUTE_FORMAT_F32;
	attrs[1].componentCount = 2;
	attrs[1].regIndex = sceGxmProgramParameterGetResourceIndex(tc);
	attrs[2].offset = 24;
	attrs[2].format = SCE_GXM_ATTRIBUTE_FORMAT_U8N;
	attrs[2].componentCount = 4;
	attrs[2].regIndex = sceGxmProgramParameterGetResourceIndex(col);
	attrs[3].offset = 28;
	attrs[3].format = SCE_GXM_ATTRIBUTE_FORMAT_U8N; // secondary colour, fog
	attrs[3].componentCount = 4;
	attrs[3].regIndex = sceGxmProgramParameterGetResourceIndex(fog);

	SceGxmVertexStream stream;
	stream.stride = sizeof(GeSgxVertex);
	stream.indexSource = SCE_GXM_INDEX_SOURCE_INDEX_16BIT;

	int res = sceGxmShaderPatcherCreateVertexProgram(patcher, id, attrs, 4, &stream, 1, &vertex_program);
	if (res < 0) {
		GeSgxLog("GeSgx render: vertex program %08X\n", (unsigned)res);
		vertex_program = NULL;
		return 0;
	}
	vertex_gxp = gxp;
	return 1;
}

static int gxmFactor(uint8_t f) {
	switch (f) {
		case GE_SGX_BF_ZERO: return SCE_GXM_BLEND_FACTOR_ZERO;
		case GE_SGX_BF_SRC_COLOR: return SCE_GXM_BLEND_FACTOR_SRC_COLOR;
		case GE_SGX_BF_INV_SRC_COLOR: return SCE_GXM_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
		case GE_SGX_BF_DST_COLOR: return SCE_GXM_BLEND_FACTOR_DST_COLOR;
		case GE_SGX_BF_INV_DST_COLOR: return SCE_GXM_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
		case GE_SGX_BF_SRC_ALPHA: return SCE_GXM_BLEND_FACTOR_SRC_ALPHA;
		case GE_SGX_BF_INV_SRC_ALPHA: return SCE_GXM_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		case GE_SGX_BF_DST_ALPHA: return SCE_GXM_BLEND_FACTOR_DST_ALPHA;
		case GE_SGX_BF_INV_DST_ALPHA: return SCE_GXM_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
		default: return SCE_GXM_BLEND_FACTOR_ONE;
	}
}

static int blendFunc(uint32_t eq) {
	switch (eq) {
		case 1: return SCE_GXM_BLEND_FUNC_SUBTRACT;
		case 2: return SCE_GXM_BLEND_FUNC_REVERSE_SUBTRACT;
		case 3: return SCE_GXM_BLEND_FUNC_MIN;
		case 4: return SCE_GXM_BLEND_FUNC_MAX;
		default: return SCE_GXM_BLEND_FUNC_ADD; // absolute difference: closest
	}
}

/*
	Blend key of a fragment program: clear: 0x80000000 | GXM colour mask;
	else 0x40000000 if blended | function | source << 4 | destination << 8,
	channels not written << 12, alpha written 1 << 15.
*/
static uint32_t blendKey(const GeSgxDraw *d, GeSgxBlend *b) {
	if (d->flags & GE_SGX_DRAW_CLEAR) {
		uint32_t mask = 0;
		if (d->flags & GE_SGX_DRAW_WRITE_COLOR)
			mask |= SCE_GXM_COLOR_MASK_R | SCE_GXM_COLOR_MASK_G | SCE_GXM_COLOR_MASK_B;
		if (d->flags & GE_SGX_DRAW_WRITE_ALPHA)
			mask |= SCE_GXM_COLOR_MASK_A;
		return 0x80000000u | mask;
	}
	uint32_t key = (uint32_t)(d->cmask & 7) << 12; // channels not written
	int alpha = (d->flags & GE_SGX_DRAW_WRITE_ALPHA) != 0;
	if (d->flags & GE_SGX_DRAW_BLEND) {
		geSgxBlendSetup(d, b);
		if (b->alpha_bogus)
			alpha = 0;
		key |= 0x40000000u | (uint32_t)blendFunc(b->eq) | ((uint32_t)gxmFactor(b->src) << 4) | ((uint32_t)gxmFactor(b->dst) << 8);
	} else {
		b->fix = 0;
	}
	if (alpha)
		key |= 1u << 15;
	return key;
}

static const SceGxmProgram *registeredGxp(uint32_t key, SceGxmShaderPatcherId *id) {
	int i;
	for (i = 0; i < nregistered; i++) {
		if (registered[i].key == key) {
			*id = registered[i].id;
			return registered[i].gxp;
		}
	}
	if (nregistered >= MAX_FRAGMENT_PROGRAMS)
		return NULL;

	const SceGxmProgram *gxp = geSgxShaderGet(key);
	if (!gxp)
		return NULL;
	if (sceGxmShaderPatcherRegisterProgram(patcher, gxp, id) < 0)
		return NULL;
	registered[nregistered].key = key;
	registered[nregistered].id = *id;
	registered[nregistered].gxp = gxp;
	nregistered++;
	return gxp;
}

static FragmentProgram *fragmentProgram(const GeSgxDraw *d, const SceGxmProgram *vgxp, GeSgxBlend *bl) {
	uint32_t key = geSgxShaderKey(d);
	uint32_t blend = blendKey(d, bl);
	int i;

	for (i = 0; i < nfragment_programs; i++) {
		if (fragment_programs[i].key == key && fragment_programs[i].blend == blend && fragment_programs[i].vgxp == vgxp)
			return &fragment_programs[i];
	}
	if (nfragment_programs >= MAX_FRAGMENT_PROGRAMS)
		return NULL;

	SceGxmShaderPatcherId id;
	const SceGxmProgram *gxp = registeredGxp(key, &id);
	if (!gxp)
		return NULL;

	/*
		Alpha is the PSP's framebuffer alpha (its stencil): written by draws
		without stencil test and clears of the stencil, read by DSTALPHA
		blending and framebuffer textures. Renderings are shown without it.
	*/
	SceGxmBlendInfo info;
	memset(&info, 0, sizeof(info));
	info.colorMask = SCE_GXM_COLOR_MASK_R | SCE_GXM_COLOR_MASK_G | SCE_GXM_COLOR_MASK_B;
	info.colorFunc = SCE_GXM_BLEND_FUNC_NONE;
	info.alphaFunc = SCE_GXM_BLEND_FUNC_NONE;
	info.colorSrc = info.alphaSrc = SCE_GXM_BLEND_FACTOR_ONE;
	info.colorDst = info.alphaDst = SCE_GXM_BLEND_FACTOR_ZERO;
	if (blend & 0x80000000u) {
		info.colorMask = blend & SCE_GXM_COLOR_MASK_ALL;
	} else {
		if (blend & (1u << 15))
			info.colorMask |= SCE_GXM_COLOR_MASK_A;
		if (blend & 0x40000000u) {
			info.colorFunc = blend & 0xF;
			info.colorSrc = (blend >> 4) & 0xF;
			info.colorDst = (blend >> 8) & 0xF;
			// The PSP writes the source alpha as it is
			info.alphaFunc = SCE_GXM_BLEND_FUNC_ADD;
			info.alphaSrc = SCE_GXM_BLEND_FACTOR_ONE;
			info.alphaDst = SCE_GXM_BLEND_FACTOR_ZERO;
		}
		if (blend & (1u << 12))
			info.colorMask &= ~SCE_GXM_COLOR_MASK_R;
		if (blend & (1u << 13))
			info.colorMask &= ~SCE_GXM_COLOR_MASK_G;
		if (blend & (1u << 14))
			info.colorMask &= ~SCE_GXM_COLOR_MASK_B;
	}
	const SceGxmBlendInfo *pinfo = &info;

	SceGxmFragmentProgram *program = NULL;
	int res = sceGxmShaderPatcherCreateFragmentProgram(patcher, id, SCE_GXM_OUTPUT_REGISTER_FORMAT_UCHAR4,
		SCE_GXM_MULTISAMPLE_NONE, pinfo, vgxp, &program);
	if (res < 0) {
		GeSgxLog("GeSgx render: fragment program %04X/%08X: %08X\n", (unsigned)key, (unsigned)blend, (unsigned)res);
		return NULL;
	}

	FragmentProgram *fp = &fragment_programs[nfragment_programs++];
	fp->key = key;
	fp->blend = blend;
	fp->vgxp = vgxp;
	fp->program = program;
	fp->alpha_ref = sceGxmProgramFindParameterByName(gxp, "uAlphaRef");
	fp->env = sceGxmProgramFindParameterByName(gxp, "uEnv");
	fp->fog_color = sceGxmProgramFindParameterByName(gxp, "uFogColor");
	fp->fix_c = sceGxmProgramFindParameterByName(gxp, "uFixC");
	fp->fix_a = sceGxmProgramFindParameterByName(gxp, "uFixA");
	return fp;
}

/*
	Vertex programs of the GPU transform: one per vertex program key and
	vertex shape, the attributes read straight from the PSP's vertex layout.
*/
#define MAX_GPU_PROGRAMS 128
#define VTYPE_ATTRIBUTES ((3u << 0) | (7u << 2) | (3u << 5) | (3u << 7) | (3u << 9) | (7u << 14))

typedef struct {
	uint32_t vkey, vtype;
	const SceGxmProgram *gxp;
	SceGxmVertexProgram *program; // NULL: could not be made
} GpuProgram;

static GpuProgram gpu_programs[MAX_GPU_PROGRAMS];
static int ngpu_programs = 0;
volatile int ge_sgx_gpu_failed = 0;
volatile uint32_t ge_sgx_gpu_bad[GE_SGX_GPU_BAD_MAX];
volatile int ge_sgx_gpu_nbad = 0;

static int attribute(SceGxmVertexAttribute *a, const SceGxmProgram *gxp, const char *name, uint32_t fmt_class, uint32_t fmt, uint32_t comps, uint32_t offset) {
	const SceGxmProgramParameter *p = sceGxmProgramFindParameterByName(gxp, name);
	if (!p)
		return 0; // not used by the program (optimised out)
	static const uint8_t unsigned_formats[4] = { 0, SCE_GXM_ATTRIBUTE_FORMAT_U8, SCE_GXM_ATTRIBUTE_FORMAT_U16, SCE_GXM_ATTRIBUTE_FORMAT_F32 };
	static const uint8_t signed_formats[4] = { 0, SCE_GXM_ATTRIBUTE_FORMAT_S8, SCE_GXM_ATTRIBUTE_FORMAT_S16, SCE_GXM_ATTRIBUTE_FORMAT_F32 };
	memset(a, 0, sizeof(*a));
	a->streamIndex = 0;
	a->offset = (uint16_t)offset;
	if (fmt_class == 2)
		a->format = SCE_GXM_ATTRIBUTE_FORMAT_U8N;
	else
		a->format = fmt_class ? signed_formats[fmt & 3] : unsigned_formats[fmt & 3];
	a->componentCount = (uint8_t)comps;
	a->regIndex = (uint16_t)sceGxmProgramParameterGetResourceIndex(p);
	return 1;
}

static GpuProgram *gpuProgram(const GeSgxDraw *d, int *waiting) {
	uint32_t vtype = d->vtype & VTYPE_ATTRIBUTES;
	int i;
	for (i = 0; i < ngpu_programs; i++) {
		if (gpu_programs[i].vkey == d->vkey && gpu_programs[i].vtype == vtype)
			return gpu_programs[i].program ? &gpu_programs[i] : NULL;
	}
	if (ngpu_programs >= MAX_GPU_PROGRAMS)
		return NULL;

	SceGxmShaderPatcherId id;
	const SceGxmProgram *gxp = registeredGxp(d->vkey, &id);
	if (!gxp) {
		if (geSgxShaderFailed(d->vkey) && !ge_sgx_gpu_failed) {
			ge_sgx_gpu_failed = 1;
			GeSgxLog("GeSgx render: GPU transform program %04X not available, CPU transform from now on\n", (unsigned)d->vkey);
		}
		*waiting = 1;
		return NULL;
	}

	GeSgxVertexLayout l;
	geSgxVertexLayout(vtype, &l);
	static const uint8_t comp_size[4] = { 0, 1, 2, 4 };
	SceGxmVertexAttribute attrs[6];
	int n = 0;
	n += attribute(&attrs[n], gxp, "aPosition", 1, l.pos_fmt, 3, l.pos_off);
	if (l.tc_fmt)
		n += attribute(&attrs[n], gxp, "aTexcoord", 0, l.tc_fmt, 2, l.tc_off);
	if (l.color_fmt == 7)
		n += attribute(&attrs[n], gxp, "aColor", 2, 0, 4, l.color_off);
	else if (l.color_fmt)
		n += attribute(&attrs[n], gxp, "aColor", 0, 2, 1, l.color_off); // U16, decoded by the program
	if (l.nrm_fmt)
		n += attribute(&attrs[n], gxp, "aNormal", 1, l.nrm_fmt, 3, l.nrm_off);
	if (l.wt_fmt) {
		n += attribute(&attrs[n], gxp, "aWeight0", 0, l.wt_fmt, l.weights < 4 ? l.weights : 4, 0);
		if (l.weights > 4)
			n += attribute(&attrs[n], gxp, "aWeight1", 0, l.wt_fmt, l.weights - 4, 4 * comp_size[l.wt_fmt]);
	}

	SceGxmVertexStream stream;
	stream.stride = (uint16_t)l.size;
	stream.indexSource = SCE_GXM_INDEX_SOURCE_INDEX_16BIT;

	GpuProgram *g = &gpu_programs[ngpu_programs++];
	g->vkey = d->vkey;
	g->vtype = vtype;
	g->gxp = gxp;
	g->program = NULL;
	int res = sceGxmShaderPatcherCreateVertexProgram(patcher, id, attrs, n, &stream, 1, &g->program);
	if (res < 0) {
		g->program = NULL;
		if (ge_sgx_gpu_nbad < GE_SGX_GPU_BAD_MAX)
			ge_sgx_gpu_bad[ge_sgx_gpu_nbad++] = vtype;
		else
			ge_sgx_gpu_failed = 1;
		GeSgxLog("GeSgx render: GPU vertex program %04X vtype %06X stride %u: %08X, this vertex type on the CPU\n", (unsigned)d->vkey, (unsigned)vtype, (unsigned)l.size, (unsigned)res);
		return NULL;
	}
	return g;
}

static SceGxmDepthFunc depthFunc(uint8_t ztest) {
	switch (ztest) {
		case 0: return SCE_GXM_DEPTH_FUNC_NEVER;
		case 2: return SCE_GXM_DEPTH_FUNC_EQUAL;
		case 3: return SCE_GXM_DEPTH_FUNC_NOT_EQUAL;
		case 4: return SCE_GXM_DEPTH_FUNC_LESS;
		case 5: return SCE_GXM_DEPTH_FUNC_LESS_EQUAL;
		case 6: return SCE_GXM_DEPTH_FUNC_GREATER;
		case 7: return SCE_GXM_DEPTH_FUNC_GREATER_EQUAL;
		default: return SCE_GXM_DEPTH_FUNC_ALWAYS;
	}
}

static void setColorUniform(void *uniforms, const SceGxmProgramParameter *p, uint32_t c) {
	float v[4] = { (c & 0xFF) / 255.0f, ((c >> 8) & 0xFF) / 255.0f, ((c >> 16) & 0xFF) / 255.0f, 1.0f };
	sceGxmSetUniformDataF(uniforms, p, 0, 4, v);
}

static SceGxmPrimitiveType primitive(uint8_t prim) {
	switch (prim) {
		case GE_SGX_DRAW_LINES: return SCE_GXM_PRIMITIVE_LINES;
		case GE_SGX_DRAW_POINTS: return SCE_GXM_PRIMITIVE_POINTS;
		default: return SCE_GXM_PRIMITIVE_TRIANGLES;
	}
}

static SceGxmStencilFunc stencilFunc(uint8_t f) {
	static const SceGxmStencilFunc map[8] = {
		SCE_GXM_STENCIL_FUNC_NEVER, SCE_GXM_STENCIL_FUNC_ALWAYS, SCE_GXM_STENCIL_FUNC_EQUAL, SCE_GXM_STENCIL_FUNC_NOT_EQUAL,
		SCE_GXM_STENCIL_FUNC_LESS, SCE_GXM_STENCIL_FUNC_LESS_EQUAL, SCE_GXM_STENCIL_FUNC_GREATER, SCE_GXM_STENCIL_FUNC_GREATER_EQUAL,
	};
	return map[f & 7];
}

static SceGxmStencilOp stencilOp(uint32_t op) {
	switch (op & 7) {
		case 1: return SCE_GXM_STENCIL_OP_ZERO;
		case 2: return SCE_GXM_STENCIL_OP_REPLACE;
		case 3: return SCE_GXM_STENCIL_OP_INVERT;
		case 4: return SCE_GXM_STENCIL_OP_INCR;
		case 5: return SCE_GXM_STENCIL_OP_DECR;
		default: return SCE_GXM_STENCIL_OP_KEEP;
	}
}

static void stencilOff(SceGxmContext *ctx) {
	sceGxmSetFrontStencilFunc(ctx, SCE_GXM_STENCIL_FUNC_ALWAYS, SCE_GXM_STENCIL_OP_KEEP, SCE_GXM_STENCIL_OP_KEEP,
		SCE_GXM_STENCIL_OP_KEEP, 0xFF, 0xFF);
}

/* ---- Framebuffer renderings ---- */

/*
	Every PSP framebuffer drawn into has its own rendering (at twice its
	size): the shown one, and those games draw to use as textures
	(reflections, blur, the menus' backgrounds). They share one depth and
	stencil buffer (in CDRAM, the GPU's fast memory, when it fits): each
	scene starts from it cleared.
*/
#define MAX_TARGET_BYTES (10 * 1024 * 1024)
#define SHARED_DEPTH_W 960
#define SHARED_DEPTH_H 544

typedef struct {
	uint32_t fb;
	uint16_t w, h;
	vita2d_texture *tex;
	int fresh;      // never drawn: cleared first
	int own_depth;  // its depth buffer is vita2d's (bigger than the shared one)
	uint32_t used;  // frame it was last drawn or shown
	uint32_t bytes;
	vita2d_texture view;   // the same, sampled without alpha (shown opaque)
} Target;

static Target targets[GE_SGX_MAX_FBS];
static uint32_t target_bytes = 0;
static uint32_t target_clock = 0;
static void *shared_depth = NULL;
static int shared_depth_tried = 0;
static int target_failures = 0;
static uint32_t shown_fb = 0;

static void *sharedDepth(void) {
	if (shared_depth_tried)
		return shared_depth;
	shared_depth_tried = 1;
	uint32_t size = (4 * SHARED_DEPTH_W * SHARED_DEPTH_H + 256 * 1024 - 1) & ~(256 * 1024 - 1);
	int cdram = 1;
	shared_depth = gpuAlloc(SCE_KERNEL_MEMBLOCK_TYPE_USER_CDRAM_RW, size, SCE_GXM_MEMORY_ATTRIB_READ | SCE_GXM_MEMORY_ATTRIB_WRITE);
	if (!shared_depth) {
		cdram = 0;
		shared_depth = gpuAlloc(SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE, size, SCE_GXM_MEMORY_ATTRIB_READ | SCE_GXM_MEMORY_ATTRIB_WRITE);
	}
	GeSgxLog("GeSgx render: shared depth %s (%s)\n", shared_depth ? "ok" : "FAILED", cdram ? "CDRAM" : "RAM");
	return shared_depth;
}

static Target *findTarget(uint32_t fb) {
	int i;
	for (i = 0; i < GE_SGX_MAX_FBS; i++) {
		if (targets[i].tex && targets[i].fb == fb)
			return &targets[i];
	}
	return NULL;
}

static void freeTarget(Target *t) {
	if (!t->tex)
		return;
	// The GPU may still read or write it
	sceGxmFinish(_vita2d_context);
	if (t->tex->gxm_rtgt)
		sceGxmDestroyRenderTarget(t->tex->gxm_rtgt);
	if (t->own_depth && t->tex->depth_UID >= 0) {
		void *base = NULL;
		if (sceKernelGetMemBlockBase(t->tex->depth_UID, &base) >= 0 && base)
			sceGxmUnmapMemory(base);
		sceKernelFreeMemBlock(t->tex->depth_UID);
	}
	vita2d_free_texture(t->tex);
	target_bytes -= t->bytes;
	memset(t, 0, sizeof(*t));
}

static Target *createTarget(Target *t, uint32_t fb, int w, int h) {
	uint32_t bytes = (uint32_t)w * h * 4;
	if (w <= SHARED_DEPTH_W && h <= SHARED_DEPTH_H)
		sharedDepth(); // before the colour: the depth buffer is the busier one, CDRAM goes to it first
	SceKernelMemBlockType type = vita2d_texture_get_alloc_memblock_type();

	// CDRAM first (the shown rendering, usually the first one made), main RAM if it is not there
	vita2d_texture *tex = NULL;
	int in_cdram = 1;
	vita2d_texture_set_alloc_memblock_type(SCE_KERNEL_MEMBLOCK_TYPE_USER_CDRAM_RW);
	tex = vita2d_create_empty_texture_format(w, h, SCE_GXM_TEXTURE_FORMAT_A8B8G8R8);
	if (!tex) {
		in_cdram = 0;
		vita2d_texture_set_alloc_memblock_type(SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE);
		tex = vita2d_create_empty_texture_format(w, h, SCE_GXM_TEXTURE_FORMAT_A8B8G8R8);
	}
	vita2d_texture_set_alloc_memblock_type(type);
	if (!tex || !tex->gxm_rtgt || tex->depth_UID < 0) {
		if (tex) {
			if (tex->gxm_rtgt)
				sceGxmDestroyRenderTarget(tex->gxm_rtgt);
			if (tex->depth_UID >= 0)
				sceKernelFreeMemBlock(tex->depth_UID);
			vita2d_free_texture(tex);
		}
		if (target_failures++ < 8)
			GeSgxLog("GeSgx render: target %X %dx%d FAILED (targets %u KB)\n", (unsigned)fb, w, h, (unsigned)(target_bytes / 1024));
		return NULL;
	}

	// vita2d's render target is for one scene a frame; a framebuffer can be drawn in several
	// (the game draws another one in between, to use it as a texture)
	SceGxmRenderTargetParams rp;
	memset(&rp, 0, sizeof(rp));
	rp.width = w;
	rp.height = h;
	rp.scenesPerFrame = 4;
	rp.multisampleMode = SCE_GXM_MULTISAMPLE_NONE;
	rp.driverMemBlock = -1;
	SceGxmRenderTarget *rt = NULL;
	if (sceGxmCreateRenderTarget(&rp, &rt) >= 0 && rt) {
		sceGxmDestroyRenderTarget(tex->gxm_rtgt);
		tex->gxm_rtgt = rt;
	}

	// vita2d made a depth buffer for it: the shared one instead, when it is big enough
	int own_depth = 1;
	void *depth = (w <= SHARED_DEPTH_W && h <= SHARED_DEPTH_H) ? sharedDepth() : NULL;
	if (depth) {
		void *old = NULL;
		sceKernelGetMemBlockBase(tex->depth_UID, &old);
		sceGxmDepthStencilSurfaceInit(&tex->gxm_sfd, SCE_GXM_DEPTH_STENCIL_FORMAT_S8D24,
			SCE_GXM_DEPTH_STENCIL_SURFACE_TILED, (w + 31) & ~31, depth, NULL);
		if (old)
			sceGxmUnmapMemory(old);
		sceKernelFreeMemBlock(tex->depth_UID);
		tex->depth_UID = -1;
		own_depth = 0;
	} else {
		bytes += (uint32_t)((w + 31) & ~31) * ((h + 31) & ~31) * 4;
	}

	t->fb = fb;
	t->w = (uint16_t)w;
	t->h = (uint16_t)h;
	t->tex = tex;
	t->fresh = 1;
	t->own_depth = own_depth;
	t->used = target_clock;
	t->bytes = bytes;
	t->view = *tex;
	sceGxmTextureSetFormat(&t->view.gxm_tex, SCE_GXM_TEXTURE_FORMAT_X8U8U8U8_1BGR);
	target_bytes += bytes;
	GeSgxLog("GeSgx render: target %X %dx%d in %s, depth %s | targets %u KB, free CDRAM %u KB, RAM %u KB\n",
		(unsigned)fb, w, h, in_cdram ? "CDRAM" : "RAM", own_depth ? "own" : "shared", (unsigned)(target_bytes / 1024),
		(unsigned)GeSgxFreeKB(SCE_KERNEL_MEMBLOCK_TYPE_USER_CDRAM_RW), (unsigned)GeSgxFreeKB(SCE_KERNEL_MEMBLOCK_TYPE_USER_RW));
	return t;
}

// The rendering of framebuffer fb, w x h, made (or remade, other size) if needed
static Target *getTarget(uint32_t fb, int w, int h) {
	Target *t = findTarget(fb);
	if (t && t->w == w && t->h == h) {
		t->used = target_clock;
		return t;
	}
	if (t)
		freeTarget(t);

	uint32_t need = (uint32_t)w * h * 4;
	for (;;) {
		// A free slot within the budget, or the least recently used one (not of this frame) given up
		int i, slot = -1, oldest = -1;
		for (i = 0; i < GE_SGX_MAX_FBS; i++) {
			if (!targets[i].tex) {
				if (slot < 0)
					slot = i;
			} else if (targets[i].used != target_clock && (oldest < 0 || targets[i].used < targets[oldest].used)) {
				oldest = i;
			}
		}
		if (slot >= 0 && target_bytes + need <= MAX_TARGET_BYTES) {
			t = createTarget(&targets[slot], fb, w, h);
			if (t || oldest < 0)
				return t;
		} else if (oldest < 0) {
			return NULL;
		}
		freeTarget(&targets[oldest]);
	}
}

/* ---- Exact scissor, culling ---- */

/*
	GXM's region clip works on whole tiles; the exact scissor rectangle is
	the depth buffer's mask bit (vitaGL does the same): a draw of the mask
	update program sets the bit where the stencil test passes (ALWAYS) and
	clears it where it fails (NEVER), and fragments are only written where
	it is set. Each scene starts with it set everywhere.
*/
static SceGxmFragmentProgram *mask_program = NULL;
static int mask_failed = 0;

// GE_SGX_CULL_CW draws with SCE_GXM_CULL_CW (1) or SCE_GXM_CULL_CCW (0); -1 not known yet, -2 no culling
static int cull_cw_is_gxm_cw = -1;

static int maskProgram(void) {
	if (mask_program)
		return 1;
	if (mask_failed)
		return 0;
	int res = sceGxmShaderPatcherCreateMaskUpdateFragmentProgram(patcher, &mask_program);
	if (res < 0) {
		mask_program = NULL;
		mask_failed = 1;
		GeSgxLog("GeSgx render: mask update program %08X: scissor on tiles only\n", (unsigned)res);
		return 0;
	}
	return 1;
}

// Two triangles over x0..x1, y0..y1 (target pixels), from the frame's vertex pool
static int quad(SceGxmContext *ctx, int x0, int y0, int x1, int y1, uint32_t color) {
	GeSgxVertex *v = vita2d_pool_memalign(4 * sizeof(GeSgxVertex), 16);
	uint16_t *idx = vita2d_pool_memalign(6 * sizeof(uint16_t), 16);
	if (!v || !idx)
		return 0;
	int i;
	for (i = 0; i < 4; i++) {
		v[i].x = (float)((i & 1) ? x1 : x0);
		v[i].y = (float)((i & 2) ? y1 : y0);
		v[i].z = 0.5f;
		v[i].w = 1.0f;
		v[i].u = v[i].v = 0.0f;
		v[i].color = color;
		v[i].spec_fog = GE_SGX_NO_FOG;
	}
	idx[0] = 0; idx[1] = 1; idx[2] = 2;
	idx[3] = 1; idx[4] = 3; idx[5] = 2;
	sceGxmSetVertexStream(ctx, 0, v);
	sceGxmDraw(ctx, SCE_GXM_PRIMITIVE_TRIANGLES, SCE_GXM_INDEX_FORMAT_U16, idx, 6);
	return 1;
}

// The mask bit set inside x0..x1, y0..y1 only (or everywhere: full) of a target of tw x th
// vertex program pixels; the region clip is left on the whole target (cw x ch pixels)
static void scissorMask(SceGxmContext *ctx, int cw, int ch, int tw, int th, int x0, int y0, int x1, int y1, int full) {
	sceGxmSetVertexProgram(ctx, vertex_program);
	sceGxmSetFragmentProgram(ctx, mask_program);
	sceGxmSetFrontDepthFunc(ctx, SCE_GXM_DEPTH_FUNC_ALWAYS);
	sceGxmSetFrontDepthWriteEnable(ctx, SCE_GXM_DEPTH_WRITE_DISABLED);
	sceGxmSetCullMode(ctx, SCE_GXM_CULL_NONE);
	sceGxmSetRegionClip(ctx, SCE_GXM_REGION_CLIP_OUTSIDE, 0, 0, cw - 1, ch - 1);
	if (!full) {
		sceGxmSetFrontStencilFunc(ctx, SCE_GXM_STENCIL_FUNC_NEVER, SCE_GXM_STENCIL_OP_KEEP, SCE_GXM_STENCIL_OP_KEEP,
			SCE_GXM_STENCIL_OP_KEEP, 0, 0);
		quad(ctx, 0, 0, tw, th, 0);
	}
	sceGxmSetFrontStencilFunc(ctx, SCE_GXM_STENCIL_FUNC_ALWAYS, SCE_GXM_STENCIL_OP_KEEP, SCE_GXM_STENCIL_OP_KEEP,
		SCE_GXM_STENCIL_OP_KEEP, 0, 0);
	if (full)
		quad(ctx, 0, 0, tw, th, 0);
	else
		quad(ctx, x0, y0, x1, y1, 0);
}

static int whiteAt(const vita2d_texture *t, int x, int y) {
	const uint32_t *p = sceGxmTextureGetData(&t->gxm_tex);
	uint32_t stride = (sceGxmTextureGetWidth(&t->gxm_tex) + 7) & ~7u;
	return (p[y * stride + x] & 0x00FFFFFFu) == 0x00FFFFFFu;
}

/*
	Which way GXM counts the winding, and whether the mask works: drawn
	once into a small rendering of its own and read back. Three triangles,
	all clockwise on screen: culled clockwise, not culled, culled counter-
	clockwise; then a white quad over everything with the mask bit set in
	the top left quarter only.
*/
static void selfTest(SceGxmContext *ctx, int *scenes) {
	GeSgxDraw cd;
	memset(&cd, 0, sizeof(cd));
	cd.flags = GE_SGX_DRAW_CLEAR | GE_SGX_DRAW_WRITE_COLOR;
	GeSgxBlend bl;
	FragmentProgram *fp = fragmentProgram(&cd, vertex_gxp, &bl);
	if (!fp)
		return; // its program is not there yet: the next frame
	maskProgram();

	SceKernelMemBlockType type = vita2d_texture_get_alloc_memblock_type();
	vita2d_texture_set_alloc_memblock_type(SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE);
	vita2d_texture *t = vita2d_create_empty_texture_format(64, 64, SCE_GXM_TEXTURE_FORMAT_A8B8G8R8);
	vita2d_texture_set_alloc_memblock_type(type);
	if (!t || !t->gxm_rtgt) {
		GeSgxLog("GeSgx render: self test target FAILED: no culling\n");
		cull_cw_is_gxm_cw = -2;
		return;
	}

	int k, r[3], m_in = 0, m_out = 0;
	static const SceGxmCullMode modes[3] = { SCE_GXM_CULL_CW, SCE_GXM_CULL_NONE, SCE_GXM_CULL_CCW };
	vita2d_start_drawing_advanced(t, *scenes == 0 ? VITA_2D_RESET_POOL : 0);
	(*scenes)++;
	vita2d_set_clear_color(RGBA8(0, 0, 0, 255));
	vita2d_clear_screen();
	sceGxmSetVertexProgram(ctx, vertex_program);
	sceGxmSetFragmentProgram(ctx, fp->program);
	sceGxmSetFrontDepthFunc(ctx, SCE_GXM_DEPTH_FUNC_ALWAYS);
	sceGxmSetFrontDepthWriteEnable(ctx, SCE_GXM_DEPTH_WRITE_DISABLED);
	for (k = 0; k < 3; k++) {
		// 960x544 pixel space (the vertex program's), a third of the width each
		GeSgxVertex *v = vita2d_pool_memalign(3 * sizeof(GeSgxVertex), 16);
		uint16_t *idx = vita2d_pool_memalign(3 * sizeof(uint16_t), 16);
		if (!v || !idx)
			break;
		float x0 = k * 320.0f;
		float px[3] = { x0 + 20.0f, x0 + 300.0f, x0 + 160.0f }, py[3] = { 100.0f, 100.0f, 450.0f };
		int i;
		for (i = 0; i < 3; i++) {
			memset(&v[i], 0, sizeof(v[i]));
			v[i].x = px[i];
			v[i].y = py[i];
			v[i].z = 0.5f;
			v[i].w = 1.0f;
			v[i].color = 0xFFFFFFFFu;
			v[i].spec_fog = GE_SGX_NO_FOG;
			idx[i] = (uint16_t)i;
		}
		sceGxmSetCullMode(ctx, modes[k]);
		sceGxmSetVertexStream(ctx, 0, v);
		sceGxmDraw(ctx, SCE_GXM_PRIMITIVE_TRIANGLES, SCE_GXM_INDEX_FORMAT_U16, idx, 3);
	}
	sceGxmSetCullMode(ctx, SCE_GXM_CULL_NONE);
	vita2d_end_drawing();
	sceGxmFinish(ctx);
	for (k = 0; k < 3; k++)
		r[k] = whiteAt(t, (k * 320 + 160) / 15, 217 * 64 / 544);

	if (mask_program) {
		vita2d_start_drawing_advanced(t, 0);
		(*scenes)++;
		vita2d_set_clear_color(RGBA8(0, 0, 0, 255));
		vita2d_clear_screen();
		// The mask in 960x544 space (this target is 64x64: the default viewport maps it)
		scissorMask(ctx, 64, 64, 960, 544, 0, 0, 480, 272, 0);
		sceGxmSetFragmentProgram(ctx, fp->program);
		sceGxmSetFrontStencilFunc(ctx, SCE_GXM_STENCIL_FUNC_ALWAYS, SCE_GXM_STENCIL_OP_KEEP, SCE_GXM_STENCIL_OP_KEEP,
			SCE_GXM_STENCIL_OP_KEEP, 0xFF, 0xFF);
		quad(ctx, 0, 0, 960, 544, 0xFFFFFFFFu);
		vita2d_end_drawing();
		sceGxmFinish(ctx);
		m_in = whiteAt(t, 8, 8);
		m_out = whiteAt(t, 48, 48) || whiteAt(t, 48, 8) || whiteAt(t, 8, 48);
		if (!m_in || m_out) {
			GeSgxLog("GeSgx render: mask test FAILED (inside %d, outside %d): scissor on tiles only\n", m_in, m_out);
			mask_failed = 1;
		}
	}

	if (r[1] && r[0] != r[2]) {
		// Culled clockwise did not draw it: GXM's clockwise is the screen's
		cull_cw_is_gxm_cw = r[0] ? 0 : 1;
	} else {
		cull_cw_is_gxm_cw = -2;
	}
	GeSgxLog("GeSgx render: self test: cull cw %d none %d ccw %d -> %s | mask inside %d outside %d -> %s\n",
		r[0], r[1], r[2],
		cull_cw_is_gxm_cw == -2 ? "NO CULLING" : cull_cw_is_gxm_cw ? "GXM CW = screen CW" : "GXM CW = screen CCW",
		m_in, m_out, (mask_program && !mask_failed) ? "exact scissor" : "tile scissor");

	if (t->gxm_rtgt)
		sceGxmDestroyRenderTarget(t->gxm_rtgt);
	if (t->depth_UID >= 0) {
		void *base = NULL;
		if (sceKernelGetMemBlockBase(t->depth_UID, &base) >= 0 && base)
			sceGxmUnmapMemory(base);
		sceKernelFreeMemBlock(t->depth_UID);
	}
	vita2d_free_texture(t);
}

static SceGxmCullMode cullMode(uint8_t cull) {
	if (cull == GE_SGX_CULL_NONE || cull_cw_is_gxm_cw < 0)
		return SCE_GXM_CULL_NONE;
	int cw = cull == GE_SGX_CULL_CW;
	return (cw == cull_cw_is_gxm_cw) ? SCE_GXM_CULL_CW : SCE_GXM_CULL_CCW;
}

/*
	Draws [first, last) of the frame (one framebuffer) into the scene being
	drawn, a target of w x h pixels.
*/
static void renderDraws(SceGxmContext *ctx, const GeSgxFrameData *frame, uint32_t first, uint32_t last,
	int tw, int th, uint32_t target_fb, GeSgxRenderStats *stats) {
	// The vertex programs map 960x544 pixels to clip space: other sizes get a viewport of those pixels
	if (tw != 960 || th != 544)
		sceGxmSetViewport(ctx, 480.0f, 480.0f, 272.0f, -272.0f, 0.5f, 0.5f);

	sceGxmSetFrontDepthFunc(ctx, SCE_GXM_DEPTH_FUNC_ALWAYS);
	sceGxmSetFrontDepthWriteEnable(ctx, SCE_GXM_DEPTH_WRITE_DISABLED);
	int last_zfunc = SCE_GXM_DEPTH_FUNC_ALWAYS, last_zwrite = 0;
	sceGxmSetCullMode(ctx, SCE_GXM_CULL_NONE);
	SceGxmCullMode last_cull = SCE_GXM_CULL_NONE;
	sceGxmSetFrontPointLineWidth(ctx, GE_SGX_SCALE);
	sceGxmSetVertexProgram(ctx, vertex_program);
	const SceGxmVertexProgram *last_vp = vertex_program;
	stencilOff(ctx);
	uint32_t last_stencil = 0xFFFFFFFFu;
	int exact = mask_program && !mask_failed;
	int mask_full = 1;

	int16_t last_scissor[4] = { -1, -1, -1, -1 };
	int scissor_empty = 0;
	const SceGxmFragmentProgram *last_fp = NULL;
	uint32_t i;

	for (i = first; i < last; i++) {
		const GeSgxDraw *d = &frame->draws[i];

		// A clear writing nothing
		if ((d->flags & GE_SGX_DRAW_CLEAR) && !(d->flags & (GE_SGX_DRAW_WRITE_COLOR | GE_SGX_DRAW_WRITE_DEPTH | GE_SGX_DRAW_WRITE_ALPHA)))
			continue;

		// The texture first: a framebuffer texture needs that framebuffer's rendering (not this one)
		SceGxmTexture tex;
		if (d->flags & GE_SGX_DRAW_TEXTURED) {
			const SceGxmTexture *t = NULL;
			if (d->fbtex) {
				uint32_t sfb = d->fbtex - 1;
				Target *src = sfb == target_fb ? NULL : findTarget(sfb);
				if (src) {
					t = (d->tex_mode & GE_SGX_TEX_OPAQUE) ? &src->view.gxm_tex : &src->tex->gxm_tex;
					stats->fb_textured++;
				}
			} else {
				t = geSgxTexGet(d->texture);
			}
			if (!t) {
				stats->no_texture++;
				continue;
			}
			tex = *t;
			sceGxmTextureSetUAddrMode(&tex, (d->tex_mode & GE_SGX_TEX_CLAMP_U) ? SCE_GXM_TEXTURE_ADDR_CLAMP : SCE_GXM_TEXTURE_ADDR_REPEAT);
			sceGxmTextureSetVAddrMode(&tex, (d->tex_mode & GE_SGX_TEX_CLAMP_V) ? SCE_GXM_TEXTURE_ADDR_CLAMP : SCE_GXM_TEXTURE_ADDR_REPEAT);
			sceGxmTextureSetMinFilter(&tex, (d->tex_mode & GE_SGX_TEX_LINEAR_MIN) ? SCE_GXM_TEXTURE_FILTER_LINEAR : SCE_GXM_TEXTURE_FILTER_POINT);
			sceGxmTextureSetMagFilter(&tex, (d->tex_mode & GE_SGX_TEX_LINEAR_MAG) ? SCE_GXM_TEXTURE_FILTER_LINEAR : SCE_GXM_TEXTURE_FILTER_POINT);
			sceGxmTextureSetMipFilter(&tex, (d->tex_mode & GE_SGX_TEX_MIP_LINEAR) ? SCE_GXM_TEXTURE_MIP_FILTER_ENABLED : SCE_GXM_TEXTURE_MIP_FILTER_DISABLED);
		}

		// The vertex program: the CPU transformed vertices', or the GPU transform's for this vertex shape
		const SceGxmVertexProgram *vp = vertex_program;
		const SceGxmProgram *vgxp = vertex_gxp;
		if (d->gpu) {
			int waiting = 0;
			GpuProgram *g = gpuProgram(d, &waiting);
			if (!g) {
				if (waiting)
					stats->waiting++;
				else
					stats->no_program++;
				continue;
			}
			vp = g->program;
			vgxp = g->gxp;
		}

		GeSgxBlend bl;
		FragmentProgram *fp = fragmentProgram(d, vgxp, &bl);
		if (!fp) {
			stats->waiting++;
			continue;
		}

		// Scissor: tiles by the region clip, the exact rectangle by the mask
		if (memcmp(last_scissor, d->scissor, sizeof(last_scissor)) != 0) {
			memcpy(last_scissor, d->scissor, sizeof(last_scissor));
			int x0 = d->scissor[0], y0 = d->scissor[1], x1 = d->scissor[2], y1 = d->scissor[3];
			if (x0 < 0) x0 = 0;
			if (y0 < 0) y0 = 0;
			if (x1 > tw) x1 = tw;
			if (y1 > th) y1 = th;
			scissor_empty = x1 <= x0 || y1 <= y0;
			if (!scissor_empty) {
				int aligned = !(x0 & 31) && !(y0 & 31) && (!(x1 & 31) || x1 == tw) && (!(y1 & 31) || y1 == th);
				if (exact && (!aligned || !mask_full)) {
					scissorMask(ctx, tw, th, tw, th, x0, y0, x1, y1, aligned);
					mask_full = aligned;
					stats->scissor_masks++;
					// The mask draws changed these
					last_vp = NULL;
					last_fp = NULL;
					last_zfunc = -1;
					last_zwrite = -1;
					last_stencil = 0xFFFFFFFFu;
					last_cull = SCE_GXM_CULL_NONE;
				}
				sceGxmSetRegionClip(ctx, SCE_GXM_REGION_CLIP_OUTSIDE, x0, y0, x1 - 1, y1 - 1);
			}
		}
		if (scissor_empty) {
			stats->scissored++;
			continue;
		}

		if (vp != last_vp) {
			sceGxmSetVertexProgram(ctx, vp);
			last_vp = vp;
		}

		if (fp->program != last_fp) {
			sceGxmSetFragmentProgram(ctx, fp->program);
			last_fp = fp->program;
		}

		int zfunc = (d->flags & GE_SGX_DRAW_CLEAR) ? SCE_GXM_DEPTH_FUNC_ALWAYS : depthFunc(d->ztest);
		int zwrite = (d->flags & GE_SGX_DRAW_WRITE_DEPTH) != 0;
		if (zfunc != last_zfunc) {
			sceGxmSetFrontDepthFunc(ctx, (SceGxmDepthFunc)zfunc);
			last_zfunc = zfunc;
		}
		if (zwrite != last_zwrite) {
			sceGxmSetFrontDepthWriteEnable(ctx, zwrite ? SCE_GXM_DEPTH_WRITE_ENABLED : SCE_GXM_DEPTH_WRITE_DISABLED);
			last_zwrite = zwrite;
		}

		SceGxmCullMode cull = cullMode(d->cull);
		if (cull != last_cull) {
			sceGxmSetCullMode(ctx, cull);
			last_cull = cull;
		}
		if (cull != SCE_GXM_CULL_NONE)
			stats->culled_draws++;

		// Stencil (the PSP's is the framebuffer alpha; here the depth buffer's 8 bits)
		uint32_t skey = d->sfunc == 0xFF ? 0xFFFFFFFEu :
			((uint32_t)d->sfunc | ((uint32_t)d->sops << 4) | ((uint32_t)d->smask << 16) | ((uint32_t)d->swmask << 24)) ^ ((uint32_t)d->sref << 8);
		if (skey != last_stencil) {
			if (d->sfunc == 0xFF) {
				stencilOff(ctx);
			} else {
				sceGxmSetFrontStencilFunc(ctx, stencilFunc(d->sfunc), stencilOp(d->sops), stencilOp(d->sops >> 4),
					stencilOp(d->sops >> 8), d->smask, d->swmask);
				sceGxmSetFrontStencilRef(ctx, d->sref);
			}
			last_stencil = skey;
		}
		if (d->sfunc != 0xFF)
			stats->stencil_draws++;
		if ((d->flags & GE_SGX_DRAW_WRITE_ALPHA) && !(d->flags & GE_SGX_DRAW_CLEAR))
			stats->alpha_draws++;

		if (fp->alpha_ref || fp->env || fp->fog_color || fp->fix_c || fp->fix_a) {
			void *uniforms = NULL;
			sceGxmReserveFragmentDefaultUniformBuffer(ctx, &uniforms);
			if (fp->alpha_ref) {
				float ref = d->atest_ref / 255.0f;
				sceGxmSetUniformDataF(uniforms, fp->alpha_ref, 0, 1, &ref);
			}
			if (fp->env)
				setColorUniform(uniforms, fp->env, d->env);
			if (fp->fog_color)
				setColorUniform(uniforms, fp->fog_color, d->fog_color);
			if (fp->fix_c || fp->fix_a) {
				// The program was chosen for this blending: bl is filled in (blendKey)
				float c[4] = { bl.rgb[0], bl.rgb[1], bl.rgb[2], 1.0f }, a[4] = { bl.as, bl.ab, bl.am, bl.aa };
				if (!bl.fix) {
					c[0] = c[1] = c[2] = 1.0f;
					a[0] = 0.0f; a[1] = 1.0f; a[2] = 1.0f; a[3] = 0.0f;
				}
				if (fp->fix_c)
					sceGxmSetUniformDataF(uniforms, fp->fix_c, 0, 4, c);
				if (fp->fix_a)
					sceGxmSetUniformDataF(uniforms, fp->fix_a, 0, 4, a);
				stats->fix_draws++;
			}
		}

		if (d->flags & GE_SGX_DRAW_TEXTURED)
			sceGxmSetFragmentTexture(ctx, 0, &tex);

		if (d->gpu) {
			sceGxmSetVertexUniformBuffer(ctx, 0, frame->uniforms_gpu + d->ucommon);
			sceGxmSetVertexUniformBuffer(ctx, 1, frame->uniforms_gpu + d->uobject);
			sceGxmSetVertexStream(ctx, 0, (d->direct ? frame->direct : frame->raw) + d->base);
			stats->gpu_draws++;
		} else {
			sceGxmSetVertexStream(ctx, 0, &frame->vertices_gpu[d->base]);
		}
		sceGxmDraw(ctx, primitive(d->prim), SCE_GXM_INDEX_FORMAT_U16, &frame->indices_gpu[d->first], d->count);

		stats->draws++;
		stats->vertices += d->count;
	}

	// Back to the state vita2d expects
	sceGxmSetFrontDepthFunc(ctx, SCE_GXM_DEPTH_FUNC_ALWAYS);
	sceGxmSetFrontDepthWriteEnable(ctx, SCE_GXM_DEPTH_WRITE_DISABLED);
	sceGxmSetCullMode(ctx, SCE_GXM_CULL_NONE);
	stencilOff(ctx);
	sceGxmSetRegionClip(ctx, SCE_GXM_REGION_CLIP_OUTSIDE, 0, 0, tw - 1, th - 1);
	sceGxmSetFrontPointLineWidth(ctx, 1);
}

vita2d_texture *geSgxRenderFrame(SceGxmContext *ctx, const GeSgxFrameData *frame, uint32_t fb,
	const GeSgxAlias *aliases, int naliases, GeSgxRenderStats *stats, int *scenes) {
	SceUInt64 t0 = sceKernelGetProcessTimeWide();
	memset(stats, 0, sizeof(*stats));
	*scenes = 0;

	if (!patcher || !vertexProgram())
		return NULL;
	target_clock++;

	if (cull_cw_is_gxm_cw == -1)
		selfTest(ctx, scenes);

	// One scene per run of draws into one framebuffer, each into that framebuffer's rendering
	uint32_t i = 0, drawn_fbs[GE_SGX_MAX_FBS * 2], ndrawn = 0;
	while (i < frame->ndraws) {
		uint32_t dfb = frame->draws[i].fb, j = i + 1;
		while (j < frame->ndraws && frame->draws[j].fb == dfb)
			j++;

		uint32_t w = 480, h = 272, k;
		for (k = 0; k < frame->nfbs; k++) {
			if (frame->fbs[k].fb == dfb && frame->fbs[k].w && frame->fbs[k].h) {
				w = frame->fbs[k].w;
				h = frame->fbs[k].h;
			}
		}
		// Drawn before in this frame: its depth buffer did not survive the scenes in between
		for (k = 0; k < ndrawn; k++) {
			if (drawn_fbs[k] == dfb) {
				stats->reentries++;
				break;
			}
		}
		if (k == ndrawn && ndrawn < GE_SGX_MAX_FBS * 2)
			drawn_fbs[ndrawn++] = dfb;

		Target *t = getTarget(dfb, w * GE_SGX_SCALE, h * GE_SGX_SCALE);
		if (!t) {
			stats->no_target += j - i;
			i = j;
			continue;
		}
		vita2d_start_drawing_advanced(t->tex, *scenes == 0 ? VITA_2D_RESET_POOL : 0);
		if (t->fresh) {
			vita2d_set_clear_color(RGBA8(0, 0, 0, 255));
			vita2d_clear_screen();
			t->fresh = 0;
		}
		renderDraws(ctx, frame, i, j, t->w, t->h, dfb, stats);
		vita2d_end_drawing();
		(*scenes)++;
		if (dfb != fb)
			stats->other_fb += j - i;
		i = j;
	}

	stats->render_us = (uint32_t)(sceKernelGetProcessTimeWide() - t0);
	stats->scenes = (uint32_t)*scenes;

	// What is shown: the flipped framebuffer's rendering, or the one copied into it
	Target *show = findTarget(fb);
	int a;
	for (a = 0; !show && a < naliases; a++) {
		if (aliases[a].dst == fb)
			show = findTarget(aliases[a].src);
	}
	int n = 0;
	for (a = 0; a < GE_SGX_MAX_FBS; a++)
		n += targets[a].tex != NULL;
	stats->targets = (uint32_t)n;
	stats->target_kb = target_bytes / 1024;
	if (show) {
		show->used = target_clock;
		shown_fb = show->fb;
		return &show->view;
	}
	return NULL;
}

vita2d_texture *geSgxRenderShown(void) {
	Target *t = shown_fb ? findTarget(shown_fb) : NULL;
	return t ? &t->view : NULL;
}
