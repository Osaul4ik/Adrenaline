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
#include <psp2/io/fcntl.h>
#include <psp2/io/stat.h>

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vita2d.h>

extern SceGxmContext *_vita2d_context;

#include "ge_sgx.h"
#include "ge_sgx_pool.h"
#include "ge_sgx_render.h"
#include "ge_sgx_shader.h"
#include "ge_sgx_tex.h"

#define PATCHER_BUFFER_SIZE (64 * 1024)
#define PATCHER_VERTEX_USSE_SIZE (64 * 1024)
#define PATCHER_FRAGMENT_USSE_SIZE (256 * 1024)

#define MAX_FRAGMENT_PROGRAMS 512

typedef struct {
	uint32_t key;
	uint32_t blend;
	const SceGxmProgram *vgxp; // the vertex program it was made for
	SceGxmFragmentProgram *program;
	const SceGxmProgramParameter *alpha_ref;
	const SceGxmProgramParameter *env;
	const SceGxmProgramParameter *fog_color;
	const SceGxmProgramParameter *scissor; // exact scissor (x0, y0, x1, y1 in pixels)
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
	attrs[3].format = SCE_GXM_ATTRIBUTE_FORMAT_F32;
	attrs[3].componentCount = 1;
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

static int factor(uint32_t f, uint32_t fix, int src) {
	switch (f) {
		case 0: return src ? SCE_GXM_BLEND_FACTOR_DST_COLOR : SCE_GXM_BLEND_FACTOR_SRC_COLOR;
		case 1: return src ? SCE_GXM_BLEND_FACTOR_ONE_MINUS_DST_COLOR : SCE_GXM_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
		case 2: case 6: return SCE_GXM_BLEND_FACTOR_SRC_ALPHA;
		case 3: case 7: return SCE_GXM_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		case 4: case 8: return SCE_GXM_BLEND_FACTOR_DST_ALPHA;
		case 5: case 9: return SCE_GXM_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
		default: {
			// Fixed colour: GXM has no constant factor; white and black are exact
			uint32_t sum = (fix & 0xFF) + ((fix >> 8) & 0xFF) + ((fix >> 16) & 0xFF);
			return sum >= 3 * 0x80 ? SCE_GXM_BLEND_FACTOR_ONE : SCE_GXM_BLEND_FACTOR_ZERO;
		}
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

// Blend state of a draw as a key: 0 for none
static uint32_t blendKey(const GeSgxDraw *d) {
	if (d->flags & GE_SGX_DRAW_CLEAR) {
		uint32_t mask = 0;
		if (d->flags & GE_SGX_DRAW_WRITE_COLOR)
			mask |= SCE_GXM_COLOR_MASK_R | SCE_GXM_COLOR_MASK_G | SCE_GXM_COLOR_MASK_B;
		// Alpha: the PSP's stencil, kept in the rendering's alpha for the blending with the destination alpha
		// (Persona 2's glows: INVDSTALPHA after a clear to alpha 0); 565 framebuffers keep alpha 1
		if ((d->flags & GE_SGX_DRAW_WRITE_ALPHA) && (d->flags & GE_SGX_DRAW_FB_ALPHA))
			mask |= SCE_GXM_COLOR_MASK_A;
		return 0x80000000u | mask;
	}
	uint32_t masked = (uint32_t)(d->cmask & 7) << 12; // channels not written
	if (!(d->flags & GE_SGX_DRAW_BLEND))
		return masked;
	uint32_t src = factor(d->blend & 0xF, d->fix_a, 1);
	uint32_t dst = factor((d->blend >> 4) & 0xF, d->fix_b, 0);
	uint32_t func = blendFunc((d->blend >> 8) & 7);
	return 0x40000000u | masked | func | (src << 4) | (dst << 8);
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

// extra: shader key bits from the render state (GE_SGX_KEY_SCISSOR)
#define GE_SGX_KEY_SCISSOR (1u << 12)
static FragmentProgram *fragmentProgram(const GeSgxDraw *d, const SceGxmProgram *vgxp, uint32_t extra) {
	uint32_t key = geSgxShaderKey(d) | extra;
	uint32_t blend = blendKey(d);
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
		Alpha (the PSP's stencil) is written by clears of framebuffers that
		have it (blendKey), as the PSP does with the stencil test off; draws
		never write it. The rendering is shown opaque whatever its alpha.
	*/
	SceGxmBlendInfo info;
	memset(&info, 0, sizeof(info));
	info.colorMask = SCE_GXM_COLOR_MASK_R | SCE_GXM_COLOR_MASK_G | SCE_GXM_COLOR_MASK_B;
	info.colorFunc = SCE_GXM_BLEND_FUNC_NONE;
	info.alphaFunc = SCE_GXM_BLEND_FUNC_NONE;
	info.colorSrc = info.alphaSrc = SCE_GXM_BLEND_FACTOR_ONE;
	info.colorDst = info.alphaDst = SCE_GXM_BLEND_FACTOR_ZERO;
	if (blend & 0x80000000u) {
		info.colorMask = blend & SCE_GXM_COLOR_MASK_ALL; // a clear: alpha too when it writes the stencil
	} else {
		if (blend & 0x40000000u) {
			info.colorFunc = blend & 0xF;
			info.colorSrc = (blend >> 4) & 0xF;
			info.colorDst = (blend >> 8) & 0xF;
		}
		if (blend & (1u << 12))
			info.colorMask &= ~SCE_GXM_COLOR_MASK_R;
		if (blend & (1u << 13))
			info.colorMask &= ~SCE_GXM_COLOR_MASK_G;
		if (blend & (1u << 14))
			info.colorMask &= ~SCE_GXM_COLOR_MASK_B;
	}
	/*
		Nothing written (all channels masked: the minimap's mask, written into
		the depth buffer only, its alpha test cutting the circle; a clear of
		the depth only, before it): a program
		writing every channel unchanged instead (blended zero and one), so the
		GPU runs it whole, its discard included. With no channel written its
		fragments were not discarded or not tested: the mask wrote no depth
		around the circle and the map showed outside it.
	*/
	if (info.colorMask == SCE_GXM_COLOR_MASK_NONE) {
		info.colorMask = SCE_GXM_COLOR_MASK_R | SCE_GXM_COLOR_MASK_G | SCE_GXM_COLOR_MASK_B;
		info.colorFunc = info.alphaFunc = SCE_GXM_BLEND_FUNC_ADD;
		info.colorSrc = info.alphaSrc = SCE_GXM_BLEND_FACTOR_ZERO;
		info.colorDst = info.alphaDst = SCE_GXM_BLEND_FACTOR_ONE;
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
	fp->scissor = sceGxmProgramFindParameterByName(gxp, "uScissor");
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
	(reflections, blur, the menus' backgrounds). Each has its own depth and
	stencil buffer (in the kernel's pool when it fits, else vita2d's): a
	frame that draws into another framebuffer in between (Need for Speed
	Carbon's reflection) goes on with the depth and stencil it had, they are
	stored at the end of its scene and loaded at the start of the next one.
	A frame's first scene of a target starts from them cleared.
*/
#define MAX_TARGET_BYTES (10 * 1024 * 1024)
#define TARGET_SCENES 8 // scenes a frame a target's render target is made for (SCE_GXM_MAX_SCENES_PER_RENDERTARGET)
// CDRAM always left to the emulator (with none left, Carbon's loading screen ended in an error)
#define RESERVE_CDRAM_KB 1024

typedef struct {
	uint32_t fb;
	uint16_t w, h;
	vita2d_texture *tex;
	int fresh;      // never drawn: cleared first
	int own_depth;  // its depth buffer is vita2d's (else pool_depth)
	void *pool_depth;
	uint32_t pool_depth_bytes;
	uint32_t scene_frame; // frame (rframes + 1) of its last scene
	uint32_t used;  // frame it was last drawn or shown
	uint32_t bytes;
} Target;

static Target targets[GE_SGX_MAX_FBS];
static uint32_t target_bytes = 0;
static uint32_t target_clock = 0;
static int target_failures = 0;
static uint32_t shown_fb = 0; // VRAM offset + 1 of the shown framebuffer (0: none; 0 is a framebuffer too)

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
	if (t->pool_depth)
		geSgxPoolFree(t->pool_depth, t->pool_depth_bytes);
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
	SceKernelMemBlockType type = vita2d_texture_get_alloc_memblock_type();

	// CDRAM if it fits with the emulator's reserve (the depth buffer usually took it), main RAM else
	vita2d_texture *tex = NULL;
	int in_cdram = 0;
	if (GeSgxFreeKB(SCE_KERNEL_MEMBLOCK_TYPE_USER_CDRAM_RW) >= bytes / 1024 + 256 + RESERVE_CDRAM_KB) {
		in_cdram = 1;
		vita2d_texture_set_alloc_memblock_type(SCE_KERNEL_MEMBLOCK_TYPE_USER_CDRAM_RW);
		tex = vita2d_create_empty_texture_format(w, h, SCE_GXM_TEXTURE_FORMAT_A8B8G8R8);
	}
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

	/*
		vita2d made its render target for one scene a frame; a frame draws into
		a framebuffer in several (Carbon: the main one before and after its
		reflection, the blur's one before and after the copy), which GXM's
		render target must be made for: one for up to 8
	*/
	{
		SceGxmRenderTargetParams params;
		memset(&params, 0, sizeof(params));
		params.width = w;
		params.height = h;
		params.scenesPerFrame = TARGET_SCENES;
		params.multisampleMode = SCE_GXM_MULTISAMPLE_NONE;
		params.driverMemBlock = -1;
		SceGxmRenderTarget *rtgt = NULL;
		int err = sceGxmCreateRenderTarget(&params, &rtgt);
		if (err >= 0 && rtgt) {
			sceGxmDestroyRenderTarget(tex->gxm_rtgt);
			tex->gxm_rtgt = rtgt;
		} else if (target_failures++ < 8) {
			GeSgxLog("GeSgx render: target %X: render target for %d scenes %08X (vita2d's kept)\n", (unsigned)fb, TARGET_SCENES, (unsigned)err);
		}
	}

	// vita2d made a depth buffer for it: one in the pool instead, when it fits
	int own_depth = 1;
	uint32_t depth_bytes = (uint32_t)((w + 31) & ~31) * ((h + 31) & ~31) * 4;
	void *depth = geSgxPoolAlloc(depth_bytes);
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
		bytes += depth_bytes;
	} else {
		bytes += (uint32_t)((w + 31) & ~31) * ((h + 31) & ~31) * 4;
	}

	t->fb = fb;
	t->w = (uint16_t)w;
	t->h = (uint16_t)h;
	t->tex = tex;
	t->fresh = 1;
	t->own_depth = own_depth;
	t->pool_depth = own_depth ? NULL : depth;
	t->pool_depth_bytes = own_depth ? 0 : depth_bytes;
	t->scene_frame = 0;
	t->used = target_clock;
	t->bytes = bytes;
	target_bytes += bytes;
	GeSgxLog("GeSgx render: target %X %dx%d in %s, depth %s | targets %u KB, free CDRAM %u KB, RAM %u KB\n",
		(unsigned)fb, w, h, in_cdram ? "CDRAM" : "RAM", own_depth ? "vita2d's" : "pool", (unsigned)(target_bytes / 1024),
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

/* ---- Diagnostics ---- */

int ge_sgx_debug_mode = 0;
volatile int ge_sgx_trace_render = 0, ge_sgx_trace_walker = 0;
const char *const ge_sgx_debug_names[GE_SGX_DEBUG_MODES] = { "normal", "no alpha test", "no fog", "no depth test",
	"no textures", "no stencil" };

/*
	Draws not drawn: each reason with its keys logged the first time, and
	every draw of a traced frame summed up in runs (consecutive draws with
	the same outcome, programs and texture), kept in memory and written at
	the end of the frame.
*/
#define RTRACE_BYTES (1024 * 1024)
#define RTRACE_LINES 9000
static char rtrace_text[RTRACE_BYTES];
static uint32_t rtrace_len = 0, rtrace_lines = 0;
static int rtrace = 0;

static const char *const outcome_names[7] = { "clear writing nothing", "fb texture without rendering", "texture not in cache",
	"no vertex program", "no fragment program", "empty scissor", "drawn" };
#define OUTCOME_DRAWN 6

static void rtraceAdd(const char *fmt, ...) {
	if (!rtrace || rtrace_lines >= RTRACE_LINES || rtrace_len >= RTRACE_BYTES - 1)
		return;
	va_list ap;
	va_start(ap, fmt);
	int n = vsnprintf(rtrace_text + rtrace_len, RTRACE_BYTES - rtrace_len, fmt, ap);
	va_end(ap);
	if (n > 0 && rtrace_len + (uint32_t)n < RTRACE_BYTES) {
		rtrace_len += (uint32_t)n;
		rtrace_lines++;
	} else {
		rtrace_text[rtrace_len] = '\0';
	}
}

// The run of draws being summed up
static struct {
	uint32_t first, last, outcome, fkey, blend, vkey, tex, n3d, verts;
	uint32_t flags, zs, stencil, scissor; // draw flags; ztest, colour mask, prim, texture mode; stencil; scissor (two corners, 8 px)
	float z0, z1; // depth range of the first draw's vertices (CPU vertices; -1: GPU)
	float x0, y0, x1, y1; // and their pixels (x/w, y/w)
	uint32_t aref, color, tfmt; // alpha test reference; first CPU vertex colour; texture format
	uint32_t tsize, topaque;    // texture: w | h << 16; texels with alpha > 0 (palette textures)
	int active;
} run;

static void runFlush(void) {
	if (!run.active)
		return;
	run.active = 0;
	rtraceAdd("GeSgx R:   draws %u-%u (%u, %u gpu, %u vertices): %s | fragment %04X blend %08X | vertex %s%04X | texture %s%X"
		" | flags %04X ztest %u cmask %u prim %u texmode %X | stencil %08X | scissor %u,%u-%u,%u | z %d-%d xy %d,%d-%d,%d | aref %u colour %08X tex format %08X %ux%u, alpha > 0 in %u texels\n",
		(unsigned)run.first, (unsigned)run.last, (unsigned)(run.last - run.first + 1), (unsigned)run.n3d, (unsigned)run.verts,
		outcome_names[run.outcome], (unsigned)run.fkey, (unsigned)run.blend, run.vkey ? "gpu " : "cpu", (unsigned)run.vkey,
		(run.tex & 0x80000000u) ? "fb " : (run.tex ? "id " : "none "), (unsigned)(run.tex & 0x7FFFFFFFu),
		(unsigned)run.flags, (unsigned)(run.zs & 0xFF), (unsigned)((run.zs >> 8) & 0xFF), (unsigned)((run.zs >> 16) & 0xFF),
		(unsigned)(run.zs >> 24), (unsigned)run.stencil,
		(unsigned)(run.scissor & 0xFF) * 8, (unsigned)((run.scissor >> 8) & 0xFF) * 8, (unsigned)((run.scissor >> 16) & 0xFF) * 8,
		(unsigned)(run.scissor >> 24) * 8, (int)(run.z0 * 65535.0f), (int)(run.z1 * 65535.0f), (int)run.x0, (int)run.y0, (int)run.x1, (int)run.y1,
		(unsigned)run.aref, (unsigned)run.color, (unsigned)run.tfmt, (unsigned)(run.tsize & 0xFFFF), (unsigned)(run.tsize >> 16),
		(unsigned)run.topaque);
}

static uint32_t skip_seen[96];
static uint32_t nskip_seen = 0;

// One draw's outcome: counted, its kind logged once if it was not drawn, and traced
static void drawOutcome(GeSgxRenderStats *stats, uint32_t outcome, const GeSgxDraw *d, uint32_t index, uint32_t target_fb,
	const GeSgxFrameData *frame) {
	uint32_t fkey = geSgxShaderKey(d), blend = blendKey(d), vkey = d->gpu ? d->vkey : 0;
	uint32_t tex = !(d->flags & GE_SGX_DRAW_TEXTURED) ? 0 : d->fbtex ? (0x80000000u | (d->fbtex - 1)) : (uint32_t)d->texture;
	if (outcome != OUTCOME_DRAWN) {
		stats->skipped[outcome]++;
		uint32_t k = (outcome << 28) ^ (fkey * 0x9E3779B1u) ^ (vkey * 0x85EBCA77u) ^ ((outcome == GE_SGX_SKIP_FBTEX) ? (tex * 31u) : 0);
		uint32_t i;
		for (i = 0; i < nskip_seen && skip_seen[i] != k; i++)
			;
		if (i == nskip_seen && nskip_seen < sizeof(skip_seen) / sizeof(skip_seen[0])) {
			skip_seen[nskip_seen++] = k;
			GeSgxLog("GeSgx render: not drawn (%s): draw %u into fb %X, fragment %04X blend %08X, vertex %s%04X vtype %06X, texture %s%X, flags %X, %u vertices%s\n",
				outcome_names[outcome], (unsigned)index, (unsigned)target_fb, (unsigned)fkey, (unsigned)blend,
				d->gpu ? "gpu " : "cpu", (unsigned)vkey, (unsigned)d->vtype, (tex & 0x80000000u) ? "fb " : "id ", (unsigned)(tex & 0x7FFFFFFFu),
				(unsigned)d->flags, (unsigned)d->count, outcome == GE_SGX_SKIP_VP && geSgxShaderFailed(vkey) ? " (program failed)" : "");
		}
	}
	if (!rtrace)
		return;
	uint32_t zs = (uint32_t)d->ztest | ((uint32_t)d->cmask << 8) | ((uint32_t)d->prim << 16) | ((uint32_t)d->tex_mode << 24);
	uint32_t st = d->sfunc == 0xFF ? 0xFFFFFFFFu :
		((uint32_t)d->sfunc << 28) | ((uint32_t)(d->sops & 0xFFF) << 16) | ((uint32_t)d->sref << 8) | (uint32_t)d->swmask;
	#define SC8(v) ((uint32_t)((v) < 0 ? 0 : (v) / 8 > 255 ? 255 : (v) / 8))
	uint32_t sc = SC8(d->scissor[0]) | (SC8(d->scissor[1]) << 8) | (SC8(d->scissor[2]) << 16) | (SC8(d->scissor[3]) << 24);
	#undef SC8
	if (run.active && run.outcome == outcome && run.fkey == fkey && run.blend == blend && run.vkey == vkey && run.tex == tex &&
	    run.flags == d->flags && run.zs == zs && run.stencil == st && run.scissor == sc && run.last + 1 == index) {
		run.last = index;
	} else {
		runFlush();
		run.active = 1;
		run.first = run.last = index;
		run.outcome = outcome;
		run.fkey = fkey;
		run.blend = blend;
		run.vkey = vkey;
		run.tex = tex;
		run.flags = d->flags;
		run.zs = zs;
		run.stencil = st;
		run.scissor = sc;
		run.z0 = run.z1 = -1.0f;
		run.x0 = run.y0 = run.x1 = run.y1 = 0.0f;
		run.aref = d->atest_ref;
		run.color = 0;
		run.tfmt = 0;
		run.tsize = run.topaque = 0;
		if ((d->flags & GE_SGX_DRAW_TEXTURED) && !d->fbtex) {
			const SceGxmTexture *gt = geSgxTexGet(d->texture);
			if (gt) {
				run.tfmt = (uint32_t)sceGxmTextureGetFormat(gt);
				uint32_t tw = sceGxmTextureGetWidth(gt), th = sceGxmTextureGetHeight(gt);
				run.tsize = tw | (th << 16);
				// Palette textures: how many texels the alpha test can let through (the minimap's mask)
				const uint8_t *data = sceGxmTextureGetData(gt);
				const uint32_t *pal = sceGxmTextureGetPalette(gt);
				if ((run.tfmt & 0x9F000000u) == SCE_GXM_TEXTURE_BASE_FORMAT_P8 && data && pal && tw * th <= 512 * 512) {
					uint32_t k;
					for (k = 0; k < tw * th; k++)
						run.topaque += (pal[data[k]] >> 24) != 0;
				}
			}
		}
		if (!d->gpu && frame && d->count)
			run.color = frame->vertices[d->base + frame->indices[d->first]].color;
		if (!d->gpu && frame && d->count) {
			uint32_t k;
			run.x0 = run.y0 = 1e9f;
			run.x1 = run.y1 = -1e9f;
			for (k = 0; k < d->count && k < 256; k++) {
				const GeSgxVertex *vv = &frame->vertices[d->base + frame->indices[d->first + k]];
				float iw = vv->w != 0.0f ? 1.0f / vv->w : 0.0f;
				if (vv->x * iw < run.x0) run.x0 = vv->x * iw;
				if (vv->y * iw < run.y0) run.y0 = vv->y * iw;
				if (vv->x * iw > run.x1) run.x1 = vv->x * iw;
				if (vv->y * iw > run.y1) run.y1 = vv->y * iw;
				float z = frame->vertices[d->base + frame->indices[d->first + k]].z;
				if (k == 0 || z < run.z0)
					run.z0 = z;
				if (k == 0 || z > run.z1)
					run.z1 = z;
			}
		}
		run.n3d = run.verts = 0;
	}
	run.n3d += d->gpu ? 1 : 0;
	run.verts += d->count;
}

/*
	The viewport: the vertex programs give positions in pixels of the
	target scaled into 960x544 (x * 2 / 960 - 1). GXM sets a scene's viewport
	to its render target's size, so in a smaller target (Carbon's 256x256
	reflection, 512x512 blur) the picture was squeezed into a corner
	(256 * 256 / 960 = 68 pixels wide). Every scene of this renderer sets
	the 960x544 one: a vertex's pixel is the target's pixel, whatever its
	size. Depth: the programs give 0..1 (z clamped into 0..w).
*/
static void setViewport(SceGxmContext *ctx) {
	sceGxmSetViewport(ctx, 480.0f, 480.0f, 272.0f, -272.0f, 0.0f, 1.0f);
}

/*
	A draw that samples the framebuffer it draws into (Need for Speed
	Carbon's blur: a pass reading one part of its 256x256 framebuffer into
	another). The GPU cannot sample the surface it renders to: the scene
	ends before such a draw, the rendering is copied into a snapshot of the
	same size, and the draw samples the snapshot. Each such draw has its own
	copy (what the draws before it wrote), up to MAX_SNAPSHOTS a frame; the
	ones after that sample the last copy.
*/
#define MAX_SNAPSHOTS 6 // with the scenes between them, within TARGET_SCENES
/*
	The snapshot: a colour surface and texture only (no depth buffer, a
	scene into it needs none), in the kernel's pool when it fits (outside
	the emulator's memory, mapped for the GPU), else a block of its own
*/
typedef struct {
	SceGxmColorSurface sfc;
	SceGxmTexture tex;
	SceGxmRenderTarget *rtgt;
	void *mem;
	uint32_t bytes;
	SceUID uid; // its block (-1: in the pool)
	int w, h;
} Snapshot;
static Snapshot snap;
static int snap_ready = 0;
/*
	The copy's quad: its 4 vertices and 6 indices in uncached memory of its
	own, written when the snapshot is made (vita2d's pool is cached memory:
	what the CPU just wrote there may not have reached the RAM the GPU
	reads, and stale indices made the GPU read past the vertices)
*/
typedef struct {
	GeSgxVertex v[4];
	uint16_t ix[8];
} SnapQuad;
static SnapQuad *snap_quad = NULL;
static uint32_t snap_fb = 0xFFFFFFFFu; // framebuffer whose rendering the snapshot holds (this frame)
static uint32_t snap_count = 0;        // snapshots this frame
static int snap_failures = 0;

static int selfSampling(const GeSgxDraw *d, uint32_t fb) {
	return (d->flags & GE_SGX_DRAW_TEXTURED) && !(d->flags & GE_SGX_DRAW_CLEAR) && d->fbtex == fb + 1;
}

static void snapshotFree(SceGxmContext *ctx) {
	if (!snap_ready)
		return;
	if (ctx)
		sceGxmFinish(ctx); // the GPU may still read or write it
	if (snap.rtgt)
		sceGxmDestroyRenderTarget(snap.rtgt);
	if (snap.uid >= 0) {
		sceGxmUnmapMemory(snap.mem);
		sceKernelFreeMemBlock(snap.uid);
	} else if (snap.mem) {
		geSgxPoolFree(snap.mem, snap.bytes);
	}
	memset(&snap, 0, sizeof(snap));
	snap_ready = 0;
}

static int snapshotMake(int w, int h) {
	memset(&snap, 0, sizeof(snap));
	snap.uid = -1;
	snap.w = w;
	snap.h = h;
	snap.bytes = ((uint32_t)w * h * 4 + 0xFFF) & ~0xFFFu;
	const char *where = "pool";
	snap.mem = geSgxPoolAlloc(snap.bytes);
	if (!snap.mem) {
		where = "RAM";
		snap.uid = sceKernelAllocMemBlock("SceGeSgxSnapshot", SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE, snap.bytes, NULL);
		if (snap.uid >= 0) {
			sceKernelGetMemBlockBase(snap.uid, &snap.mem);
			if (sceGxmMapMemory(snap.mem, snap.bytes, SCE_GXM_MEMORY_ATTRIB_READ | SCE_GXM_MEMORY_ATTRIB_WRITE) < 0) {
				sceKernelFreeMemBlock(snap.uid);
				snap.uid = -1;
				snap.mem = NULL;
			}
		} else {
			snap.mem = NULL;
		}
	}
	if (!snap_quad)
		snap_quad = gpuAlloc(SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE, 4096, SCE_GXM_MEMORY_ATTRIB_READ);
	int err = snap.mem && snap_quad ? 0 : -1;
	if (!err) {
		// Pixels of the target (the viewport is the screen's); the GPU no longer reads the old ones (finished)
		int k;
		for (k = 0; k < 4; k++) {
			GeSgxVertex *v = &snap_quad->v[k];
			v->x = (k & 1) ? (float)w : 0.0f;
			v->y = (k & 2) ? (float)h : 0.0f;
			v->z = 0.5f;
			v->w = 1.0f;
			v->u = (k & 1) ? 1.0f : 0.0f;
			v->v = (k & 2) ? 1.0f : 0.0f;
			v->color = 0xFFFFFFFFu;
			v->fog = 1.0f;
		}
		static const uint16_t ix[6] = { 0, 1, 2, 1, 3, 2 };
		memcpy(snap_quad->ix, ix, sizeof(ix));
	}
	if (!err) {
		// Alpha 1 (the draws sampling it with its alpha get an opaque framebuffer, as the targets' are)
		uint32_t *px = snap.mem, k, n = (uint32_t)w * h;
		for (k = 0; k < n; k++)
			px[k] = 0xFF000000u;
		err = sceGxmColorSurfaceInit(&snap.sfc, SCE_GXM_COLOR_FORMAT_A8B8G8R8, SCE_GXM_COLOR_SURFACE_LINEAR,
			SCE_GXM_COLOR_SURFACE_SCALE_NONE, SCE_GXM_OUTPUT_REGISTER_SIZE_32BIT, w, h, w, snap.mem);
	}
	if (!err)
		err = sceGxmTextureInitLinear(&snap.tex, snap.mem, SCE_GXM_TEXTURE_FORMAT_A8B8G8R8, w, h, 0);
	if (!err) {
		SceGxmRenderTargetParams params;
		memset(&params, 0, sizeof(params));
		params.width = w;
		params.height = h;
		params.scenesPerFrame = TARGET_SCENES; // more: invalid value
		params.multisampleMode = SCE_GXM_MULTISAMPLE_NONE;
		params.driverMemBlock = -1;
		err = sceGxmCreateRenderTarget(&params, &snap.rtgt);
	}
	snap_ready = 1;
	if (err) {
		if (snap_failures++ < 8)
			GeSgxLog("GeSgx render: snapshot %dx%d FAILED (%08X)\n", w, h, (unsigned)err);
		snapshotFree(NULL);
		return 0;
	}
	GeSgxLog("GeSgx render: snapshot %dx%d in %s, for draws sampling their own framebuffer\n", w, h, where);
	return 1;
}

// 1: copied; 0: a scene but no copy; -1: no scene (nothing to copy into)
static int snapshotTarget(SceGxmContext *ctx, const vita2d_texture *src, int w, int h, int reset_pool) {
	if (snap_ready && (snap.w != w || snap.h != h))
		snapshotFree(ctx);
	if (!snap_ready && !snapshotMake(w, h))
		return -1;

	// Replace with the texture's colour, no blending: a copy (alpha is never written, the clear's 1 stays)
	GeSgxDraw d;
	memset(&d, 0, sizeof(d));
	d.flags = GE_SGX_DRAW_TEXTURED | GE_SGX_DRAW_TCC;
	d.tfunc = 3;
	d.sfunc = 0xFF;
	FragmentProgram *fp = fragmentProgram(&d, vertex_gxp, 0);
	if (!fp)
		return -1;

	if (reset_pool)
		vita2d_pool_reset();
	sceGxmBeginScene(ctx, 0, snap.rtgt, NULL, NULL, NULL, &snap.sfc, NULL);
	setViewport(ctx);
	{
		SceGxmTexture tex = src->gxm_tex;
		sceGxmTextureSetUAddrMode(&tex, SCE_GXM_TEXTURE_ADDR_CLAMP);
		sceGxmTextureSetVAddrMode(&tex, SCE_GXM_TEXTURE_ADDR_CLAMP);
		sceGxmTextureSetMinFilter(&tex, SCE_GXM_TEXTURE_FILTER_POINT);
		sceGxmTextureSetMagFilter(&tex, SCE_GXM_TEXTURE_FILTER_POINT);
		sceGxmSetFrontDepthFunc(ctx, SCE_GXM_DEPTH_FUNC_ALWAYS);
		sceGxmSetFrontDepthWriteEnable(ctx, SCE_GXM_DEPTH_WRITE_DISABLED);
		stencilOff(ctx);
		sceGxmSetCullMode(ctx, SCE_GXM_CULL_NONE);
		sceGxmSetRegionClip(ctx, SCE_GXM_REGION_CLIP_OUTSIDE, 0, 0, w - 1, h - 1);
		sceGxmSetVertexProgram(ctx, vertex_program);
		sceGxmSetFragmentProgram(ctx, fp->program);
		if (fp->alpha_ref || fp->env || fp->fog_color) {
			void *uniforms = NULL;
			sceGxmReserveFragmentDefaultUniformBuffer(ctx, &uniforms);
			if (fp->alpha_ref) {
				float ref = 0.0f;
				sceGxmSetUniformDataF(uniforms, fp->alpha_ref, 0, 1, &ref);
			}
			if (fp->env)
				setColorUniform(uniforms, fp->env, 0);
			if (fp->fog_color)
				setColorUniform(uniforms, fp->fog_color, 0);
		}
		sceGxmSetFragmentTexture(ctx, 0, &tex);
		sceGxmSetVertexStream(ctx, 0, snap_quad->v);
		sceGxmDraw(ctx, SCE_GXM_PRIMITIVE_TRIANGLES, SCE_GXM_INDEX_FORMAT_U16, snap_quad->ix, 6);
		sceGxmSetRegionClip(ctx, SCE_GXM_REGION_CLIP_OUTSIDE, 0, 0, 959, 543);
	}
	sceGxmEndScene(ctx, NULL, NULL);
	return 1;
}

/*
	Draws [first, last) of the frame (one framebuffer) into the scene being
	drawn, a target of w x h pixels.
*/
static void renderDraws(SceGxmContext *ctx, const GeSgxFrameData *frame, uint32_t first, uint32_t last,
	int tw, int th, uint32_t target_fb, GeSgxRenderStats *stats) {
	setViewport(ctx);
	sceGxmSetFrontDepthFunc(ctx, SCE_GXM_DEPTH_FUNC_ALWAYS);
	sceGxmSetFrontDepthWriteEnable(ctx, SCE_GXM_DEPTH_WRITE_DISABLED);
	int last_zfunc = SCE_GXM_DEPTH_FUNC_ALWAYS, last_zwrite = 0;
	sceGxmSetCullMode(ctx, SCE_GXM_CULL_NONE);
	sceGxmSetFrontPointLineWidth(ctx, GE_SGX_SCALE);
	sceGxmSetVertexProgram(ctx, vertex_program);
	const SceGxmVertexProgram *last_vp = vertex_program;
	stencilOff(ctx);
	uint32_t last_stencil = 0xFFFFFFFFu;

	int16_t last_scissor[4] = { -1, -1, -1, -1 };
	int scissor_empty = 0;
	const SceGxmFragmentProgram *last_fp = NULL;
	uint32_t i;

	for (i = first; i < last; i++) {
		const GeSgxDraw *d = &frame->draws[i];

		// Debug mode (R + SELECT): one feature off for every draw, to see what hides something
		GeSgxDraw dbg;
		if (ge_sgx_debug_mode && !(d->flags & GE_SGX_DRAW_CLEAR)) {
			dbg = *d;
			if (ge_sgx_debug_mode == 1)
				dbg.flags &= ~GE_SGX_DRAW_ALPHATEST;
			else if (ge_sgx_debug_mode == 2)
				dbg.flags &= ~GE_SGX_DRAW_FOG;
			else if (ge_sgx_debug_mode == 4)
				dbg.flags &= ~(GE_SGX_DRAW_TEXTURED | GE_SGX_DRAW_TCC);
			else if (ge_sgx_debug_mode == 5)
				dbg.sfunc = 0xFF;
			d = &dbg;
		}

		// A clear writing nothing
		if ((d->flags & GE_SGX_DRAW_CLEAR) && !(d->flags & (GE_SGX_DRAW_WRITE_COLOR | GE_SGX_DRAW_WRITE_DEPTH | GE_SGX_DRAW_WRITE_ALPHA))) {
			drawOutcome(stats, GE_SGX_SKIP_CLEAR, d, i, target_fb, frame);
			continue;
		}

		// The texture first: a framebuffer texture needs that framebuffer's rendering (not this one)
		SceGxmTexture tex;
		if (d->flags & GE_SGX_DRAW_TEXTURED) {
			const SceGxmTexture *t = NULL;
			if (d->fbtex) {
				uint32_t sfb = d->fbtex - 1;
				Target *src = sfb == target_fb ? NULL : findTarget(sfb);
				if (src)
					t = &src->tex->gxm_tex;
				else if (sfb == target_fb && snap_fb == sfb && snap_ready)
					t = &snap.tex; // its own framebuffer: the copy made before it
			} else {
				t = geSgxTexGet(d->texture);
			}
			if (!t) {
				drawOutcome(stats, d->fbtex ? GE_SGX_SKIP_FBTEX : GE_SGX_SKIP_TEX, d, i, target_fb, frame);
				continue;
			}
			tex = *t;
			sceGxmTextureSetUAddrMode(&tex, (d->tex_mode & GE_SGX_TEX_CLAMP_U) ? SCE_GXM_TEXTURE_ADDR_CLAMP : SCE_GXM_TEXTURE_ADDR_REPEAT);
			sceGxmTextureSetVAddrMode(&tex, (d->tex_mode & GE_SGX_TEX_CLAMP_V) ? SCE_GXM_TEXTURE_ADDR_CLAMP : SCE_GXM_TEXTURE_ADDR_REPEAT);
			sceGxmTextureSetMinFilter(&tex, (d->tex_mode & GE_SGX_TEX_LINEAR_MIN) ? SCE_GXM_TEXTURE_FILTER_LINEAR : SCE_GXM_TEXTURE_FILTER_POINT);
			sceGxmTextureSetMagFilter(&tex, (d->tex_mode & GE_SGX_TEX_LINEAR_MAG) ? SCE_GXM_TEXTURE_FILTER_LINEAR : SCE_GXM_TEXTURE_FILTER_POINT);
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
				drawOutcome(stats, GE_SGX_SKIP_VP, d, i, target_fb, frame);
				continue;
			}
			vp = g->program;
			vgxp = g->gxp;
		}

		/*
			The scissor: the region clip (below) works in 32x32 tiles; a scissor
			not on them (the minimap's square: its lines showed past it, outside
			the circle) is made exact in the fragment program
		*/
		int sx0 = d->scissor[0] < 0 ? 0 : d->scissor[0], sy0 = d->scissor[1] < 0 ? 0 : d->scissor[1];
		int sx1 = d->scissor[2] > tw ? tw : d->scissor[2], sy1 = d->scissor[3] > th ? th : d->scissor[3];
		int exact = (sx0 & 31) || (sy0 & 31) || (sx1 != tw && (sx1 & 31)) || (sy1 != th && (sy1 & 31));
		FragmentProgram *fp = fragmentProgram(d, vgxp, exact ? GE_SGX_KEY_SCISSOR : 0);
		// The exact scissor's program not compiled yet (the app compiles it at Adrenaline's next start): the tiles' scissor
		if (!fp && exact)
			fp = fragmentProgram(d, vgxp, 0);
		if (!fp) {
			stats->waiting++;
			drawOutcome(stats, GE_SGX_SKIP_FP, d, i, target_fb, frame);
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

		if (memcmp(last_scissor, d->scissor, sizeof(last_scissor)) != 0) {
			memcpy(last_scissor, d->scissor, sizeof(last_scissor));
			int x0 = d->scissor[0], y0 = d->scissor[1], x1 = d->scissor[2], y1 = d->scissor[3];
			if (x0 < 0) x0 = 0;
			if (y0 < 0) y0 = 0;
			if (x1 > tw) x1 = tw;
			if (y1 > th) y1 = th;
			scissor_empty = x1 <= x0 || y1 <= y0;
			// Tile granularity (32x32): exact for the full screen, the usual case
			if (!scissor_empty)
				sceGxmSetRegionClip(ctx, SCE_GXM_REGION_CLIP_OUTSIDE, x0, y0, x1 - 1, y1 - 1);
		}
		if (scissor_empty) {
			drawOutcome(stats, GE_SGX_SKIP_SCISSOR, d, i, target_fb, frame);
			continue;
		}

		int zfunc = (d->flags & GE_SGX_DRAW_CLEAR) ? SCE_GXM_DEPTH_FUNC_ALWAYS : depthFunc(d->ztest);
		if (ge_sgx_debug_mode == 3)
			zfunc = SCE_GXM_DEPTH_FUNC_ALWAYS;
		int zwrite = (d->flags & GE_SGX_DRAW_WRITE_DEPTH) != 0;
		if (zfunc != last_zfunc) {
			sceGxmSetFrontDepthFunc(ctx, (SceGxmDepthFunc)zfunc);
			last_zfunc = zfunc;
		}
		if (zwrite != last_zwrite) {
			sceGxmSetFrontDepthWriteEnable(ctx, zwrite ? SCE_GXM_DEPTH_WRITE_ENABLED : SCE_GXM_DEPTH_WRITE_DISABLED);
			last_zwrite = zwrite;
		}

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

		if (fp->alpha_ref || fp->env || fp->fog_color || fp->scissor) {
			void *uniforms = NULL;
			sceGxmReserveFragmentDefaultUniformBuffer(ctx, &uniforms);
			if (fp->scissor) {
				float sc[4] = { (float)sx0, (float)sy0, (float)sx1, (float)sy1 };
				sceGxmSetUniformDataF(uniforms, fp->scissor, 0, 4, sc);
			}
			if (fp->alpha_ref) {
				float ref = d->atest_ref / 255.0f;
				sceGxmSetUniformDataF(uniforms, fp->alpha_ref, 0, 1, &ref);
			}
			if (fp->env)
				setColorUniform(uniforms, fp->env, d->env);
			if (fp->fog_color)
				setColorUniform(uniforms, fp->fog_color, d->fog_color);
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
		drawOutcome(stats, OUTCOME_DRAWN, d, i, target_fb, frame);
	}
	runFlush();

	// Back to the state vita2d expects
	sceGxmSetFrontDepthFunc(ctx, SCE_GXM_DEPTH_FUNC_ALWAYS);
	sceGxmSetFrontDepthWriteEnable(ctx, SCE_GXM_DEPTH_WRITE_DISABLED);
	stencilOff(ctx);
	sceGxmSetRegionClip(ctx, SCE_GXM_REGION_CLIP_OUTSIDE, 0, 0, 959, 543);
	sceGxmSetFrontPointLineWidth(ctx, 1);
}

/*
	Diagnostics (L + SELECT): the renderings of the traced frame written as
	pictures (ux0:data/adrenaline/sgx_dump/<frame>_<fb>.bmp), what this
	renderer drew, to compare with what is on the screen
*/
static void dumpBmp(const char *path, const uint32_t *px, int w, int h, int stride) {
	SceUID fd = sceIoOpen(path, SCE_O_WRONLY | SCE_O_CREAT | SCE_O_TRUNC, 0777);
	if (fd < 0)
		return;
	uint8_t hdr[54];
	memset(hdr, 0, sizeof(hdr));
	uint32_t size = 54 + (uint32_t)w * h * 4;
	hdr[0] = 'B'; hdr[1] = 'M';
	memcpy(hdr + 2, &size, 4);
	uint32_t v = 54; memcpy(hdr + 10, &v, 4);
	v = 40; memcpy(hdr + 14, &v, 4);
	int32_t iw = w, ih = -h; // top-down
	memcpy(hdr + 18, &iw, 4);
	memcpy(hdr + 22, &ih, 4);
	uint16_t s16 = 1; memcpy(hdr + 26, &s16, 2);
	s16 = 32; memcpy(hdr + 28, &s16, 2);
	sceIoWrite(fd, hdr, sizeof(hdr));
	static uint32_t row[1024];
	int x, y;
	for (y = 0; y < h; y++) {
		const uint32_t *src = px + (size_t)y * stride;
		for (x = 0; x < w && x < 1024; x++) {
			uint32_t c = src[x]; // ABGR -> BGRA in memory (ARGB word)
			row[x] = (c & 0xFF00FF00u) | ((c & 0xFFu) << 16) | ((c >> 16) & 0xFFu);
		}
		sceIoWrite(fd, row, (SceSize)w * 4);
	}
	sceIoClose(fd);
}

static void dumpFrame(SceGxmContext *ctx, uint32_t frame_no) {
	sceGxmFinish(ctx);
	sceIoMkdir("ux0:data/adrenaline/sgx_dump", 0777);
	char path[96];
	int i;
	for (i = 0; i < GE_SGX_MAX_FBS; i++) {
		Target *t = &targets[i];
		if (!t->tex)
			continue;
		snprintf(path, sizeof(path), "ux0:data/adrenaline/sgx_dump/%u_%X.bmp", (unsigned)frame_no, (unsigned)t->fb);
		dumpBmp(path, vita2d_texture_get_datap(t->tex), t->w, t->h, (int)vita2d_texture_get_stride(t->tex) / 4);
		GeSgxLog("GeSgx render: dumped %s (%dx%d)\n", path, t->w, t->h);
	}
	if (snap_ready) {
		snprintf(path, sizeof(path), "ux0:data/adrenaline/sgx_dump/%u_snapshot.bmp", (unsigned)frame_no);
		dumpBmp(path, snap.mem, snap.w, snap.h, snap.w);
		GeSgxLog("GeSgx render: dumped %s (%dx%d)\n", path, snap.w, snap.h);
	}
}

/*
	Diagnostics: the last HIST_FRAMES frames, each in one line (what it
	flipped, what was shown, its draws in runs of the same texture: handle,
	size, count; fb<X> for a framebuffer's rendering), written to the log
	with an asked trace (L + SELECT): a glitch of one frame is gone before
	the buttons are pressed, its frame is still here.
*/
#define HIST_FRAMES 160
#define HIST_BYTES 1024
static char hist[HIST_FRAMES][HIST_BYTES];
static uint32_t hist_next = 0;

static void histRecord(uint32_t frame_no, const GeSgxFrameData *frame, uint32_t fb, const Target *show) {
	char *l = hist[hist_next % HIST_FRAMES];
	hist_next++;
	int n = snprintf(l, HIST_BYTES, "GeSgx H: frame %u flips %X shows %X, %u draws:", (unsigned)frame_no, (unsigned)fb,
		show ? (unsigned)show->fb : 0xFFFFFFFFu, (unsigned)frame->ndraws);
	uint32_t i, run_key = 0xFFFFFFFFu, run_n = 0, run_fb = 0;
	for (i = 0; i <= frame->ndraws && n > 0 && n < HIST_BYTES - 48; i++) {
		uint32_t key = 0xFFFFFFFEu, dfb = 0;
		if (i < frame->ndraws) {
			const GeSgxDraw *d = &frame->draws[i];
			dfb = d->fb;
			key = !(d->flags & GE_SGX_DRAW_TEXTURED) ? 0xFFFFFFFDu : d->fbtex ? (0x80000000u | (d->fbtex - 1)) : (uint32_t)d->texture;
		}
		if (i < frame->ndraws && key == run_key && dfb == run_fb) {
			run_n++;
			continue;
		}
		if (run_n) {
			if (run_key == 0xFFFFFFFDu) {
				n += snprintf(l + n, HIST_BYTES - n, " %X:-x%u", (unsigned)run_fb, (unsigned)run_n);
			} else if (run_key & 0x80000000u) {
				n += snprintf(l + n, HIST_BYTES - n, " %X:fb%Xx%u", (unsigned)run_fb, (unsigned)(run_key & 0x7FFFFFFFu), (unsigned)run_n);
			} else {
				const SceGxmTexture *t = geSgxTexGet((int32_t)run_key);
				n += snprintf(l + n, HIST_BYTES - n, " %X:%X/%ux%ux%u", (unsigned)run_fb, (unsigned)run_key,
					t ? (unsigned)sceGxmTextureGetWidth(t) : 0u, t ? (unsigned)sceGxmTextureGetHeight(t) : 0u, (unsigned)run_n);
			}
		}
		run_key = key;
		run_fb = dfb;
		run_n = 1;
	}
	if (n > 0 && n < HIST_BYTES - 2) {
		l[n++] = '\n';
		l[n] = '\0';
	} else {
		l[HIST_BYTES - 2] = '\n';
		l[HIST_BYTES - 1] = '\0';
	}
}

static void histDump(void) {
	uint32_t k, count = hist_next < HIST_FRAMES ? hist_next : HIST_FRAMES;
	GeSgxLog("GeSgx H: the last %u frames\n", (unsigned)count);
	for (k = hist_next - count; k != hist_next; k++)
		GeSgxLogText(hist[k % HIST_FRAMES], strlen(hist[k % HIST_FRAMES]));
}

vita2d_texture *geSgxRenderFrame(SceGxmContext *ctx, const GeSgxFrameData *frame, uint32_t fb,
	const GeSgxAlias *aliases, int naliases, GeSgxRenderStats *stats, int *scenes) {
	SceUInt64 t0 = sceKernelGetProcessTimeWide();
	memset(stats, 0, sizeof(*stats));
	*scenes = 0;

	if (!patcher || !vertexProgram())
		return NULL;
	target_clock++;

	// Diagnostics: a frame traced now and then (three of the race, two others)
	static uint32_t rframes = 0, last_trace = 0, big_traces = 0;
	rframes++;
	rtrace = 0;
	int asked = ge_sgx_trace_render && frame->ndraws;
	if (asked) {
		ge_sgx_trace_render = 0;
		histDump();
	}
	if (asked || (frame->ndraws >= 1000 && big_traces < 3 && rframes - last_trace >= 300) || rframes == 600 || rframes == 1800) {
		rtrace = 1;
		rtrace_len = rtrace_lines = 0;
		last_trace = rframes;
		if (frame->ndraws >= 1000)
			big_traces++;
		rtraceAdd("GeSgx R: frame %u%s: %u draws, %u vertices, %u indices, flipped fb %X, %d aliases, framebuffers:", (unsigned)rframes,
			asked ? " (asked)" : "",
			(unsigned)frame->ndraws, (unsigned)frame->nvertices, (unsigned)frame->nindices, (unsigned)fb, naliases);
		uint32_t k;
		for (k = 0; k < frame->nfbs; k++)
			rtraceAdd(" %X %ux%u", (unsigned)frame->fbs[k].fb, (unsigned)frame->fbs[k].w, (unsigned)frame->fbs[k].h);
		rtraceAdd("\n");
	}

	/*
		One scene per run of draws into one framebuffer, each into that
		framebuffer's rendering; a draw sampling that framebuffer starts a
		scene of its own, after a copy of it
	*/
	snap_fb = 0xFFFFFFFFu;
	snap_count = 0;
	uint32_t i = 0;
	while (i < frame->ndraws) {
		uint32_t dfb = frame->draws[i].fb, j = i + 1;
		while (j < frame->ndraws && frame->draws[j].fb == dfb && !(snap_count < MAX_SNAPSHOTS && selfSampling(&frame->draws[j], dfb)))
			j++;

		uint32_t w = 480, h = 272, k;
		for (k = 0; k < frame->nfbs; k++) {
			if (frame->fbs[k].fb == dfb && frame->fbs[k].w && frame->fbs[k].h) {
				w = frame->fbs[k].w;
				h = frame->fbs[k].h;
			}
		}
		Target *t = getTarget(dfb, w * GE_SGX_SCALE, h * GE_SGX_SCALE);
		if (!t) {
			rtraceAdd("GeSgx R: scene: draws %u-%u into fb %X: no rendering (%ux%u)\n", (unsigned)i, (unsigned)(j - 1), (unsigned)dfb, (unsigned)w, (unsigned)h);
			stats->other_fb += j - i;
			i = j;
			continue;
		}
		if (selfSampling(&frame->draws[i], dfb) && snap_count < MAX_SNAPSHOTS && !t->fresh) {
			snap_count++;
			int r = snapshotTarget(ctx, t->tex, t->w, t->h, *scenes == 0);
			snap_fb = r > 0 ? dfb : 0xFFFFFFFFu;
			rtraceAdd("GeSgx R: scene %d: snapshot of fb %X for draw %u%s\n", *scenes, (unsigned)dfb, (unsigned)i, r > 0 ? "" : " FAILED");
			if (asked && r > 0) {
				char path[96];
				sceGxmFinish(ctx);
				snprintf(path, sizeof(path), "ux0:data/adrenaline/sgx_dump/%u_s%d_snapshot.bmp", (unsigned)rframes, *scenes);
				dumpBmp(path, snap.mem, snap.w, snap.h, snap.w);
			}
			if (r >= 0)
				(*scenes)++;
		}
		rtraceAdd("GeSgx R: scene %d: draws %u-%u into fb %X, rendering %dx%d%s\n", *scenes, (unsigned)i, (unsigned)(j - 1), (unsigned)dfb,
			t->w, t->h, t->fresh ? " (new, cleared)" : "");
		// Depth and stencil: kept from its scene before in this frame, kept for one after
		int again = 0;
		for (k = j; k < frame->ndraws && !again; k++)
			again = frame->draws[k].fb == dfb;
		int resumed = t->scene_frame == rframes + 1;
		sceGxmDepthStencilSurfaceSetForceLoadMode(&t->tex->gxm_sfd,
			resumed ? SCE_GXM_DEPTH_STENCIL_FORCE_LOAD_ENABLED : SCE_GXM_DEPTH_STENCIL_FORCE_LOAD_DISABLED);
		sceGxmDepthStencilSurfaceSetForceStoreMode(&t->tex->gxm_sfd,
			again ? SCE_GXM_DEPTH_STENCIL_FORCE_STORE_ENABLED : SCE_GXM_DEPTH_STENCIL_FORCE_STORE_DISABLED);
		t->scene_frame = rframes + 1;
		if (resumed || again)
			rtraceAdd("GeSgx R:   depth and stencil%s%s\n", resumed ? " loaded" : "", again ? " stored" : "");
		vita2d_start_drawing_advanced(t->tex, *scenes == 0 ? VITA_2D_RESET_POOL : 0);
		if (t->fresh) {
			vita2d_set_clear_color(RGBA8(0, 0, 0, 255));
			vita2d_clear_screen();
			t->fresh = 0;
		}
		renderDraws(ctx, frame, i, j, t->w, t->h, dfb, stats);
		vita2d_end_drawing();
		// Asked (L + SELECT): the smaller renderings after each of their scenes (the blur's steps)
		if (asked && (t->w != 960 || t->h != 544)) {
			char path[96];
			sceGxmFinish(ctx);
			sceIoMkdir("ux0:data/adrenaline/sgx_dump", 0777);
			snprintf(path, sizeof(path), "ux0:data/adrenaline/sgx_dump/%u_s%d_%X.bmp", (unsigned)rframes, *scenes, (unsigned)dfb);
			dumpBmp(path, vita2d_texture_get_datap(t->tex), t->w, t->h, (int)vita2d_texture_get_stride(t->tex) / 4);
		}
		(*scenes)++;
		if (dfb != fb)
			stats->other_fb += j - i;
		i = j;
	}

	stats->render_us = (uint32_t)(sceKernelGetProcessTimeWide() - t0);

	// What is shown: the flipped framebuffer's rendering, or the one copied into it
	uint32_t k;
	Target *show = findTarget(fb);
	int a;
	for (a = 0; !show && a < naliases; a++) {
		if (aliases[a].dst == fb)
			show = findTarget(aliases[a].src);
	}
	/*
		A flipped framebuffer not drawn since before the newest screen-sized
		rendering: its picture came from elsewhere (a copy, the CPU), and its
		rendering is an old one, maybe from when the game used it for
		something else. Valkyria Chronicles 2 flips a third framebuffer
		(15C000) it also composes its title's logo and menu in: shown as it
		was, those came up enlarged over the screen for a frame. The newest
		screen-sized rendering is shown instead.
	*/
	if (show && show->scene_frame != rframes + 1) {
		Target *newest = NULL;
		uint32_t k2;
		for (k2 = 0; k2 < GE_SGX_MAX_FBS; k2++) {
			Target *c = &targets[k2];
			if (c->tex && c->w == show->w && c->h == show->h && c->scene_frame &&
			    (int32_t)(c->scene_frame - show->scene_frame) > 0 && (!newest || (int32_t)(c->scene_frame - newest->scene_frame) > 0))
				newest = c;
		}
		if (newest) {
			static uint32_t stale_logs = 0;
			if (stale_logs++ < 20)
				GeSgxLog("GeSgx render: frame %u flips fb %X, last drawn %u frames ago: shown fb %X's newer rendering\n",
					(unsigned)rframes, (unsigned)fb, (unsigned)(rframes + 1 - show->scene_frame), (unsigned)newest->fb);
			show = newest;
		}
	}
	/*
		Frames without draws for half a second: the game writes the picture
		itself (CPU, transfers, a video), so the renderings hold older ones; the
		PSP picture is what it shows. A few empty frames are not that: frames
		the walker could not read whole (its lists were already overwritten)
		or repeated flips, and the PSP picture then was a flash of another one.
	*/
	static uint32_t empty_frames = 0;
	empty_frames = frame->ndraws ? 0 : empty_frames + 1;
	if (empty_frames >= 30) {
		show = NULL;
		shown_fb = 0;
	}
	if (show) {
		show->used = target_clock;
		shown_fb = show->fb + 1;
	}

	/*
		Diagnostics: frames that do not show their own fresh rendering (nothing:
		the PSP picture; another framebuffer's; or one not drawn this frame,
		which still holds an older picture) - candidates for flicker
	*/
	static uint32_t odd_logs = 0;
	uint32_t now_shown = show ? show->fb : 0xFFFFFFFFu;
	int drawn_now = 0;
	for (k = 0; show && k < frame->ndraws; k++) {
		if (frame->draws[k].fb == show->fb) {
			drawn_now = 1;
			break;
		}
	}
	if ((!show || now_shown != fb || !drawn_now) && odd_logs < 100) {
		odd_logs++;
		GeSgxLog("GeSgx render: frame %u flips fb %X (%u draws): shown %s %X%s\n", (unsigned)rframes, (unsigned)fb, (unsigned)frame->ndraws,
			show ? "rendering of fb" : "nothing (the PSP picture)", show ? (unsigned)now_shown : 0u,
			show && !drawn_now ? ", not drawn this frame (an older picture)" : "");
	}

	if (rtrace) {
		rtraceAdd("GeSgx R: end: %u drawn, not drawn: clear %u, fb texture %u, texture %u, vertex program %u, fragment program %u, scissor %u, other fb %u; shown %X\n",
			(unsigned)stats->draws, (unsigned)stats->skipped[0], (unsigned)stats->skipped[1], (unsigned)stats->skipped[2],
			(unsigned)stats->skipped[3], (unsigned)stats->skipped[4], (unsigned)stats->skipped[5], (unsigned)stats->other_fb, (unsigned)now_shown);
		GeSgxLogText(rtrace_text, rtrace_len);
		rtrace = 0;
		if (asked)
			dumpFrame(ctx, rframes);
	}
	histRecord(rframes, frame, fb, show);
	return show ? show->tex : NULL;
}

vita2d_texture *geSgxRenderShown(void) {
	Target *t = shown_fb ? findTarget(shown_fb - 1) : NULL;
	return t ? t->tex : NULL;
}
