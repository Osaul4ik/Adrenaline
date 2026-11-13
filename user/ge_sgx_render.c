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

#include "ge_sgx.h"
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
		if (d->flags & GE_SGX_DRAW_WRITE_ALPHA)
			mask |= SCE_GXM_COLOR_MASK_A;
		return 0x80000000u | mask;
	}
	if (!(d->flags & GE_SGX_DRAW_BLEND))
		return 0;
	uint32_t src = factor(d->blend & 0xF, d->fix_a, 1);
	uint32_t dst = factor((d->blend >> 4) & 0xF, d->fix_b, 0);
	uint32_t func = blendFunc((d->blend >> 8) & 7);
	return 0x40000000u | func | (src << 4) | (dst << 8);
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

static FragmentProgram *fragmentProgram(const GeSgxDraw *d, const SceGxmProgram *vgxp) {
	uint32_t key = geSgxShaderKey(d);
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
		Alpha is never written: the target keeps the alpha 1 of its clear, so
		it shows opaque (the PSP keeps the stencil there, which only matters
		for DSTALPHA blending; that is not emulated yet).
	*/
	SceGxmBlendInfo info;
	memset(&info, 0, sizeof(info));
	info.colorMask = SCE_GXM_COLOR_MASK_R | SCE_GXM_COLOR_MASK_G | SCE_GXM_COLOR_MASK_B;
	info.colorFunc = SCE_GXM_BLEND_FUNC_NONE;
	info.alphaFunc = SCE_GXM_BLEND_FUNC_NONE;
	info.colorSrc = info.alphaSrc = SCE_GXM_BLEND_FACTOR_ONE;
	info.colorDst = info.alphaDst = SCE_GXM_BLEND_FACTOR_ZERO;
	if (blend & 0x80000000u) {
		info.colorMask = blend & (SCE_GXM_COLOR_MASK_R | SCE_GXM_COLOR_MASK_G | SCE_GXM_COLOR_MASK_B);
	} else if (blend & 0x40000000u) {
		info.colorFunc = blend & 0xF;
		info.colorSrc = (blend >> 4) & 0xF;
		info.colorDst = (blend >> 8) & 0xF;
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

static int shown(uint32_t draw_fb, uint32_t fb, const GeSgxAlias *aliases, int naliases) {
	if (draw_fb == fb)
		return 1;
	int i;
	for (i = 0; i < naliases; i++) {
		if (aliases[i].dst == fb && aliases[i].src == draw_fb)
			return 1;
	}
	return 0;
}

void geSgxRender(SceGxmContext *ctx, const GeSgxFrameData *frame, uint32_t fb,
	const GeSgxAlias *aliases, int naliases, GeSgxRenderStats *stats) {
	SceUInt64 t0 = sceKernelGetProcessTimeWide();
	memset(stats, 0, sizeof(*stats));

	if (!patcher || !vertexProgram())
		return;

	// No culling (yet): the depth test hides the back faces
	sceGxmSetFrontDepthFunc(ctx, SCE_GXM_DEPTH_FUNC_ALWAYS);
	sceGxmSetFrontDepthWriteEnable(ctx, SCE_GXM_DEPTH_WRITE_DISABLED);
	int last_zfunc = SCE_GXM_DEPTH_FUNC_ALWAYS, last_zwrite = 0;
	sceGxmSetCullMode(ctx, SCE_GXM_CULL_NONE);
	sceGxmSetFrontPointLineWidth(ctx, GE_SGX_SCALE);
	sceGxmSetVertexProgram(ctx, vertex_program);
	const SceGxmVertexProgram *last_vp = vertex_program;

	int16_t last_scissor[4] = { -1, -1, -1, -1 };
	int scissor_empty = 0;
	const SceGxmFragmentProgram *last_fp = NULL;
	uint32_t i;

	for (i = 0; i < frame->ndraws; i++) {
		const GeSgxDraw *d = &frame->draws[i];

		if (!shown(d->fb, fb, aliases, naliases)) {
			stats->other_fb++;
			continue;
		}

		// Clear of the stencil only: nothing to draw (no stencil yet)
		if ((d->flags & GE_SGX_DRAW_CLEAR) && !(d->flags & (GE_SGX_DRAW_WRITE_COLOR | GE_SGX_DRAW_WRITE_DEPTH)))
			continue;

		// The vertex program: the CPU transformed vertices', or the GPU transform's for this vertex shape
		const SceGxmVertexProgram *vp = vertex_program;
		const SceGxmProgram *vgxp = vertex_gxp;
		if (d->gpu) {
			int waiting = 0;
			GpuProgram *g = gpuProgram(d, &waiting);
			if (!g) {
				if (waiting)
					stats->waiting++;
				continue;
			}
			vp = g->program;
			vgxp = g->gxp;
		}

		FragmentProgram *fp = fragmentProgram(d, vgxp);
		if (!fp) {
			stats->waiting++;
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
			if (x1 > 960) x1 = 960;
			if (y1 > 544) y1 = 544;
			scissor_empty = x1 <= x0 || y1 <= y0;
			// Tile granularity (32x32): exact for the full screen, the usual case
			if (!scissor_empty)
				sceGxmSetRegionClip(ctx, SCE_GXM_REGION_CLIP_OUTSIDE, x0, y0, x1 - 1, y1 - 1);
		}
		if (scissor_empty)
			continue;

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

		if (fp->alpha_ref || fp->env || fp->fog_color) {
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
		}

		if (d->flags & GE_SGX_DRAW_TEXTURED) {
			const SceGxmTexture *t = geSgxTexGet(d->texture);
			if (!t)
				continue;
			SceGxmTexture tex = *t;
			sceGxmTextureSetUAddrMode(&tex, (d->tex_mode & GE_SGX_TEX_CLAMP_U) ? SCE_GXM_TEXTURE_ADDR_CLAMP : SCE_GXM_TEXTURE_ADDR_REPEAT);
			sceGxmTextureSetVAddrMode(&tex, (d->tex_mode & GE_SGX_TEX_CLAMP_V) ? SCE_GXM_TEXTURE_ADDR_CLAMP : SCE_GXM_TEXTURE_ADDR_REPEAT);
			sceGxmTextureSetMinFilter(&tex, (d->tex_mode & GE_SGX_TEX_LINEAR_MIN) ? SCE_GXM_TEXTURE_FILTER_LINEAR : SCE_GXM_TEXTURE_FILTER_POINT);
			sceGxmTextureSetMagFilter(&tex, (d->tex_mode & GE_SGX_TEX_LINEAR_MAG) ? SCE_GXM_TEXTURE_FILTER_LINEAR : SCE_GXM_TEXTURE_FILTER_POINT);
			sceGxmSetFragmentTexture(ctx, 0, &tex);
		}

		if (d->gpu) {
			sceGxmSetVertexUniformBuffer(ctx, 0, frame->uniforms_gpu + d->ucommon);
			sceGxmSetVertexUniformBuffer(ctx, 1, frame->uniforms_gpu + d->uobject);
			sceGxmSetVertexStream(ctx, 0, frame->raw + d->base);
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
	sceGxmSetRegionClip(ctx, SCE_GXM_REGION_CLIP_OUTSIDE, 0, 0, 959, 543);
	sceGxmSetFrontPointLineWidth(ctx, 1);

	stats->render_us = (uint32_t)(sceKernelGetProcessTimeWide() - t0);
}
