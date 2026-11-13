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
	SGX renderer: draw records from the GE state (see ge_sgx_frame.h).

	Stage 1 builds the draws in through mode (2D: menus, HUD, text, 2D
	games): vertices are decoded on the CPU, scaled to 960x544, sprites
	become two triangles, line strips become lines. Transformed (3D) draws
	are counted for now. Register meanings follow PPSSPP (GPU/ge_constants.h,
	GPU/GPUState.h).
*/

#include <string.h>

#include "ge_sgx_walk.h"
#include "ge_sgx_texconv.h"

#define GE_CMD_TEXTUREMAPENABLE 0x1Eu
#define GE_CMD_ALPHABLENDENABLE 0x21u
#define GE_CMD_ALPHATESTENABLE  0x22u
#define GE_CMD_MATERIALAMBIENT  0x55u
#define GE_CMD_MATERIALALPHA    0x58u
#define GE_CMD_FRAMEBUFPTR      0x9Cu
#define GE_CMD_FRAMEBUFWIDTH    0x9Du
#define GE_CMD_TEXADDR0         0xA0u
#define GE_CMD_TEXBUFWIDTH0     0xA8u
#define GE_CMD_CLUTADDR         0xB0u
#define GE_CMD_CLUTADDRUPPER    0xB1u
#define GE_CMD_TRANSFERSRC      0xB2u
#define GE_CMD_TRANSFERSRCW     0xB3u
#define GE_CMD_TRANSFERDST      0xB4u
#define GE_CMD_TRANSFERDSTW     0xB5u
#define GE_CMD_TEXSIZE0         0xB8u
#define GE_CMD_TEXMODE          0xC2u
#define GE_CMD_TEXFORMAT        0xC3u
#define GE_CMD_CLUTFORMAT       0xC5u
#define GE_CMD_TEXFILTER        0xC6u
#define GE_CMD_TEXWRAP          0xC7u
#define GE_CMD_TEXFUNC          0xC9u
#define GE_CMD_TEXENVCOLOR      0xCAu
#define GE_CMD_CLEARMODE        0xD3u
#define GE_CMD_SCISSOR1         0xD4u
#define GE_CMD_SCISSOR2         0xD5u
#define GE_CMD_ALPHATEST        0xDBu
#define GE_CMD_BLENDMODE        0xDFu
#define GE_CMD_BLENDFIXEDA      0xE0u
#define GE_CMD_BLENDFIXEDB      0xE1u
#define GE_CMD_TRANSFERSIZE     0xEEu
#define GE_CMD_VERTEXTYPE       0x12u

#define GE_VTYPE_THROUGH (1u << 23)

enum {
	GE_PRIM_POINTS,
	GE_PRIM_LINES,
	GE_PRIM_LINE_STRIP,
	GE_PRIM_TRIANGLES,
	GE_PRIM_TRIANGLE_STRIP,
	GE_PRIM_TRIANGLE_FAN,
	GE_PRIM_RECTANGLES,
};

// The vertex data the GE reads for one vertex
typedef struct {
	float x, y, z;
	float u, v;
	uint32_t color;
} RawVertex;

static inline uint32_t reg(const GeSgxWalker *w, uint32_t cmd) {
	return w->regs[cmd] & 0x00FFFFFF;
}

static uint32_t expand(uint32_t v, int bits) {
	v &= (1u << bits) - 1;
	return (v << (8 - bits)) | (v >> (2 * bits - 8));
}

uint32_t geSgxDecodeColor16(uint32_t fmt, uint32_t c) {
	uint32_t r, g, b, a;
	switch (fmt) {
		case 0: // 565
			r = expand(c, 5);
			g = expand(c >> 5, 6);
			b = expand(c >> 11, 5);
			a = 0xFF;
			break;
		case 1: // 5551
			r = expand(c, 5);
			g = expand(c >> 5, 5);
			b = expand(c >> 10, 5);
			a = (c & 0x8000) ? 0xFF : 0;
			break;
		default: // 4444
			r = (c & 0xF) * 0x11;
			g = ((c >> 4) & 0xF) * 0x11;
			b = ((c >> 8) & 0xF) * 0x11;
			a = ((c >> 12) & 0xF) * 0x11;
			break;
	}
	return (a << 24) | (b << 16) | (g << 8) | r;
}

static int readVertex(GeSgxWalker *w, const GeSgxVertexLayout *l, uint32_t index, RawVertex *out) {
	uint8_t buf[64];
	uint32_t size = l->size;
	if (size > sizeof(buf))
		size = sizeof(buf); // morph targets: the first one only
	if (!geSgxRead(w, w->vaddr + index * l->size, buf, size))
		return 0;

	// Position: in through mode x and y are pixels, z an unsigned 16 bit depth
	switch (l->pos_fmt) {
		case 1: {
			int8_t p[3];
			memcpy(p, buf + l->pos_off, 3);
			out->x = p[0];
			out->y = p[1];
			out->z = (uint8_t)p[2] / 255.0f;
			break;
		}
		case 2: {
			int16_t p[3];
			memcpy(p, buf + l->pos_off, 6);
			out->x = p[0];
			out->y = p[1];
			out->z = (uint16_t)p[2] / 65535.0f;
			break;
		}
		case 3: {
			float p[3];
			memcpy(p, buf + l->pos_off, 12);
			out->x = p[0];
			out->y = p[1];
			out->z = p[2] / 65535.0f;
			break;
		}
		default:
			return 0;
	}

	// Texture coordinates: texels in through mode
	switch (l->tc_fmt) {
		case 1:
			out->u = buf[l->tc_off];
			out->v = buf[l->tc_off + 1];
			break;
		case 2: {
			uint16_t t[2];
			memcpy(t, buf + l->tc_off, 4);
			out->u = t[0];
			out->v = t[1];
			break;
		}
		case 3: {
			float t[2];
			memcpy(t, buf + l->tc_off, 8);
			out->u = t[0];
			out->v = t[1];
			break;
		}
		default:
			out->u = out->v = 0.0f;
			break;
	}

	// Colour: without one, the material colour
	if (l->color_fmt == 7) {
		memcpy(&out->color, buf + l->color_off, 4);
	} else if (l->color_fmt) {
		uint16_t c;
		memcpy(&c, buf + l->color_off, 2);
		out->color = geSgxDecodeColor16(l->color_fmt - 4, c);
	} else {
		out->color = reg(w, GE_CMD_MATERIALAMBIENT) | ((reg(w, GE_CMD_MATERIALALPHA) & 0xFF) << 24);
	}

	return 1;
}

static int vertexIndex(GeSgxWalker *w, const GeSgxVertexLayout *l, uint32_t i, uint32_t *index) {
	if (!l->index_size) {
		*index = i;
		return 1;
	}
	uint32_t v = 0;
	if (!geSgxRead(w, w->iaddr + i * l->index_size, &v, l->index_size))
		return 0;
	*index = v;
	return 1;
}

static void emit(GeSgxFrameData *f, const RawVertex *r, float tex_w, float tex_h) {
	GeSgxVertex *v = &f->vertices[f->nvertices++];
	v->x = r->x * GE_SGX_SCALE;
	v->y = r->y * GE_SGX_SCALE;
	v->z = r->z;
	v->u = r->u * tex_w;
	v->v = r->v * tex_h;
	v->color = r->color;
}

uint32_t geSgxHash(const void *data, uint32_t size) {
	// FNV-1a
	const uint8_t *p = (const uint8_t *)data;
	uint32_t h = 2166136261u;
	uint32_t i;
	for (i = 0; i < size; i++) {
		h ^= p[i];
		h *= 16777619u;
	}
	return h;
}

static void readClut(GeSgxWalker *w);

static int32_t drawTexture(GeSgxWalker *w) {
	if (!w->texture_fn)
		return -1;

	GeSgxTexDesc d;
	memset(&d, 0, sizeof(d));
	uint32_t bufw = reg(w, GE_CMD_TEXBUFWIDTH0);
	d.addr = (reg(w, GE_CMD_TEXADDR0) & 0x00FFFFF0u) | ((bufw << 8) & 0x0F000000u);
	d.bufw = bufw & 0x7FF;
	uint32_t size = reg(w, GE_CMD_TEXSIZE0);
	d.w = (uint16_t)(1u << (size & 0xF));
	d.h = (uint16_t)(1u << ((size >> 8) & 0xF));
	d.format = (uint8_t)(reg(w, GE_CMD_TEXFORMAT) & 0xF);
	d.swizzle = (uint8_t)(reg(w, GE_CMD_TEXMODE) & 1);
	if (d.format >= 4 && d.format <= 7) {
		readClut(w);
		d.clut_format = reg(w, GE_CMD_CLUTFORMAT);
		d.clut = w->clut;
		d.clut_bytes = w->clut_bytes;
		d.clut_hash = w->clut_hash;
	}
	return w->texture_fn(w->texture_user, &d);
}

static void drawState(GeSgxWalker *w, GeSgxDraw *d) {
	memset(d, 0, sizeof(*d));

	uint32_t clear = reg(w, GE_CMD_CLEARMODE);
	if (clear & 1) {
		d->flags = GE_SGX_DRAW_CLEAR;
		if (clear & 0x100)
			d->flags |= GE_SGX_DRAW_WRITE_COLOR;
		if (clear & 0x200)
			d->flags |= GE_SGX_DRAW_WRITE_ALPHA;
	} else {
		if (reg(w, GE_CMD_TEXTUREMAPENABLE) & 1)
			d->flags |= GE_SGX_DRAW_TEXTURED;
		if (reg(w, GE_CMD_ALPHABLENDENABLE) & 1)
			d->flags |= GE_SGX_DRAW_BLEND;
		if (reg(w, GE_CMD_ALPHATESTENABLE) & 1)
			d->flags |= GE_SGX_DRAW_ALPHATEST;
		uint32_t tfunc = reg(w, GE_CMD_TEXFUNC);
		d->tfunc = (uint8_t)(tfunc & 7);
		if (d->tfunc > 4)
			d->tfunc = 0;
		if (tfunc & 0x100)
			d->flags |= GE_SGX_DRAW_TCC;
		if (tfunc & 0x10000)
			d->flags |= GE_SGX_DRAW_DOUBLE;
		uint32_t at = reg(w, GE_CMD_ALPHATEST);
		d->atest_func = (uint8_t)(at & 7);
		d->atest_ref = (uint8_t)((at >> 8) & 0xFF);
		d->blend = reg(w, GE_CMD_BLENDMODE);
		d->fix_a = reg(w, GE_CMD_BLENDFIXEDA);
		d->fix_b = reg(w, GE_CMD_BLENDFIXEDB);
		d->env = reg(w, GE_CMD_TEXENVCOLOR);
		if (d->flags & GE_SGX_DRAW_TEXTURED) {
			uint32_t wrap = reg(w, GE_CMD_TEXWRAP), filter = reg(w, GE_CMD_TEXFILTER);
			if (wrap & 1)
				d->tex_mode |= GE_SGX_TEX_CLAMP_U;
			if (wrap & 0x100)
				d->tex_mode |= GE_SGX_TEX_CLAMP_V;
			if (filter & 1)
				d->tex_mode |= GE_SGX_TEX_LINEAR_MIN;
			if (filter & 0x100)
				d->tex_mode |= GE_SGX_TEX_LINEAR_MAG;
		}
	}

	uint32_t s1 = reg(w, GE_CMD_SCISSOR1), s2 = reg(w, GE_CMD_SCISSOR2);
	d->scissor[0] = (int16_t)((s1 & 0x3FF) * GE_SGX_SCALE);
	d->scissor[1] = (int16_t)(((s1 >> 10) & 0x3FF) * GE_SGX_SCALE);
	d->scissor[2] = (int16_t)(((s2 & 0x3FF) + 1) * GE_SGX_SCALE);
	d->scissor[3] = (int16_t)((((s2 >> 10) & 0x3FF) + 1) * GE_SGX_SCALE);

	d->fb = (((reg(w, GE_CMD_FRAMEBUFWIDTH) & 0x00FF0000u) << 8) | reg(w, GE_CMD_FRAMEBUFPTR)) & 0x001FFFFFu;
	d->texture = -1;
}

// Same state as the previous draw: one draw for both (batches of sprites)
static int sameState(const GeSgxDraw *a, const GeSgxDraw *b) {
	return a->prim == b->prim && a->flags == b->flags && a->tfunc == b->tfunc &&
		a->atest_func == b->atest_func && a->atest_ref == b->atest_ref && a->tex_mode == b->tex_mode &&
		a->blend == b->blend && a->fix_a == b->fix_a && a->fix_b == b->fix_b &&
		a->env == b->env && a->texture == b->texture && a->fb == b->fb &&
		memcmp(a->scissor, b->scissor, sizeof(a->scissor)) == 0;
}

void geSgxBuildPrim(GeSgxWalker *w, uint32_t type, uint32_t count, const GeSgxVertexLayout *l) {
	GeSgxFrameData *f = w->frame;

	if (!(reg(w, GE_CMD_VERTEXTYPE) & GE_VTYPE_THROUGH)) {
		f->skipped_3d++;
		return;
	}
	if (count == 0)
		return;

	GeSgxDraw d;
	drawState(w, &d);

	uint32_t out;
	switch (type) {
		case GE_PRIM_POINTS:
			d.prim = GE_SGX_DRAW_POINTS;
			out = count;
			break;
		case GE_PRIM_LINES:
			d.prim = GE_SGX_DRAW_LINES;
			out = count & ~1u;
			break;
		case GE_PRIM_LINE_STRIP:
			d.prim = GE_SGX_DRAW_LINES;
			out = count >= 2 ? (count - 1) * 2 : 0;
			break;
		case GE_PRIM_TRIANGLES:
			d.prim = GE_SGX_DRAW_TRIANGLES;
			out = count - count % 3;
			break;
		case GE_PRIM_TRIANGLE_STRIP:
			d.prim = GE_SGX_DRAW_TRIANGLE_STRIP;
			out = count;
			break;
		case GE_PRIM_TRIANGLE_FAN:
			d.prim = GE_SGX_DRAW_TRIANGLE_FAN;
			out = count;
			break;
		case GE_PRIM_RECTANGLES:
			d.prim = GE_SGX_DRAW_TRIANGLES;
			out = (count / 2) * 6;
			break;
		default:
			f->skipped_other++;
			return;
	}
	if (out == 0)
		return;

	if (f->nvertices + out > f->max_vertices || f->ndraws >= f->max_draws || out > GE_SGX_MAX_DRAW_VERTICES) {
		f->skipped_other++;
		return;
	}

	float tex_w = 1.0f, tex_h = 1.0f;
	if (d.flags & GE_SGX_DRAW_TEXTURED) {
		d.texture = drawTexture(w);
		if (d.texture < 0) {
			f->skipped_other++;
			return;
		}
		uint32_t size = reg(w, GE_CMD_TEXSIZE0);
		tex_w = 1.0f / (float)(1u << (size & 0xF));
		tex_h = 1.0f / (float)(1u << ((size >> 8) & 0xF));
	}

	uint32_t first = f->nvertices;
	uint32_t i;
	RawVertex a, b;

	switch (type) {
		case GE_PRIM_RECTANGLES:
			for (i = 0; i + 1 < count; i += 2) {
				uint32_t ia, ib;
				if (!vertexIndex(w, l, i, &ia) || !vertexIndex(w, l, i + 1, &ib) ||
				    !readVertex(w, l, ia, &a) || !readVertex(w, l, ib, &b))
					goto fail;
				// Corners: positions and texture coordinates of the two vertices,
				// colour and depth of the second
				RawVertex c[4];
				c[0] = b; c[0].x = a.x; c[0].y = a.y; c[0].u = a.u; c[0].v = a.v;
				c[1] = b; c[1].x = b.x; c[1].y = a.y; c[1].u = b.u; c[1].v = a.v;
				c[2] = b; c[2].x = a.x; c[2].y = b.y; c[2].u = a.u; c[2].v = b.v;
				c[3] = b;
				emit(f, &c[0], tex_w, tex_h);
				emit(f, &c[1], tex_w, tex_h);
				emit(f, &c[2], tex_w, tex_h);
				emit(f, &c[1], tex_w, tex_h);
				emit(f, &c[3], tex_w, tex_h);
				emit(f, &c[2], tex_w, tex_h);
			}
			break;
		case GE_PRIM_LINE_STRIP:
			for (i = 0; i + 1 < count; i++) {
				uint32_t ia, ib;
				if (!vertexIndex(w, l, i, &ia) || !vertexIndex(w, l, i + 1, &ib) ||
				    !readVertex(w, l, ia, &a) || !readVertex(w, l, ib, &b))
					goto fail;
				emit(f, &a, tex_w, tex_h);
				emit(f, &b, tex_w, tex_h);
			}
			break;
		default:
			for (i = 0; i < out; i++) {
				uint32_t ia;
				if (!vertexIndex(w, l, i, &ia) || !readVertex(w, l, ia, &a))
					goto fail;
				emit(f, &a, tex_w, tex_h);
			}
			break;
	}

	d.first = first;
	d.count = f->nvertices - first;

	// Lists of the same state follow each other: extend the previous draw
	if (f->ndraws > 0 && (d.prim == GE_SGX_DRAW_TRIANGLES || d.prim == GE_SGX_DRAW_LINES || d.prim == GE_SGX_DRAW_POINTS)) {
		GeSgxDraw *p = &f->draws[f->ndraws - 1];
		if (p->first + p->count == first && sameState(p, &d) && p->count + d.count <= GE_SGX_MAX_DRAW_VERTICES) {
			p->count += d.count;
			return;
		}
	}

	f->draws[f->ndraws++] = d;
	return;

fail:
	f->nvertices = first;
	f->skipped_other++;
}

void geSgxBuildLoadClut(GeSgxWalker *w, uint32_t data) {
	uint32_t bytes = (data & 0x3F) * 32;
	if (bytes > GE_SGX_CLUT_BYTES)
		bytes = GE_SGX_CLUT_BYTES;
	w->clut_addr = (reg(w, GE_CMD_CLUTADDR) & 0x00FFFFF0u) | ((reg(w, GE_CMD_CLUTADDRUPPER) << 8) & 0x0F000000u);
	w->clut_load_bytes = bytes;
	w->clut_read = 0;
}

// The CLUT of the last LOADCLUT, read the first time a draw uses it
static void readClut(GeSgxWalker *w) {
	if (w->clut_read)
		return;
	w->clut_read = 1;
	if (w->clut_load_bytes == 0 || !geSgxRead(w, w->clut_addr, w->clut, w->clut_load_bytes)) {
		w->clut_bytes = 0;
		w->clut_hash = 0;
		return;
	}
	w->clut_bytes = w->clut_load_bytes;
	w->clut_hash = geSgxHash(w->clut, w->clut_bytes);
}

void geSgxBuildTransfer(GeSgxWalker *w) {
	if (!w->frame)
		return;

	uint32_t src = ((reg(w, GE_CMD_TRANSFERSRCW) & 0x00FF0000u) << 8) | reg(w, GE_CMD_TRANSFERSRC);
	uint32_t dst = ((reg(w, GE_CMD_TRANSFERDSTW) & 0x00FF0000u) << 8) | reg(w, GE_CMD_TRANSFERDST);
	uint32_t size = reg(w, GE_CMD_TRANSFERSIZE);
	uint32_t tw = (size & 0x3FF) + 1, th = ((size >> 10) & 0x3FF) + 1;

	src &= 0x0FFFFFFF;
	dst &= 0x0FFFFFFF;

	// A copy of (most of) a frame from one VRAM buffer into another
	if (tw * th < 400 * 200 || src < 0x04000000 || src >= 0x04800000 || dst < 0x04000000 || dst >= 0x04800000)
		return;

	if (w->naliases < GE_SGX_MAX_ALIASES) {
		w->aliases[w->naliases].src = src & 0x001FFFFF;
		w->aliases[w->naliases].dst = dst & 0x001FFFFF;
		w->naliases++;
	}
}
