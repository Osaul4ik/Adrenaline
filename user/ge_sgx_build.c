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

	Vertices are decoded and transformed on the CPU (through mode: scaled
	pixels; 3D: morphing, skinning, the matrices, lighting, texture
	coordinate generation and fog), into 960x544 pixels times the clip w.
	Every primitive becomes an indexed triangle, line or point list, so
	primitives of one state share a draw. Register meanings follow PPSSPP
	(GPU/ge_constants.h, GPU/GPUState.h).
*/

#include <math.h>
#include <string.h>

#include "ge_sgx_walk.h"
#include "ge_sgx_texconv.h"
#include "ge_sgx_gpu.h"

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
#define GE_CMD_LIGHTINGENABLE   0x17u
#define GE_CMD_LIGHTENABLE0     0x18u
#define GE_CMD_FOGENABLE        0x1Fu
#define GE_CMD_ZTESTENABLE      0x23u
#define GE_CMD_MORPHWEIGHT0     0x2Cu
#define GE_CMD_VIEWPORTXSCALE   0x42u
#define GE_CMD_VIEWPORTYSCALE   0x43u
#define GE_CMD_VIEWPORTZSCALE   0x44u
#define GE_CMD_VIEWPORTXCENTER  0x45u
#define GE_CMD_VIEWPORTYCENTER  0x46u
#define GE_CMD_VIEWPORTZCENTER  0x47u
#define GE_CMD_TEXSCALEU        0x48u
#define GE_CMD_TEXSCALEV        0x49u
#define GE_CMD_TEXOFFSETU       0x4Au
#define GE_CMD_TEXOFFSETV       0x4Bu
#define GE_CMD_OFFSETX          0x4Cu
#define GE_CMD_OFFSETY          0x4Du
#define GE_CMD_REVERSENORMAL    0x51u
#define GE_CMD_MATERIALUPDATE   0x53u
#define GE_CMD_MATERIALEMISSIVE 0x54u
#define GE_CMD_MATERIALDIFFUSE  0x56u
#define GE_CMD_MATERIALSPECULAR 0x57u
#define GE_CMD_MATERIALSPECULARCOEF 0x5Bu
#define GE_CMD_AMBIENTCOLOR     0x5Cu
#define GE_CMD_AMBIENTALPHA     0x5Du
#define GE_CMD_LIGHTTYPE0       0x5Fu
#define GE_CMD_LX0              0x63u
#define GE_CMD_LDX0             0x6Fu
#define GE_CMD_LKA0             0x7Bu
#define GE_CMD_LKS0             0x87u
#define GE_CMD_LKO0             0x8Bu
#define GE_CMD_LAC0             0x8Fu
#define GE_CMD_LDC0             0x90u
#define GE_CMD_LSC0             0x91u
#define GE_CMD_TEXMAPMODE       0xC0u
#define GE_CMD_TEXSHADELS       0xC1u
#define GE_CMD_FOG1             0xCDu
#define GE_CMD_FOG2             0xCEu
#define GE_CMD_FOGCOLOR         0xCFu
#define GE_CMD_ZTEST            0xDEu
#define GE_CMD_ZWRITEDISABLE    0xE7u

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

/*
	Vertex pipeline (PPSSPP's software transform, GPU/Common/SoftwareTransformCommon.cpp
	and TransformCommon.cpp): decode, morph, skinning, world/view/projection,
	lighting, texture coordinate generation, fog, then the viewport into
	960x544 pixels times w.
*/

typedef struct {
	int through;
	const GeSgxVertexLayout *l;
	uint32_t vtype;

	// Morph weights (morph targets > 1)
	float morph[8];

	// Through mode: texels to normalized
	float tex_w, tex_h;

	// 3D
	int skinning;
	float wv[12];   // view * world (no skinning)
	float proj[16];
	float mvp[16];  // proj * view * world, 4x4 column-major
	const GeSgxMatrices *m;
	float vp_x[2], vp_y[2], vp_z[2]; // pixel = c[0] * clip + c[1] * w
	float uscale, vscale, uoff, voff;
	int uvgen, uvproj;
	float envls[2][3];
	int fog;
	float fog1, fog2;

	// Lighting
	int lighting;
	int matupdate;
	int reverse_normal;
	float mat_emissive[3], mat_ambient[4], mat_diffuse[3], mat_specular[3];
	float global_ambient[4];
	float spec_coef;
	int nlights;
	struct {
		int type, comp;
		float pos[3], dir[3], att[3];
		float spot_exp, spot_cut;
		float ambient[3], diffuse[3], specular[3];
	} light[4];

	uint32_t material_color; // without lighting and vertex colour
} Xform;

static void rgb(uint32_t c, float *out) {
	out[0] = (c & 0xFF) * (1.0f / 255.0f);
	out[1] = ((c >> 8) & 0xFF) * (1.0f / 255.0f);
	out[2] = ((c >> 16) & 0xFF) * (1.0f / 255.0f);
}

static inline float freg(const GeSgxWalker *w, uint32_t cmd) {
	return geSgxFloat24(w->regs[cmd]);
}

// out = a * b, 4x3 matrices (a after b)
static void mul43(float *out, const float *a, const float *b) {
	int j;
	for (j = 0; j < 4; j++) {
		const float *c = &b[j * 3];
		out[j * 3 + 0] = a[0] * c[0] + a[3] * c[1] + a[6] * c[2];
		out[j * 3 + 1] = a[1] * c[0] + a[4] * c[1] + a[7] * c[2];
		out[j * 3 + 2] = a[2] * c[0] + a[5] * c[1] + a[8] * c[2];
	}
	out[9] += a[9];
	out[10] += a[10];
	out[11] += a[11];
}

static inline void xf43(float *out, const float *m, float x, float y, float z) {
	out[0] = x * m[0] + y * m[3] + z * m[6] + m[9];
	out[1] = x * m[1] + y * m[4] + z * m[7] + m[10];
	out[2] = x * m[2] + y * m[5] + z * m[8] + m[11];
}

static inline void xf33(float *out, const float *m, float x, float y, float z) {
	out[0] = x * m[0] + y * m[3] + z * m[6];
	out[1] = x * m[1] + y * m[4] + z * m[7];
	out[2] = x * m[2] + y * m[5] + z * m[8];
}

static inline float invLength(const float *v) {
	float l2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
	return l2 > 0.0f ? 1.0f / sqrtf(l2) : 0.0f;
}

static void normalize3(float *v, float dz) {
	float s = invLength(v);
	if (s == 0.0f) {
		v[0] = v[1] = 0.0f;
		v[2] = dz;
		return;
	}
	v[0] *= s;
	v[1] *= s;
	v[2] *= s;
}

static void xformInit(GeSgxWalker *w, const GeSgxVertexLayout *l, Xform *x) {
	uint32_t vtype = reg(w, GE_CMD_VERTEXTYPE);
	int i;

	x->l = l;
	x->vtype = vtype;
	x->through = (vtype & GE_VTYPE_THROUGH) != 0;
	for (i = 0; i < 8; i++)
		x->morph[i] = freg(w, GE_CMD_MORPHWEIGHT0 + i);
	if (l->morphs == 1)
		x->morph[0] = 1.0f;

	uint32_t mat = reg(w, GE_CMD_MATERIALAMBIENT) | ((reg(w, GE_CMD_MATERIALALPHA) & 0xFF) << 24);
	x->material_color = mat;

	if (x->through)
		return;

	const GeSgxMatrices *m = &w->mtx;
	x->m = m;
	x->skinning = l->wt_fmt != 0;
	mul43(x->wv, m->view, m->world);
	memcpy(x->proj, m->proj, sizeof(x->proj));
	// The 4x3 wv as 4x4 (last row 0 0 0 1), times proj
	for (i = 0; i < 4; i++) {
		int r;
		for (r = 0; r < 4; r++) {
			const float *P = x->proj;
			float v = P[r] * x->wv[i * 3] + P[4 + r] * x->wv[i * 3 + 1] + P[8 + r] * x->wv[i * 3 + 2];
			if (i == 3)
				v += P[12 + r];
			x->mvp[i * 4 + r] = v;
		}
	}

	// screen = center + scale * clip / w - offset, in PSP pixels; times w and the output scale
	float offx = (reg(w, GE_CMD_OFFSETX) & 0xFFFF) * (1.0f / 16.0f);
	float offy = (reg(w, GE_CMD_OFFSETY) & 0xFFFF) * (1.0f / 16.0f);
	x->vp_x[0] = freg(w, GE_CMD_VIEWPORTXSCALE) * GE_SGX_SCALE;
	x->vp_x[1] = (freg(w, GE_CMD_VIEWPORTXCENTER) - offx) * GE_SGX_SCALE;
	x->vp_y[0] = freg(w, GE_CMD_VIEWPORTYSCALE) * GE_SGX_SCALE;
	x->vp_y[1] = (freg(w, GE_CMD_VIEWPORTYCENTER) - offy) * GE_SGX_SCALE;
	x->vp_z[0] = freg(w, GE_CMD_VIEWPORTZSCALE) * (1.0f / 65535.0f);
	x->vp_z[1] = freg(w, GE_CMD_VIEWPORTZCENTER) * (1.0f / 65535.0f);

	x->uscale = freg(w, GE_CMD_TEXSCALEU);
	x->vscale = freg(w, GE_CMD_TEXSCALEV);
	x->uoff = freg(w, GE_CMD_TEXOFFSETU);
	x->voff = freg(w, GE_CMD_TEXOFFSETV);
	uint32_t texmap = reg(w, GE_CMD_TEXMAPMODE);
	x->uvgen = texmap & 3;
	x->uvproj = (texmap >> 8) & 3;
	if (x->uvgen == 2) {
		uint32_t ls = reg(w, GE_CMD_TEXSHADELS);
		for (i = 0; i < 2; i++) {
			int n = (ls >> (i * 8)) & 3;
			x->envls[i][0] = freg(w, GE_CMD_LX0 + n * 3);
			x->envls[i][1] = freg(w, GE_CMD_LX0 + n * 3 + 1);
			x->envls[i][2] = freg(w, GE_CMD_LX0 + n * 3 + 2);
			normalize3(x->envls[i], 1.0f);
		}
	}

	x->fog = (reg(w, GE_CMD_FOGENABLE) & 1) && !(reg(w, GE_CMD_CLEARMODE) & 1);
	x->fog1 = freg(w, GE_CMD_FOG1);
	x->fog2 = freg(w, GE_CMD_FOG2);

	x->lighting = (reg(w, GE_CMD_LIGHTINGENABLE) & 1) && !(reg(w, GE_CMD_CLEARMODE) & 1);
	if (!x->lighting)
		return;

	x->matupdate = l->color_fmt ? (int)(reg(w, GE_CMD_MATERIALUPDATE) & 7) : 0;
	x->reverse_normal = reg(w, GE_CMD_REVERSENORMAL) & 1;
	rgb(reg(w, GE_CMD_MATERIALEMISSIVE), x->mat_emissive);
	rgb(reg(w, GE_CMD_MATERIALAMBIENT), x->mat_ambient);
	x->mat_ambient[3] = (reg(w, GE_CMD_MATERIALALPHA) & 0xFF) * (1.0f / 255.0f);
	rgb(reg(w, GE_CMD_MATERIALDIFFUSE), x->mat_diffuse);
	rgb(reg(w, GE_CMD_MATERIALSPECULAR), x->mat_specular);
	rgb(reg(w, GE_CMD_AMBIENTCOLOR), x->global_ambient);
	x->global_ambient[3] = (reg(w, GE_CMD_AMBIENTALPHA) & 0xFF) * (1.0f / 255.0f);
	x->spec_coef = freg(w, GE_CMD_MATERIALSPECULARCOEF);

	x->nlights = 0;
	for (i = 0; i < 4; i++) {
		if (!(reg(w, GE_CMD_LIGHTENABLE0 + i) & 1))
			continue;
		int n = x->nlights++;
		uint32_t t = reg(w, GE_CMD_LIGHTTYPE0 + i);
		x->light[n].comp = t & 3;
		x->light[n].type = (t >> 8) & 3;
		int k;
		for (k = 0; k < 3; k++) {
			x->light[n].pos[k] = freg(w, GE_CMD_LX0 + i * 3 + k);
			x->light[n].dir[k] = freg(w, GE_CMD_LDX0 + i * 3 + k);
			x->light[n].att[k] = freg(w, GE_CMD_LKA0 + i * 3 + k);
		}
		if (x->light[n].type == 0)
			normalize3(x->light[n].pos, 1.0f);
		normalize3(x->light[n].dir, 1.0f);
		x->light[n].spot_exp = freg(w, GE_CMD_LKS0 + i);
		x->light[n].spot_cut = freg(w, GE_CMD_LKO0 + i);
		rgb(reg(w, GE_CMD_LAC0 + i * 3), x->light[n].ambient);
		rgb(reg(w, GE_CMD_LDC0 + i * 3), x->light[n].diffuse);
		rgb(reg(w, GE_CMD_LSC0 + i * 3), x->light[n].specular);
	}
}

static inline float clamp01(float v) {
	return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

/*
	x^e for x in 0..1 (lighting exponents): 2^(e * log2 x) with short
	polynomials (about 1e-4 relative), much cheaper than powf.
*/
static inline float fastPow(float x, float e) {
	if (x <= 0.0f)
		return e == 0.0f ? 1.0f : 0.0f;
	if (e == 0.0f)
		return 1.0f;
	union { float f; int32_t i; } u = { x };
	float lg = (float)((u.i >> 23) & 0xFF) - 127.0f;
	u.i = (u.i & 0x007FFFFF) | 0x3F800000; // mantissa 1..2
	float m = u.f;
	if (m > 1.41421356f) { // around 1 for the series: m in 0.707..1.414
		m *= 0.5f;
		lg += 1.0f;
	}
	float t = (m - 1.0f) / (m + 1.0f), t2 = t * t;
	lg += 2.88539008f * t * (1.0f + t2 * (0.33333333f + t2 * (0.2f + t2 * 0.14285714f)));
	float p = e * lg;
	if (p < -126.0f)
		return 0.0f;
	if (p > 0.0f)
		p = 0.0f; // x <= 1, e >= 0
	float fl = (float)(int32_t)p;
	if (fl > p)
		fl -= 1.0f;
	float fr = p - fl; // 0..1
	union { float f; int32_t i; } r;
	r.f = 1.0f + fr * (0.69314718f + fr * (0.24022651f + fr * (0.05550411f + fr * (0.00961813f + fr * 0.00133336f))));
	r.i += (int32_t)fl << 23;
	return r.f;
}

// Colour of a lit vertex (PPSSPP Lighter::Light); specular is added to the primary colour
static void light(const Xform *x, const float *in, const float *pos, const float *nrm, float *out) {
	const float *amb = (x->matupdate & 1) ? in : x->mat_ambient;
	const float *dif = (x->matupdate & 2) ? in : x->mat_diffuse;
	const float *spc = (x->matupdate & 4) ? in : x->mat_specular;
	float sum[4], spec[3] = { 0.0f, 0.0f, 0.0f };
	int i, k;

	for (k = 0; k < 3; k++)
		sum[k] = x->global_ambient[k] * amb[k] + x->mat_emissive[k];
	sum[3] = x->global_ambient[3] * amb[3];

	for (i = 0; i < x->nlights; i++) {
		const typeof(x->light[0]) *lt = &x->light[i];
		float to[3];
		float dist = 0.0f;
		if (lt->type == 0) {
			to[0] = lt->pos[0];
			to[1] = lt->pos[1];
			to[2] = lt->pos[2];
			dist = 1.0f;
		} else {
			to[0] = lt->pos[0] - pos[0];
			to[1] = lt->pos[1] - pos[1];
			to[2] = lt->pos[2] - pos[2];
			float s = invLength(to);
			if (s > 0.0f) {
				dist = 1.0f / s;
				to[0] *= s;
				to[1] *= s;
				to[2] *= s;
			}
		}

		float dot = to[0] * nrm[0] + to[1] * nrm[1] + to[2] * nrm[2];
		if (dot < 0.0f)
			dot = 0.0f;
		if (lt->comp == 2)
			dot = fastPow(dot, x->spec_coef);

		float scale = 1.0f;
		if (lt->type != 0) {
			float d = lt->att[0] + lt->att[1] * dist + lt->att[2] * dist * dist;
			scale = d > 0.0f ? clamp01(1.0f / d) : 1.0f;
			if (lt->type >= 2) {
				float angle = to[0] * lt->dir[0] + to[1] * lt->dir[1] + to[2] * lt->dir[2];
				scale = angle >= lt->spot_cut ? scale * fastPow(angle, lt->spot_exp) : 0.0f;
			}
		}
		if (scale == 0.0f)
			continue;

		for (k = 0; k < 3; k++)
			sum[k] += (lt->ambient[k] * amb[k] + lt->diffuse[k] * dif[k] * dot) * scale;

		if (lt->comp == 1) {
			float h[3] = { to[0], to[1], to[2] + 1.0f }; // the PSP's viewer is (0, 0, 1)
			normalize3(h, 1.0f);
			float sd = h[0] * nrm[0] + h[1] * nrm[1] + h[2] * nrm[2];
			if (sd > 0.0f) {
				float p = fastPow(sd > 1.0f ? 1.0f : sd, x->spec_coef) * scale;
				for (k = 0; k < 3; k++)
					spec[k] += lt->specular[k] * spc[k] * p;
			}
		}
	}

	for (k = 0; k < 3; k++)
		out[k] = clamp01(sum[k] + spec[k]);
	out[3] = clamp01(sum[3]);
}

static inline float comp(const uint8_t *p, uint32_t fmt, int i, int is_signed) {
	switch (fmt) {
		case 1: return is_signed ? ((int8_t)p[i]) * (1.0f / 128.0f) : p[i] * (1.0f / 128.0f);
		case 2: {
			uint16_t v;
			memcpy(&v, p + i * 2, 2);
			return is_signed ? ((int16_t)v) * (1.0f / 32768.0f) : v * (1.0f / 32768.0f);
		}
		default: {
			float f;
			memcpy(&f, p + i * 4, 4);
			return f;
		}
	}
}

static void decodeColor(const GeSgxVertexLayout *l, const uint8_t *p, float *c) {
	uint32_t v;
	if (l->color_fmt == 7) {
		memcpy(&v, p + l->color_off, 4);
	} else {
		uint16_t s;
		memcpy(&s, p + l->color_off, 2);
		v = geSgxDecodeColor16(l->color_fmt - 4, s);
	}
	c[0] = (v & 0xFF) * (1.0f / 255.0f);
	c[1] = ((v >> 8) & 0xFF) * (1.0f / 255.0f);
	c[2] = ((v >> 16) & 0xFF) * (1.0f / 255.0f);
	c[3] = (v >> 24) * (1.0f / 255.0f);
}

static inline uint32_t packColor(const float *c) {
	return (uint32_t)(clamp01(c[0]) * 255.0f + 0.5f) | ((uint32_t)(clamp01(c[1]) * 255.0f + 0.5f) << 8) |
		((uint32_t)(clamp01(c[2]) * 255.0f + 0.5f) << 16) | ((uint32_t)(clamp01(c[3]) * 255.0f + 0.5f) << 24);
}

#define MAX_VERTEX_BYTES 512

/*
	Vertex data of the current primitive: its whole range copied in one go
	(one DMA for big ones), or vertex by vertex for scattered indices.
*/
#define BULK_BYTES (256 * 1024)
static uint8_t bulk[BULK_BYTES];
static uint32_t bulk_first, bulk_count;

static int bulkLoad(GeSgxWalker *w, const GeSgxVertexLayout *l, uint32_t first, uint32_t count) {
	bulk_count = 0;
	uint32_t bytes = count * l->size;
	if (bytes > BULK_BYTES)
		return 0;
	uint32_t addr = w->vaddr + first * l->size;
	if (!(w->bulk_fn && bytes >= 8 * 1024 && w->bulk_fn(addr, bulk, bytes)) && !geSgxRead(w, addr, bulk, bytes))
		return 0;
	bulk_first = first;
	bulk_count = count;
	return 1;
}

static const uint8_t *vertexData(GeSgxWalker *w, const GeSgxVertexLayout *l, uint32_t index, uint8_t *tmp) {
	if (index - bulk_first < bulk_count)
		return bulk + (index - bulk_first) * l->size;
	if (l->size > MAX_VERTEX_BYTES || !geSgxRead(w, w->vaddr + index * l->size, tmp, l->size))
		return NULL;
	return tmp;
}

// One vertex of the current primitive, ready for the GPU
static int vertexOut(GeSgxWalker *w, const Xform *x, uint32_t index, GeSgxVertex *out) {
	const GeSgxVertexLayout *l = x->l;
	uint8_t tmp[MAX_VERTEX_BYTES];
	const uint8_t *buf = vertexData(w, l, index, tmp);
	if (!buf)
		return 0;

	if (x->through) {
		// Pixels, an unsigned 16 bit depth, texels; no morphing
		const uint8_t *p = buf + l->pos_off;
		float px, py, pz;
		switch (l->pos_fmt) {
			case 1: px = (int8_t)p[0]; py = (int8_t)p[1]; pz = p[2] * (1.0f / 255.0f); break;
			case 2: {
				int16_t s[3];
				memcpy(s, p, 6);
				px = s[0]; py = s[1]; pz = (uint16_t)s[2] * (1.0f / 65535.0f);
				break;
			}
			default: {
				float f[3];
				memcpy(f, p, 12);
				px = f[0]; py = f[1]; pz = f[2] * (1.0f / 65535.0f);
				break;
			}
		}
		out->x = px * GE_SGX_SCALE;
		out->y = py * GE_SGX_SCALE;
		out->z = pz;
		out->w = 1.0f;
		if (l->tc_fmt) {
			const uint8_t *t = buf + l->tc_off;
			float u, v;
			switch (l->tc_fmt) {
				case 1: u = t[0]; v = t[1]; break;
				case 2: {
					uint16_t s[2];
					memcpy(s, t, 4);
					u = s[0]; v = s[1];
					break;
				}
				default: memcpy(&u, t, 4); memcpy(&v, t + 4, 4); break;
			}
			out->u = u * x->tex_w;
			out->v = v * x->tex_h;
		} else {
			out->u = out->v = 0.0f;
		}
		if (l->color_fmt) {
			float c[4];
			decodeColor(l, buf, c);
			out->color = packColor(c);
		} else {
			out->color = x->material_color;
		}
		out->fog = 1.0f;
		return 1;
	}

	// Morph targets blended by the morph weights
	float pos[3] = { 0, 0, 0 }, nrm[3] = { 0, 0, 0 }, tc[2] = { 0, 0 }, col[4] = { 0, 0, 0, 0 };
	float wts[8] = { 0 };
	uint32_t k, j;
	for (k = 0; k < l->morphs; k++) {
		const uint8_t *p = buf + k * l->stride;
		float mw = x->morph[k];
		if (mw == 0.0f && l->morphs > 1)
			continue;
		for (j = 0; j < 3; j++)
			pos[j] += comp(p + l->pos_off, l->pos_fmt, j, 1) * mw;
		if (l->nrm_fmt) {
			for (j = 0; j < 3; j++)
				nrm[j] += comp(p + l->nrm_off, l->nrm_fmt, j, 1) * mw;
		}
		if (l->tc_fmt) {
			tc[0] += comp(p + l->tc_off, l->tc_fmt, 0, 0) * mw;
			tc[1] += comp(p + l->tc_off, l->tc_fmt, 1, 0) * mw;
		}
		if (l->color_fmt) {
			float c[4];
			decodeColor(l, p, c);
			for (j = 0; j < 4; j++)
				col[j] += c[j] * mw;
		}
		if (k == 0 && l->wt_fmt) {
			for (j = 0; j < l->weights; j++)
				wts[j] = comp(p, l->wt_fmt, j, 0);
		}
	}
	if (!l->nrm_fmt) {
		nrm[2] = 1.0f;
	}
	if (x->lighting && x->reverse_normal) {
		nrm[0] = -nrm[0];
		nrm[1] = -nrm[1];
		nrm[2] = -nrm[2];
	}

	// Skinning: the bone matrices weighted, in model space
	float mpos[3], mnrm[3];
	if (x->skinning) {
		mpos[0] = mpos[1] = mpos[2] = 0.0f;
		mnrm[0] = mnrm[1] = mnrm[2] = 0.0f;
		for (j = 0; j < l->weights; j++) {
			float wt = wts[j];
			if (wt == 0.0f)
				continue;
			float t[3];
			const float *b = &x->m->bone[j * 12];
			xf43(t, b, pos[0], pos[1], pos[2]);
			mpos[0] += t[0] * wt; mpos[1] += t[1] * wt; mpos[2] += t[2] * wt;
			xf33(t, b, nrm[0], nrm[1], nrm[2]);
			mnrm[0] += t[0] * wt; mnrm[1] += t[1] * wt; mnrm[2] += t[2] * wt;
		}
	} else {
		memcpy(mpos, pos, sizeof(mpos));
		memcpy(mnrm, nrm, sizeof(mnrm));
	}

	float view[3] = { 0, 0, 0 };
	float wpos[3], wnrm[3];
	int need_world = x->lighting || x->uvgen == 2;
	float cx, cy, cz, cw;
	if (need_world) {
		xf43(wpos, x->m->world, mpos[0], mpos[1], mpos[2]);
		xf33(wnrm, x->m->world, mnrm[0], mnrm[1], mnrm[2]);
		normalize3(wnrm, 1.0f);
		xf43(view, x->m->view, wpos[0], wpos[1], wpos[2]);
		const float *P = x->proj;
		cx = view[0] * P[0] + view[1] * P[4] + view[2] * P[8] + P[12];
		cy = view[0] * P[1] + view[1] * P[5] + view[2] * P[9] + P[13];
		cz = view[0] * P[2] + view[1] * P[6] + view[2] * P[10] + P[14];
		cw = view[0] * P[3] + view[1] * P[7] + view[2] * P[11] + P[15];
	} else {
		// One 4x4 transform; the view depth only for the fog
		const float *M = x->mvp;
		cx = mpos[0] * M[0] + mpos[1] * M[4] + mpos[2] * M[8] + M[12];
		cy = mpos[0] * M[1] + mpos[1] * M[5] + mpos[2] * M[9] + M[13];
		cz = mpos[0] * M[2] + mpos[1] * M[6] + mpos[2] * M[10] + M[14];
		cw = mpos[0] * M[3] + mpos[1] * M[7] + mpos[2] * M[11] + M[15];
		if (x->fog)
			view[2] = mpos[0] * x->wv[2] + mpos[1] * x->wv[5] + mpos[2] * x->wv[8] + x->wv[11];
	}

	out->x = x->vp_x[0] * cx + x->vp_x[1] * cw;
	out->y = x->vp_y[0] * cy + x->vp_y[1] * cw;
	out->z = x->vp_z[0] * cz + x->vp_z[1] * cw;
	out->w = cw;
	// The PSP clamps the depth instead of clipping at the far plane (and only
	// clips at the near one): keep z inside 0..w, the GPU then clips at w = 0
	if (cw > 0.0f) {
		if (out->z < 0.0f)
			out->z = 0.0f;
		else if (out->z > cw)
			out->z = cw;
	}

	// Texture coordinates
	switch (x->uvgen) {
		case 1: { // texture matrix
			float src[3];
			switch (x->uvproj) {
				case 0: memcpy(src, mpos, sizeof(src)); break;
				case 1: src[0] = tc[0]; src[1] = tc[1]; src[2] = 0.0f; break;
				case 2: memcpy(src, mnrm, sizeof(src)); normalize3(src, 1.0f); break;
				default: memcpy(src, mnrm, sizeof(src)); break;
			}
			float uvw[3];
			xf43(uvw, x->m->tgen, src[0], src[1], src[2]);
			float q = (uvw[2] > 1e-6f || uvw[2] < -1e-6f) ? 1.0f / uvw[2] : 1.0f;
			out->u = uvw[0] * q;
			out->v = uvw[1] * q;
			break;
		}
		case 2: // environment map
			out->u = (1.0f + x->envls[0][0] * wnrm[0] + x->envls[0][1] * wnrm[1] + x->envls[0][2] * wnrm[2]) * 0.5f;
			out->v = (1.0f + x->envls[1][0] * wnrm[0] + x->envls[1][1] * wnrm[1] + x->envls[1][2] * wnrm[2]) * 0.5f;
			break;
		default:
			out->u = tc[0] * x->uscale + x->uoff;
			out->v = tc[1] * x->vscale + x->voff;
			break;
	}

	// Colour
	float c[4];
	if (l->color_fmt) {
		memcpy(c, col, sizeof(c));
	} else {
		c[0] = (x->material_color & 0xFF) * (1.0f / 255.0f);
		c[1] = ((x->material_color >> 8) & 0xFF) * (1.0f / 255.0f);
		c[2] = ((x->material_color >> 16) & 0xFF) * (1.0f / 255.0f);
		c[3] = (x->material_color >> 24) * (1.0f / 255.0f);
	}
	if (x->lighting) {
		float lit[4];
		light(x, c, wpos, wnrm, lit);
		out->color = packColor(lit);
	} else {
		out->color = l->color_fmt ? packColor(c) : x->material_color;
	}

	// Fog from the view space depth
	if (x->fog) {
		float f = (view[2] + x->fog1) * x->fog2;
		out->fog = (f == f) ? clamp01(f) : 1.0f;
	} else {
		out->fog = 1.0f;
	}
	return 1;
}

static void drawState(GeSgxWalker *w, GeSgxDraw *d) {
	memset(d, 0, sizeof(*d));
	d->ztest = 1;

	uint32_t clear = reg(w, GE_CMD_CLEARMODE);
	if (clear & 1) {
		d->flags = GE_SGX_DRAW_CLEAR;
		if (clear & 0x100)
			d->flags |= GE_SGX_DRAW_WRITE_COLOR;
		if (clear & 0x200)
			d->flags |= GE_SGX_DRAW_WRITE_ALPHA;
		if (clear & 0x400)
			d->flags |= GE_SGX_DRAW_WRITE_DEPTH;
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
		// Depth: tested and written only with the test on (PPSSPP)
		if (reg(w, GE_CMD_ZTESTENABLE) & 1) {
			d->ztest = (uint8_t)(reg(w, GE_CMD_ZTEST) & 7);
			if (!(reg(w, GE_CMD_ZWRITEDISABLE) & 1))
				d->flags |= GE_SGX_DRAW_WRITE_DEPTH;
		}
		if (!(reg(w, GE_CMD_VERTEXTYPE) & GE_VTYPE_THROUGH) && (reg(w, GE_CMD_FOGENABLE) & 1)) {
			d->flags |= GE_SGX_DRAW_FOG;
			d->fog_color = reg(w, GE_CMD_FOGCOLOR);
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

// Same state as the previous draw: one draw for both
static int sameState(const GeSgxDraw *a, const GeSgxDraw *b) {
	return a->prim == b->prim && a->flags == b->flags && a->tfunc == b->tfunc &&
		a->atest_func == b->atest_func && a->atest_ref == b->atest_ref && a->tex_mode == b->tex_mode &&
		a->ztest == b->ztest && a->blend == b->blend && a->fix_a == b->fix_a && a->fix_b == b->fix_b &&
		a->env == b->env && a->fog_color == b->fog_color && a->texture == b->texture && a->fb == b->fb &&
		memcmp(a->scissor, b->scissor, sizeof(a->scissor)) == 0;
}

// Indices of the primitive (up to 65535, 4 bytes each), read once
static uint32_t prim_index[65536];

/* ---- GPU transform ---- */

// Biggest vertex range one primitive copies to the GPU (bigger: CPU transform)
#define GPU_MAX_RANGE_BYTES (256 * 1024)

// Vertex shapes the vertex programs read as they are
static int gpuLayout(const GeSgxWalker *w, const GeSgxVertexLayout *l) {
	if (!(l->pos_fmt && l->morphs == 1 && l->index_size <= 2))
		return 0;
	uint32_t attrs = reg(w, GE_CMD_VERTEXTYPE) & ((3u << 0) | (7u << 2) | (3u << 5) | (3u << 7) | (3u << 9) | (7u << 14));
	int i;
	for (i = 0; i < w->ngpu_bad; i++) {
		if (w->gpu_bad[i] == attrs)
			return 0;
	}
	return 1;
}

static uint32_t gpuVertexKey(const GeSgxVertexLayout *l, const Xform *x) {
	uint32_t k = GE_SGX_VS_BASE;
	if (l->tc_fmt)
		k |= GE_SGX_VS_TC;
	if (l->color_fmt == 7)
		k |= GE_SGX_VS_COLOR;
	else if (l->color_fmt)
		k |= GE_SGX_VS_COLOR16;
	if (l->nrm_fmt)
		k |= GE_SGX_VS_NORMAL;
	if (l->wt_fmt)
		k |= l->weights > 4 ? GE_SGX_VS_W8 : GE_SGX_VS_W4;
	if (x->lighting)
		k |= GE_SGX_VS_LIGHT;
	return k;
}

/*
	The uniform buffers of the current state, built once per state and kept
	in the frame's uniform arena; a block equal to the previous one is
	shared (objects drawn one after the other with the same camera and lights).
*/
static int uni_valid = 0;
static const GeSgxFrameData *uni_frame = NULL;
static uint32_t gpu_vkey, gpu_ucommon, gpu_uobject;
static float last_common[GE_SGX_UC_COUNT][4], last_object[GE_SGX_UO_COUNT][4];
static int have_common, have_object;
static uint32_t last_common_off, last_object_off;

static void set4(float *d, float a, float b, float c, float e) {
	d[0] = a; d[1] = b; d[2] = c; d[3] = e;
}

static void col4(float *d, const float *m, int j) {
	set4(d, m[j * 3], m[j * 3 + 1], m[j * 3 + 2], j == 3 ? 1.0f : 0.0f);
}

static void rgb4(float *d, const float *c, float w) {
	set4(d, c[0], c[1], c[2], w);
}

static float compScale(uint32_t fmt) {
	return fmt == 1 ? 1.0f / 128.0f : fmt == 2 ? 1.0f / 32768.0f : 1.0f;
}

static int putBlock(GeSgxFrameData *f, const void *data, uint32_t bytes, uint32_t *off) {
	uint32_t o = (f->nuniforms + 15u) & ~15u;
	if (o + bytes > f->max_uniforms)
		return 0;
	memcpy(f->uniforms + o, data, bytes);
	f->nuniforms = o + bytes;
	*off = o;
	return 1;
}

static int gpuUniforms(GeSgxWalker *w, GeSgxFrameData *f, const GeSgxVertexLayout *l, const Xform *x) {
	if (uni_frame != f) {
		uni_frame = f;
		uni_valid = 0;
		have_common = have_object = 0;
	}
	if (uni_valid)
		return 1;

	const GeSgxMatrices *m = &w->mtx;
	float c[GE_SGX_UC_COUNT][4], o[GE_SGX_UO_COUNT][4];
	int i;
	memset(c, 0, sizeof(c));
	memset(o, 0, sizeof(o));

	for (i = 0; i < 4; i++) {
		set4(c[GE_SGX_UC_PROJ + i], m->proj[i * 4], m->proj[i * 4 + 1], m->proj[i * 4 + 2], m->proj[i * 4 + 3]);
		col4(c[GE_SGX_UC_VIEW + i], m->view, i);
		col4(c[GE_SGX_UC_TGEN + i], m->tgen, i);
		col4(o[GE_SGX_UO_WORLD + i], m->world, i);
	}
	set4(c[GE_SGX_UC_VP], x->vp_x[0], x->vp_x[1], x->vp_y[0], x->vp_y[1]);
	set4(c[GE_SGX_UC_VPZ], x->vp_z[0], x->vp_z[1], x->fog1, x->fog2);
	set4(c[GE_SGX_UC_FLAGS], (float)x->uvgen, (float)x->uvproj, x->fog ? 1.0f : 0.0f, (x->lighting && x->reverse_normal) ? -1.0f : 1.0f);
	uint32_t mc = x->material_color;
	set4(c[GE_SGX_UC_MATAMB], (mc & 0xFF) / 255.0f, ((mc >> 8) & 0xFF) / 255.0f, ((mc >> 16) & 0xFF) / 255.0f, (mc >> 24) / 255.0f);
	for (i = 0; i < 2; i++)
		rgb4(c[GE_SGX_UC_ENVLS + i], x->envls[i], 0.0f);
	for (i = 0; i < 4; i++)
		c[GE_SGX_UC_LIGHT + i * GE_SGX_UC_LIGHT_SIZE][3] = -1.0f; // off
	if (l->color_fmt >= 4 && l->color_fmt <= 6)
		c[GE_SGX_UC_MATUPD][1] = (float)(l->color_fmt - 4);
	if (x->lighting) {
		rgb4(c[GE_SGX_UC_EMISSIVE], x->mat_emissive, x->spec_coef);
		rgb4(c[GE_SGX_UC_MATDIF], x->mat_diffuse, (x->matupdate & 1) ? 1.0f : 0.0f);
		rgb4(c[GE_SGX_UC_MATSPC], x->mat_specular, (x->matupdate & 2) ? 1.0f : 0.0f);
		set4(c[GE_SGX_UC_AMBIENT], x->global_ambient[0], x->global_ambient[1], x->global_ambient[2], x->global_ambient[3]);
		c[GE_SGX_UC_MATUPD][0] = (x->matupdate & 4) ? 1.0f : 0.0f;
		for (i = 0; i < x->nlights; i++) {
			float (*L)[4] = &c[GE_SGX_UC_LIGHT + i * GE_SGX_UC_LIGHT_SIZE];
			rgb4(L[0], x->light[i].pos, (float)x->light[i].type);
			rgb4(L[1], x->light[i].dir, (float)x->light[i].comp);
			rgb4(L[2], x->light[i].att, x->light[i].spot_exp);
			set4(L[3], x->light[i].spot_cut, 0.0f, 0.0f, 0.0f);
			rgb4(L[4], x->light[i].ambient, 0.0f);
			rgb4(L[5], x->light[i].diffuse, 0.0f);
			rgb4(L[6], x->light[i].specular, 0.0f);
		}
	}

	set4(o[GE_SGX_UO_SCALE], compScale(l->pos_fmt), compScale(l->nrm_fmt), compScale(l->tc_fmt), compScale(l->wt_fmt));
	set4(o[GE_SGX_UO_UV], x->uscale, x->vscale, x->uoff, x->voff);
	if (l->wt_fmt) {
		for (i = 0; i < 8; i++)
			o[GE_SGX_UO_WMASK + i / 4][i & 3] = (uint32_t)i < l->weights ? 1.0f : 0.0f;
		for (i = 0; i < 8 * 4; i++)
			col4(o[GE_SGX_UO_BONES + i], &m->bone[(i / 4) * 12], i & 3);
	}

	if (!(have_common && memcmp(c, last_common, sizeof(c)) == 0)) {
		if (!putBlock(f, c, sizeof(c), &last_common_off))
			return 0;
		memcpy(last_common, c, sizeof(c));
		have_common = 1;
	}
	if (!(have_object && memcmp(o, last_object, sizeof(o)) == 0)) {
		if (!putBlock(f, o, sizeof(o), &last_object_off))
			return 0;
		memcpy(last_object, o, sizeof(o));
		have_object = 1;
	}
	gpu_ucommon = last_common_off;
	gpu_uobject = last_object_off;
	gpu_vkey = gpuVertexKey(l, x);
	uni_valid = 1;
	return 1;
}

/*
	The raw arena: vertex data copied from PSP memory. Primitives usually
	use consecutive memory, so their ranges are collected into one pending
	run and copied together (one DMA when big); ranges copied before in this
	frame are found again. dst keeps the source's alignment modulo 16.
*/
#define RAW_DONE 16
static GeSgxFrameData *raw_frame = NULL;
static struct { uint32_t src, len, dst; } raw_done[RAW_DONE];
static int raw_ndone = 0, raw_next = 0;
static uint32_t pend_src, pend_len, pend_dst;

static void rawFlush(GeSgxWalker *w) {
	if (!pend_len || !raw_frame)
		return;
	uint8_t *dst = raw_frame->raw + pend_dst;
	if (!(w->bulk_fn && pend_len >= 8 * 1024 && w->bulk_fn(pend_src, dst, pend_len)))
		geSgxRead(w, pend_src, dst, pend_len);
	raw_done[raw_next].src = pend_src;
	raw_done[raw_next].len = pend_len;
	raw_done[raw_next].dst = pend_dst;
	raw_next = (raw_next + 1) % RAW_DONE;
	if (raw_ndone < RAW_DONE)
		raw_ndone++;
	pend_len = 0;
}

static int rawCopy(GeSgxWalker *w, GeSgxFrameData *f, uint32_t src, uint32_t bytes, uint32_t *off) {
	int i;
	if (raw_frame != f) {
		raw_frame = f;
		raw_ndone = raw_next = 0;
		pend_len = 0;
	}
	if (pend_len && src >= pend_src && src + bytes <= pend_src + pend_len) {
		*off = pend_dst + (src - pend_src);
		return 1;
	}
	// Continues the pending run (it ends the arena)
	if (pend_len && src >= pend_src && src <= pend_src + pend_len && pend_dst + pend_len == f->nraw) {
		uint32_t len = src + bytes - pend_src;
		if (pend_dst + len > f->max_raw)
			return 0;
		pend_len = len;
		f->nraw = pend_dst + len;
		*off = pend_dst + (src - pend_src);
		return 1;
	}
	for (i = 0; i < raw_ndone; i++) {
		if (src >= raw_done[i].src && src + bytes <= raw_done[i].src + raw_done[i].len) {
			*off = raw_done[i].dst + (src - raw_done[i].src);
			return 1;
		}
	}
	rawFlush(w);
	uint32_t dst = ((f->nraw + 15u) & ~15u) + (src & 15u);
	if (dst + bytes > f->max_raw)
		return 0;
	pend_src = src;
	pend_len = bytes;
	pend_dst = dst;
	f->nraw = dst + bytes;
	*off = dst;
	return 1;
}

/*
	Scattered indices: each vertex they use once, appended to the raw
	arena; prim_index becomes local. The stamp tells this primitive's
	vertices from older ones without clearing the map.
*/
static uint16_t gather_map[65536];
static uint32_t gather_stamp[65536];
static uint32_t gather_cur = 0;

static int rawGather(GeSgxWalker *w, GeSgxFrameData *f, const GeSgxVertexLayout *l, uint32_t count, uint32_t *off, uint32_t *nv) {
	uint32_t size = l->size, dst = (f->nraw + 15u) & ~15u, n = 0, i;
	if (dst + count * size > f->max_raw)
		return 0;
	if (++gather_cur == 0) {
		memset(gather_stamp, 0, sizeof(gather_stamp));
		gather_cur = 1;
	}
	for (i = 0; i < count; i++) {
		uint32_t idx = prim_index[i];
		if (idx > 0xFFFF)
			return 0;
		if (gather_stamp[idx] != gather_cur) {
			gather_stamp[idx] = gather_cur;
			gather_map[idx] = (uint16_t)n;
			if (!geSgxRead(w, w->vaddr + idx * size, f->raw + dst + n * size, size))
				return 0;
			n++;
		}
		prim_index[i] = gather_map[idx];
	}
	f->nraw = dst + n * size;
	*off = dst;
	*nv = n;
	return 1;
}

void geSgxBuildFinish(GeSgxWalker *w) {
	if (raw_frame == w->frame)
		rawFlush(w);
	raw_frame = NULL;
	uni_frame = NULL;
}


/*
	Builds one PRIM. The vertices it uses are transformed once each into the
	frame (an indexed primitive over a compact range: that range, otherwise
	every referenced vertex in order), the topology becomes a triangle, line
	or point list of indices relative to them, which lets consecutive
	primitives of one state share a GXM draw.
*/
void geSgxBuildPrim(GeSgxWalker *w, uint32_t type, uint32_t count, const GeSgxVertexLayout *l) {
	GeSgxFrameData *f = w->frame;

	if (count == 0 || !l->pos_fmt)
		return;

	/*
		The draw state and the transform setup only change when a register,
		matrix or CLUT did (lists draw many primitives in one state).
	*/
	static GeSgxDraw cached_d;
	static Xform cached_x;
	static const GeSgxWalker *cached_w = NULL;
	uint32_t (* clk)(void) = w->clock_fn;
	uint32_t t0 = clk ? clk() : 0, t1;
	if (w->state_dirty || cached_w != w) {
		drawState(w, &cached_d);
		xformInit(w, l, &cached_x);
		cached_x.tex_w = cached_x.tex_h = 1.0f;
		if (cached_d.flags & GE_SGX_DRAW_TEXTURED) {
			if (clk) {
				t1 = clk();
				f->us_state += t1 - t0;
				t0 = t1;
			}
			cached_d.texture = drawTexture(w);
			if (clk) {
				t1 = clk();
				f->us_texture += t1 - t0;
				t0 = t1;
			}
			uint32_t size = reg(w, GE_CMD_TEXSIZE0);
			cached_x.tex_w = 1.0f / (float)(1u << (size & 0xF));
			cached_x.tex_h = 1.0f / (float)(1u << ((size >> 8) & 0xF));
		}
		cached_w = w;
		w->state_dirty = 0;
		uni_valid = 0;
		f->state_builds++;
	}
	if (clk) {
		t1 = clk();
		f->us_state += t1 - t0;
		t0 = t1;
	}
	if ((cached_d.flags & GE_SGX_DRAW_TEXTURED) && cached_d.texture < 0) {
		f->skipped_other++;
		return;
	}

	GeSgxDraw d = cached_d;
	const Xform *xp = &cached_x;

	// Output indices per input vertex, at most
	uint32_t nidx;
	switch (type) {
		case GE_PRIM_POINTS: d.prim = GE_SGX_DRAW_POINTS; nidx = count; break;
		case GE_PRIM_LINES: d.prim = GE_SGX_DRAW_LINES; nidx = count; break;
		case GE_PRIM_LINE_STRIP: d.prim = GE_SGX_DRAW_LINES; nidx = count * 2; break;
		case GE_PRIM_TRIANGLES: d.prim = GE_SGX_DRAW_TRIANGLES; nidx = count; break;
		case GE_PRIM_TRIANGLE_STRIP:
		case GE_PRIM_TRIANGLE_FAN: d.prim = GE_SGX_DRAW_TRIANGLES; nidx = count * 3; break;
		case GE_PRIM_RECTANGLES: d.prim = GE_SGX_DRAW_TRIANGLES; nidx = count * 3; break;
		default:
			f->skipped_other++;
			return;
	}
	uint32_t nverts_max = type == GE_PRIM_RECTANGLES ? count * 2 : count;
	if (nverts_max > GE_SGX_MAX_DRAW_VERTICES) {
		f->skipped_other++;
		return;
	}

	if (f->nvertices + nverts_max > f->max_vertices || f->nindices + nidx > f->max_indices || f->ndraws >= f->max_draws) {
		f->skipped_other++;
		return;
	}

	// Which vertex each primitive vertex is
	uint32_t i;
	uint32_t imin = 0, imax = count - 1;
	if (l->index_size) {
		uint32_t bytes = count * l->index_size;
		static uint8_t raw[65536 * 4];
		if (!geSgxRead(w, w->iaddr, raw, bytes)) {
			f->skipped_other++;
			return;
		}
		imin = 0xFFFFFFFFu;
		imax = 0;
		for (i = 0; i < count; i++) {
			uint32_t v;
			switch (l->index_size) {
				case 1: v = raw[i]; break;
				case 2: { uint16_t s; memcpy(&s, raw + i * 2, 2); v = s; break; }
				default: memcpy(&v, raw + i * 4, 4); break;
			}
			prim_index[i] = v;
			if (v < imin) imin = v;
			if (v > imax) imax = v;
		}
	} else {
		for (i = 0; i < count; i++)
			prim_index[i] = i;
	}

	uint32_t vstart = f->nvertices, istart = f->nindices;
	GeSgxVertex *vout = &f->vertices[vstart];
	uint16_t *iout = &f->indices[istart];
	uint32_t nv = 0, ni = 0;
	uint32_t t_vert = 0;

	// The vertex data: the range of a non-indexed or compact indexed primitive at once
	uint32_t range = imax - imin + 1;
	int compact = l->index_size && range <= count;

	/*
		GPU transform: the vertex range is copied as it is into the raw arena,
		the vertex program decodes and transforms it. Indices scattered over a
		big vertex buffer: only the vertices they use are gathered. Sprites,
		morphing and 32 bit indices stay on the CPU path.
	*/
	int gpu = 0;
	uint32_t raw_off = 0;
	if (w->gpu_transform && f->raw && !xp->through) {
		int why = -1;
		if (type == GE_PRIM_RECTANGLES)
			why = GE_SGX_CPU_SPRITES;
		else if (!gpuLayout(w, l))
			why = GE_SGX_CPU_LAYOUT;
		else if (!gpuUniforms(w, f, l, xp))
			why = GE_SGX_CPU_FULL;
		else if (range <= GE_SGX_MAX_DRAW_VERTICES && range * l->size <= GPU_MAX_RANGE_BYTES &&
		         (!l->index_size || range <= 2 * count + 16)) {
			if (rawCopy(w, f, w->vaddr + imin * l->size, range * l->size, &raw_off)) {
				gpu = 1;
				for (i = 0; i < count; i++)
					prim_index[i] -= imin;
				nv = range;
			} else {
				why = GE_SGX_CPU_FULL;
			}
		} else if (rawGather(w, f, l, count, &raw_off, &nv)) {
			gpu = 1;
		} else {
			why = GE_SGX_CPU_FULL;
		}
		if (why >= 0)
			f->cpu_reason[why]++;
	}

	bulk_count = 0;
	if (gpu)
		;
	else if (!l->index_size)
		bulkLoad(w, l, 0, count);
	else if (compact)
		bulkLoad(w, l, imin, range);

	if (gpu) {
		// Indices only (below)
	} else if (type == GE_PRIM_RECTANGLES) {
		// Sprites: two corners each, screen aligned (3D ones after the projection)
		for (i = 0; i + 1 < count; i += 2) {
			GeSgxVertex a, b;
			if (!vertexOut(w, xp, prim_index[i], &a) || !vertexOut(w, xp, prim_index[i + 1], &b))
				goto fail;
			if (!xp->through) {
				if (a.w <= 0.0f || b.w <= 0.0f)
					continue;
				float ia = 1.0f / a.w, ib = 1.0f / b.w;
				a.x *= ia; a.y *= ia; a.z *= ia; a.w = 1.0f;
				b.x *= ib; b.y *= ib; b.z *= ib; b.w = 1.0f;
			}
			// Corners: positions and texture coordinates of the two vertices,
			// colour, depth and fog of the second
			GeSgxVertex *c = &vout[nv];
			c[0] = b; c[0].x = a.x; c[0].y = a.y; c[0].u = a.u; c[0].v = a.v;
			c[1] = b; c[1].y = a.y; c[1].v = a.v;
			c[2] = b; c[2].x = a.x; c[2].u = a.u;
			c[3] = b;
			iout[ni++] = (uint16_t)(nv + 0);
			iout[ni++] = (uint16_t)(nv + 1);
			iout[ni++] = (uint16_t)(nv + 2);
			iout[ni++] = (uint16_t)(nv + 1);
			iout[ni++] = (uint16_t)(nv + 3);
			iout[ni++] = (uint16_t)(nv + 2);
			nv += 4;
		}
	}
	if (gpu || type != GE_PRIM_RECTANGLES) {
		// Local vertex numbers: the compact index range, or one vertex per index
		if (gpu) {
			// prim_index is already local
		} else if (compact) {
			for (i = 0; i < range; i++) {
				if (!vertexOut(w, xp, imin + i, &vout[i]))
					goto fail;
			}
			nv = range;
			for (i = 0; i < count; i++)
				prim_index[i] -= imin;
		} else {
			for (i = 0; i < count; i++) {
				if (!vertexOut(w, xp, prim_index[i], &vout[i]))
					goto fail;
				prim_index[i] = i;
			}
			nv = count;
		}
		const uint32_t *s = prim_index;
		if (clk) {
			t1 = clk();
			t_vert = t1 - t0;
			t0 = t1;
		}

		switch (type) {
			case GE_PRIM_POINTS:
				for (i = 0; i < count; i++)
					iout[ni++] = (uint16_t)s[i];
				break;
			case GE_PRIM_LINES:
				for (i = 0; i + 1 < count; i += 2) {
					iout[ni++] = (uint16_t)s[i];
					iout[ni++] = (uint16_t)s[i + 1];
				}
				break;
			case GE_PRIM_LINE_STRIP:
				for (i = 0; i + 1 < count; i++) {
					iout[ni++] = (uint16_t)s[i];
					iout[ni++] = (uint16_t)s[i + 1];
				}
				break;
			case GE_PRIM_TRIANGLES:
				for (i = 0; i + 2 < count; i += 3) {
					iout[ni++] = (uint16_t)s[i];
					iout[ni++] = (uint16_t)s[i + 1];
					iout[ni++] = (uint16_t)s[i + 2];
				}
				break;
			case GE_PRIM_TRIANGLE_STRIP:
				for (i = 0; i + 2 < count; i++) {
					uint32_t a = s[i], b = s[i + 1], c = s[i + 2];
					if (a == b || b == c || a == c)
						continue; // degenerate (strip joints)
					if (i & 1) {
						uint32_t t = a;
						a = b;
						b = t;
					}
					iout[ni++] = (uint16_t)a;
					iout[ni++] = (uint16_t)b;
					iout[ni++] = (uint16_t)c;
				}
				break;
			case GE_PRIM_TRIANGLE_FAN:
				for (i = 1; i + 1 < count; i++) {
					iout[ni++] = (uint16_t)s[0];
					iout[ni++] = (uint16_t)s[i];
					iout[ni++] = (uint16_t)s[i + 1];
				}
				break;
		}
	}

	if (ni == 0 || nv == 0)
		return;

	if (!gpu) {
		f->nvertices = vstart + nv;
		f->vertices_built += nv;
	}
	f->nindices = istart + ni;
	if (clk) {
		t1 = clk();
		if (type == GE_PRIM_RECTANGLES)
			t_vert += t1 - t0; // corners and indices together
		else
			f->us_indices += t1 - t0;
		f->us_vertices += t_vert;
	}
	if (!xp->through)
		f->draws_3d++;

	if (gpu) {
		f->prims_gpu++;
		d.gpu = 1;
		d.vkey = (uint16_t)gpu_vkey;
		d.vtype = reg(w, GE_CMD_VERTEXTYPE);
		d.ucommon = gpu_ucommon;
		d.uobject = gpu_uobject;

		// The same vertex buffer as the previous draw: its indices continue from that base
		if (f->ndraws > 0) {
			GeSgxDraw *p = &f->draws[f->ndraws - 1];
			if (p->gpu && p->first + p->count == istart && sameState(p, &d) && p->vtype == d.vtype &&
			    p->ucommon == d.ucommon && p->uobject == d.uobject && raw_off >= p->base &&
			    (raw_off - p->base) % l->size == 0 && (raw_off - p->base) / l->size + nv <= GE_SGX_MAX_DRAW_VERTICES) {
				uint16_t delta = (uint16_t)((raw_off - p->base) / l->size);
				if (delta) {
					for (i = 0; i < ni; i++)
						iout[i] = (uint16_t)(iout[i] + delta);
				}
				p->count += ni;
				return;
			}
		}
		d.base = raw_off;
		d.first = istart;
		d.count = ni;
		f->draws[f->ndraws++] = d;
		return;
	}

	// Lists of the same state follow each other: extend the previous draw
	if (f->ndraws > 0) {
		GeSgxDraw *p = &f->draws[f->ndraws - 1];
		if (!p->gpu && p->first + p->count == istart && sameState(p, &d) && vstart + nv - p->base <= GE_SGX_MAX_DRAW_VERTICES) {
			uint16_t delta = (uint16_t)(vstart - p->base);
			if (delta) {
				for (i = 0; i < ni; i++)
					iout[i] = (uint16_t)(iout[i] + delta);
			}
			p->count += ni;
			return;
		}
	}

	d.base = vstart;
	d.first = istart;
	d.count = ni;
	f->draws[f->ndraws++] = d;
	return;

fail:
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
