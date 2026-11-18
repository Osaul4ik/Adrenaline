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

#ifndef __GE_SGX_SHADER_H__
#define __GE_SGX_SHADER_H__

#include <stdint.h>
#include <psp2/gxm.h>
#include "ge_sgx_frame.h"

#include "ge_sgx_cg.h"

/*
	PSP blend factors the GPU has no constant for (FIX: the fixed colours,
	BLENDFIXEDA/B), made in the fragment program: the source factor
	multiplied into the colour (GE_SGX_KEY_PREMUL: rgb times uPre.rgb +
	uPre.a * alpha, the blend's source factor then ONE), a grey fixed
	destination factor put in the output alpha (GE_SGX_KEY_ALPHA_OUT: the
	blend's destination factor then SRC_ALPHA; the alpha is not written to
	the framebuffer, draws mask it). Rounded to 0 or 1 they were: The 3rd
	Birthday's glowing chips the wrong colour.
*/
#define GE_SGX_KEY_PREMUL    (1u << 14)
#define GE_SGX_KEY_ALPHA_OUT (1u << 15)

typedef struct {
	int premul, aout;
	float pre[4];
	float aval;
} GeSgxBlendEmu;

static inline int geSgxFixExact(uint32_t fix) {
	fix &= 0x00FFFFFFu;
	return fix == 0 || fix == 0x00FFFFFFu;
}

static inline int geSgxBlendEmu(const GeSgxDraw *d, GeSgxBlendEmu *e) {
	e->premul = e->aout = 0;
	e->pre[0] = e->pre[1] = e->pre[2] = e->pre[3] = 0.0f;
	e->aval = 1.0f;
	if (!(d->flags & GE_SGX_DRAW_BLEND) || (d->flags & GE_SGX_DRAW_CLEAR))
		return 0;
	uint32_t fs = d->blend & 0xF, fd = (d->blend >> 4) & 0xF;
	if (((d->blend >> 8) & 7) >= 3)
		return 0; // minimum, maximum, difference: no factors
	uint32_t fa = d->fix_a & 0x00FFFFFFu, fb = d->fix_b & 0x00FFFFFFu;
	int sfix = fs >= 10 && !geSgxFixExact(fa);
	int dfix = fd >= 10 && !geSgxFixExact(fb);
	if (dfix) {
		int r = fb & 0xFF, g = (fb >> 8) & 0xFF, b = (fb >> 16) & 0xFF;
		int lo = r < g ? (r < b ? r : b) : (g < b ? g : b), hi = r > g ? (r > b ? r : b) : (g > b ? g : b);
		if (hi - lo > 3)
			dfix = 0; // a coloured one: not made (rounded as before)
		else
			e->aval = (float)(r + g + b) / (3.0f * 255.0f);
	}
	if (dfix) {
		// The source factor must then not use the source alpha: multiplied in
		switch (fs) {
			case 2: e->pre[3] = 1.0f; e->premul = 1; break;                                            // SRC_ALPHA
			case 3: e->pre[0] = e->pre[1] = e->pre[2] = 1.0f; e->pre[3] = -1.0f; e->premul = 1; break; // 1 - SRC_ALPHA
			case 6: e->pre[3] = 2.0f; e->premul = 1; break;                                            // 2 SRC_ALPHA
			case 0: case 1: case 4: case 5: case 8: case 9: break;                                     // no source alpha in them
			case 7: return 0;
			default: // FIX
				e->pre[0] = (fa & 0xFF) / 255.0f;
				e->pre[1] = ((fa >> 8) & 0xFF) / 255.0f;
				e->pre[2] = ((fa >> 16) & 0xFF) / 255.0f;
				e->premul = 1;
				break;
		}
		e->aout = 1;
	} else if (sfix && fd >= 2) {
		// (not with the source colour as destination factor: it would be multiplied too)
		e->pre[0] = (fa & 0xFF) / 255.0f;
		e->pre[1] = ((fa >> 8) & 0xFF) / 255.0f;
		e->pre[2] = ((fa >> 16) & 0xFF) / 255.0f;
		e->premul = 1;
	}
	return e->premul || e->aout;
}

// Fragment program key of a draw: the GE state the fragment shader depends on
uint32_t geSgxShaderKey(const GeSgxDraw *d);

/*
	The compiled program of key, or NULL while it is not there yet: the key
	is then queued for the compiler thread (libshacccg.suprx, results kept
	in ux0:data/adrenaline/sgx_shaders). Draw thread only.
*/
const SceGxmProgram *geSgxShaderGet(uint32_t key);

// The program of key could not be made (not cached and no compiler, or a compile error)
int geSgxShaderFailed(uint32_t key);

typedef struct {
	uint32_t ready;    // programs available
	uint32_t queued;   // waiting for the compiler
	uint32_t failed;   // did not compile
	uint32_t compiled; // compiled in this session
	uint32_t loaded;   // read from the disk cache
	int compiler;      // 1: loaded, 0: not (yet), < 0: error loading it
	uint32_t compile_us; // time of the last compile
} GeSgxShaderStats;

void geSgxShaderStats(GeSgxShaderStats *stats);

#endif
