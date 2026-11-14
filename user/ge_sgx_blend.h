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
	SGX renderer: the PSP's blending in GXM terms (pure C: also used by the
	host tests).

	GXM has no constant blend colour, the PSP has two (BLENDFIXEDA/B). A
	fixed colour that is black or white is the factor ZERO or ONE. Other
	ones move into the fragment program (PPSSPP does the same on GPUs
	without constant colours):
	- source FIXA: the colour is multiplied by FIXA in the program, the
	  source factor becomes ONE;
	- destination FIXB: the program outputs FIXB (its grey) as the alpha and
	  the destination factor becomes SRC_ALPHA, or ONE_MINUS_SRC_ALPHA of
	  FIXA when FIXB is 1 - FIXA (a cross-fade); a source factor that used
	  the source alpha is then applied in the program as well.
	The program does: c.rgb *= rgb * (c.a * as + ab); c.a = c.a * am + aa
	(after the alpha test). Such an alpha is no real alpha: it is not
	written to the framebuffer (alpha_bogus).
*/

#ifndef __GE_SGX_BLEND_H__
#define __GE_SGX_BLEND_H__

#include <stdint.h>
#include "ge_sgx_frame.h"
#include "ge_sgx_gpu.h"

enum {
	GE_SGX_BF_ZERO,
	GE_SGX_BF_ONE,
	GE_SGX_BF_SRC_COLOR,
	GE_SGX_BF_INV_SRC_COLOR,
	GE_SGX_BF_DST_COLOR,
	GE_SGX_BF_INV_DST_COLOR,
	GE_SGX_BF_SRC_ALPHA,
	GE_SGX_BF_INV_SRC_ALPHA,
	GE_SGX_BF_DST_ALPHA,
	GE_SGX_BF_INV_DST_ALPHA,
	GE_SGX_BF_FIX, // internal only
};

typedef struct {
	uint8_t src, dst;   // GE_SGX_BF_*
	uint8_t eq;         // BLENDMODE equation 0..5
	uint8_t fix;        // the program changes the colour (fragment key bit GE_SGX_FS_FIX)
	uint8_t alpha_bogus; // the output alpha is not the fragment's
	float rgb[3], as, ab, am, aa;
} GeSgxBlend;

// 0: black, 1: white, -1: another colour
static inline int geSgxFixTrivial(uint32_t c) {
	int i, black = 1, white = 1;
	for (i = 0; i < 3; i++) {
		uint32_t v = (c >> (i * 8)) & 0xFF;
		if (v > 1)
			black = 0;
		if (v < 254)
			white = 0;
	}
	return black ? 0 : white ? 1 : -1;
}

static inline float geSgxFixGrey(uint32_t c) {
	return ((c & 0xFF) + ((c >> 8) & 0xFF) + ((c >> 16) & 0xFF)) * (1.0f / (3.0f * 255.0f));
}

static inline uint8_t geSgxPspFactor(uint32_t f, uint32_t fix, int src) {
	switch (f) {
		case 0: return src ? GE_SGX_BF_DST_COLOR : GE_SGX_BF_SRC_COLOR;
		case 1: return src ? GE_SGX_BF_INV_DST_COLOR : GE_SGX_BF_INV_SRC_COLOR;
		case 2: case 6: return GE_SGX_BF_SRC_ALPHA;     // doubled alpha: as the single one
		case 3: case 7: return GE_SGX_BF_INV_SRC_ALPHA;
		case 4: case 8: return GE_SGX_BF_DST_ALPHA;
		case 5: case 9: return GE_SGX_BF_INV_DST_ALPHA;
		default: {
			int t = geSgxFixTrivial(fix);
			return t == 0 ? GE_SGX_BF_ZERO : t == 1 ? GE_SGX_BF_ONE : GE_SGX_BF_FIX;
		}
	}
}

// The blending of a draw with GE_SGX_DRAW_BLEND
static inline void geSgxBlendSetup(const GeSgxDraw *d, GeSgxBlend *b) {
	int i;
	b->eq = (uint8_t)((d->blend >> 8) & 7);
	b->fix = 0;
	b->alpha_bogus = 0;
	b->rgb[0] = b->rgb[1] = b->rgb[2] = 1.0f;
	b->as = 0.0f;
	b->ab = 1.0f;
	b->am = 1.0f;
	b->aa = 0.0f;
	b->src = geSgxPspFactor(d->blend & 0xF, d->fix_a, 1);
	b->dst = geSgxPspFactor((d->blend >> 4) & 0xF, d->fix_b, 0);

	if (b->src == GE_SGX_BF_FIX) {
		b->fix = 1;
		for (i = 0; i < 3; i++)
			b->rgb[i] = ((d->fix_a >> (i * 8)) & 0xFF) * (1.0f / 255.0f);
		b->src = GE_SGX_BF_ONE;
		if (b->dst == GE_SGX_BF_FIX) {
			int cross = 1;
			for (i = 0; i < 3; i++) {
				int s = (int)((d->fix_a >> (i * 8)) & 0xFF) + (int)((d->fix_b >> (i * 8)) & 0xFF);
				if (s < 252 || s > 258)
					cross = 0;
			}
			b->am = 0.0f;
			b->alpha_bogus = 1;
			if (cross) {
				b->aa = geSgxFixGrey(d->fix_a);
				b->dst = GE_SGX_BF_INV_SRC_ALPHA;
			} else {
				b->aa = geSgxFixGrey(d->fix_b);
				b->dst = GE_SGX_BF_SRC_ALPHA;
			}
		}
	} else if (b->dst == GE_SGX_BF_FIX) {
		b->fix = 1;
		if (b->src == GE_SGX_BF_SRC_ALPHA) {
			b->as = 1.0f;
			b->ab = 0.0f;
			b->src = GE_SGX_BF_ONE;
		} else if (b->src == GE_SGX_BF_INV_SRC_ALPHA) {
			b->as = -1.0f;
			b->ab = 1.0f;
			b->src = GE_SGX_BF_ONE;
		}
		b->am = 0.0f;
		b->aa = geSgxFixGrey(d->fix_b);
		b->alpha_bogus = 1;
		b->dst = GE_SGX_BF_SRC_ALPHA;
	}
}

#endif
