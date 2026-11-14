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
	SGX renderer: PSP texture data to GXM texture data. See ge_sgx_texconv.h.
	Layouts follow PPSSPP (GPU/Common/TextureDecoder.cpp).
*/

#include <string.h>

#include "ge_sgx_texconv.h"

uint32_t geSgxTexRowBytes(uint32_t format, uint32_t bufw) {
	switch (format) {
		case GE_TFMT_5650:
		case GE_TFMT_5551:
		case GE_TFMT_4444:
		case GE_TFMT_CLUT16:
			return bufw * 2;
		case GE_TFMT_8888:
		case GE_TFMT_CLUT32:
			return bufw * 4;
		case GE_TFMT_CLUT8:
			return bufw;
		case GE_TFMT_CLUT4:
			return (bufw + 1) / 2;
		// DXT: one row of 4x4 blocks
		case GE_TFMT_DXT1:
			return ((bufw + 3) / 4) * 8;
		case GE_TFMT_DXT3:
		case GE_TFMT_DXT5:
			return ((bufw + 3) / 4) * 16;
		default:
			return 0;
	}
}

static inline int isDxt(uint32_t format) {
	return format >= GE_TFMT_DXT1 && format <= GE_TFMT_DXT5;
}

uint32_t geSgxTexRows(const GeSgxTexDesc *d) {
	if (isDxt(d->format))
		return (d->h + 3u) / 4; // block rows, never swizzled
	return d->swizzle ? ((d->h + 7u) & ~7u) : d->h;
}

/*
	PSP DXT blocks (PPSSPP DXTDecoder): the colour part has the four index
	bytes first, then the two 565 colours (blue in the low bits); DXT3/5
	start with that colour part, then the alpha.
*/
static inline uint32_t rgba(uint32_t r, uint32_t g, uint32_t b, uint32_t a) {
	return r | (g << 8) | (b << 16) | (a << 24);
}

static void dxtColors(const uint8_t *blk, int dxt1, uint32_t *c) {
	uint32_t c1 = (uint32_t)blk[4] | ((uint32_t)blk[5] << 8);
	uint32_t c2 = (uint32_t)blk[6] | ((uint32_t)blk[7] << 8);
	uint32_t b1 = (c1 << 3) & 0xF8, b2 = (c2 << 3) & 0xF8;
	uint32_t g1 = (c1 >> 3) & 0xFC, g2 = (c2 >> 3) & 0xFC;
	uint32_t r1 = (c1 >> 8) & 0xF8, r2 = (c2 >> 8) & 0xF8;
	uint32_t a = dxt1 ? 0xFF : 0;
	c[0] = rgba(r1, g1, b1, a);
	c[1] = rgba(r2, g2, b2, a);
	if (c1 > c2) {
		c[2] = rgba((r1 * 2 + r2) / 3, (g1 * 2 + g2) / 3, (b1 * 2 + b2) / 3, a);
		c[3] = rgba((r2 * 2 + r1) / 3, (g2 * 2 + g1) / 3, (b2 * 2 + b1) / 3, a);
	} else {
		c[2] = rgba((r1 + r2) / 2, (g1 + g2) / 2, (b1 + b2) / 2, a);
		c[3] = 0;
	}
}

static void dxtBlock(uint32_t format, const uint8_t *blk, uint32_t *out, uint32_t stride_px, uint32_t bw, uint32_t bh) {
	uint32_t c[4], x, y;
	dxtColors(blk, format == GE_TFMT_DXT1, c);

	uint8_t alpha[8];
	uint64_t abits = 0;
	if (format == GE_TFMT_DXT5) {
		uint32_t d2;
		memcpy(&d2, blk + 8, 4);
		uint64_t d1 = (uint64_t)blk[12] | ((uint64_t)blk[13] << 8);
		abits = (d1 << 32) | d2;
		uint32_t a0 = blk[14], a1 = blk[15];
		alpha[0] = (uint8_t)a0;
		alpha[1] = (uint8_t)a1;
		if (a0 > a1) {
			for (x = 1; x < 7; x++)
				alpha[x + 1] = (uint8_t)(((7 - x) * a0 + x * a1) / 7);
		} else {
			for (x = 1; x < 5; x++)
				alpha[x + 1] = (uint8_t)(((5 - x) * a0 + x * a1) / 5);
			alpha[6] = 0;
			alpha[7] = 255;
		}
	}

	for (y = 0; y < 4; y++) {
		uint32_t idx = blk[y];
		uint32_t aline = 0;
		if (format == GE_TFMT_DXT3)
			aline = (uint32_t)blk[8 + y * 2] | ((uint32_t)blk[9 + y * 2] << 8);
		for (x = 0; x < 4; x++) {
			uint32_t v = c[idx & 3];
			idx >>= 2;
			if (format == GE_TFMT_DXT3) {
				v = (v & 0x00FFFFFFu) | (((aline >> (x * 4)) & 0xF) * 0x11u) << 24;
			} else if (format == GE_TFMT_DXT5) {
				v = (v & 0x00FFFFFFu) | ((uint32_t)alpha[abits & 7] << 24);
				abits >>= 3;
			}
			if (x < bw && y < bh)
				out[y * stride_px + x] = v;
		}
	}
}

uint32_t geSgxTexOutBpp(uint32_t format) {
	return format <= GE_TFMT_4444 ? 2 : 4;
}

void geSgxUnswizzle(uint8_t *dst, const uint8_t *src, uint32_t rowbytes, uint32_t rows) {
	uint32_t bxc = rowbytes / 16;
	uint32_t byc = rows / 8;
	uint32_t by, bx, line;

	for (by = 0; by < byc; by++) {
		for (bx = 0; bx < bxc; bx++) {
			const uint8_t *block = src + (by * bxc + bx) * 128;
			for (line = 0; line < 8; line++)
				memcpy(dst + (by * 8 + line) * rowbytes + bx * 16, block + line * 16, 16);
		}
	}
}

void geSgxClutDecode(uint32_t clut_format, const uint8_t *clut, uint32_t bytes, uint32_t *out) {
	uint32_t fmt = clut_format & 3;
	uint32_t i, n;

	memset(out, 0, GE_SGX_CLUT_ENTRIES * sizeof(uint32_t));

	if (fmt == 3) {
		n = bytes / 4;
		if (n > GE_SGX_CLUT_ENTRIES)
			n = GE_SGX_CLUT_ENTRIES;
		memcpy(out, clut, n * 4);
	} else {
		n = bytes / 2;
		if (n > GE_SGX_CLUT_ENTRIES)
			n = GE_SGX_CLUT_ENTRIES;
		for (i = 0; i < n; i++)
			out[i] = geSgxDecodeColor16(fmt, (uint32_t)clut[i * 2] | ((uint32_t)clut[i * 2 + 1] << 8));
	}
}

static inline uint32_t clutIndex(uint32_t clut_format, uint32_t raw) {
	uint32_t shift = (clut_format >> 2) & 0x1F;
	uint32_t mask = (clut_format >> 8) & 0xFF;
	uint32_t base = ((clut_format >> 16) & 0x1F) * 16;
	// 32 bit entries: the CLUT has 256 of them, 16 bit ones 512
	uint32_t limit = (clut_format & 3) == 3 ? 0xFF : 0x1FF;
	return (((raw >> shift) & mask) | base) & limit;
}

void geSgxTexPalette(const GeSgxTexDesc *d, const uint32_t *clut_rgba, uint32_t *palette) {
	uint32_t i, n = d->format == GE_TFMT_CLUT4 ? 16 : 256;
	for (i = 0; i < 256; i++)
		palette[i] = i < n ? clut_rgba[clutIndex(d->clut_format, i)] : 0;
}

int geSgxTexConvertIndices(const GeSgxTexDesc *d, const uint8_t *src, uint32_t src_rowbytes, uint8_t *dst, uint32_t dst_stride) {
	uint32_t x, y, w = d->w, h = d->h;
	for (y = 0; y < h; y++) {
		const uint8_t *s = src + y * src_rowbytes;
		uint8_t *o = dst + y * dst_stride;
		if (d->format == GE_TFMT_CLUT8) {
			uint32_t n = w < src_rowbytes ? w : src_rowbytes;
			memcpy(o, s, n);
			if (n < w)
				memset(o + n, 0, w - n);
		} else if (d->format == GE_TFMT_CLUT4) {
			for (x = 0; x < w; x++)
				o[x] = x / 2 < src_rowbytes ? (uint8_t)((s[x / 2] >> ((x & 1) * 4)) & 0xF) : 0;
		} else {
			return 0;
		}
	}
	return 1;
}

int geSgxTexConvert(const GeSgxTexDesc *d, const uint8_t *src, uint32_t src_rowbytes,
	uint8_t *dst, uint32_t dst_stride, const uint32_t *clut) {
	uint32_t x, y;
	uint32_t w = d->w, h = d->h;

	switch (d->format) {
		case GE_TFMT_5650:
		case GE_TFMT_5551:
		case GE_TFMT_4444:
		case GE_TFMT_8888: {
			uint32_t row = w * geSgxTexOutBpp(d->format);
			if (row > src_rowbytes)
				row = src_rowbytes;
			for (y = 0; y < h; y++) {
				memcpy(dst + y * dst_stride, src + y * src_rowbytes, row);
				if (row < dst_stride)
					memset(dst + y * dst_stride + row, 0, dst_stride - row);
			}
			return 1;
		}
		case GE_TFMT_CLUT4:
			for (y = 0; y < h; y++) {
				const uint8_t *s = src + y * src_rowbytes;
				uint32_t *o = (uint32_t *)(dst + y * dst_stride);
				for (x = 0; x < w && x / 2 < src_rowbytes; x++) {
					uint32_t raw = (s[x / 2] >> ((x & 1) * 4)) & 0xF;
					o[x] = clut[clutIndex(d->clut_format, raw)];
				}
			}
			return 1;
		case GE_TFMT_CLUT8:
			for (y = 0; y < h; y++) {
				const uint8_t *s = src + y * src_rowbytes;
				uint32_t *o = (uint32_t *)(dst + y * dst_stride);
				for (x = 0; x < w && x < src_rowbytes; x++)
					o[x] = clut[clutIndex(d->clut_format, s[x])];
			}
			return 1;
		case GE_TFMT_CLUT16:
			for (y = 0; y < h; y++) {
				const uint8_t *s = src + y * src_rowbytes;
				uint32_t *o = (uint32_t *)(dst + y * dst_stride);
				for (x = 0; x < w && x * 2 + 1 < src_rowbytes; x++)
					o[x] = clut[clutIndex(d->clut_format, (uint32_t)s[x * 2] | ((uint32_t)s[x * 2 + 1] << 8))];
			}
			return 1;
		case GE_TFMT_CLUT32:
			for (y = 0; y < h; y++) {
				const uint8_t *s = src + y * src_rowbytes;
				uint32_t *o = (uint32_t *)(dst + y * dst_stride);
				for (x = 0; x < w && x * 4 + 3 < src_rowbytes; x++) {
					uint32_t raw;
					memcpy(&raw, s + x * 4, 4);
					o[x] = clut[clutIndex(d->clut_format, raw)];
				}
			}
			return 1;
		case GE_TFMT_DXT1:
		case GE_TFMT_DXT3:
		case GE_TFMT_DXT5: {
			uint32_t bsize = d->format == GE_TFMT_DXT1 ? 8 : 16;
			uint32_t stride_px = dst_stride / 4;
			for (y = 0; y < h; y += 4) {
				const uint8_t *s = src + (y / 4) * src_rowbytes;
				uint32_t *o = (uint32_t *)(dst + y * dst_stride);
				for (x = 0; x < w && (x / 4 + 1) * bsize <= src_rowbytes; x += 4)
					dxtBlock(d->format, s + (x / 4) * bsize, o + x, stride_px, w - x < 4 ? w - x : 4, h - y < 4 ? h - y : 4);
			}
			return 1;
		}
		default:
			return 0;
	}
}
