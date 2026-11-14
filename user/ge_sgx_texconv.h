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
	SGX renderer: PSP texture data to GXM texture data (pure C, tested on
	the host). 16 and 32 bit formats keep their layout (the GXM formats
	U5U6U5_BGR, U1U5U5U5_ABGR, U4U4U4U4_ABGR and U8U8U8U8_ABGR are the PSP
	ones), CLUT formats are expanded to 32 bit with the CLUT.
*/

#ifndef __GE_SGX_TEXCONV_H__
#define __GE_SGX_TEXCONV_H__

#include <stdint.h>
#include "ge_sgx_walk.h"

enum GeSgxTexFormats {
	GE_TFMT_5650, GE_TFMT_5551, GE_TFMT_4444, GE_TFMT_8888,
	GE_TFMT_CLUT4, GE_TFMT_CLUT8, GE_TFMT_CLUT16, GE_TFMT_CLUT32,
	GE_TFMT_DXT1, GE_TFMT_DXT3, GE_TFMT_DXT5,
};

#define GE_SGX_CLUT_ENTRIES 512

// Bytes of one row of bufw texels in PSP memory, 0 if the format is not supported
uint32_t geSgxTexRowBytes(uint32_t format, uint32_t bufw);

// Rows to read from PSP memory (swizzled textures are stored in blocks of 8 rows)
uint32_t geSgxTexRows(const GeSgxTexDesc *d);

// Bytes per texel of the converted texture (2 or 4)
uint32_t geSgxTexOutBpp(uint32_t format);

/*
	CLUT4/CLUT8 textures can stay indexed (GXM P8 with a 256 entry palette):
	a quarter of the memory of 8888. The palette applies the CLUT index
	shift, mask and start; CLUT4 indices are widened to bytes.
*/
static inline int geSgxTexIndexed(uint32_t format) {
	return format == 4 || format == 5; // GE_TFMT_CLUT4, GE_TFMT_CLUT8
}
void geSgxTexPalette(const GeSgxTexDesc *d, const uint32_t *clut_rgba, uint32_t *palette);
int geSgxTexConvertIndices(const GeSgxTexDesc *d, const uint8_t *src, uint32_t src_rowbytes, uint8_t *dst, uint32_t dst_stride);

// Row stride of the converted texture: GXM linear textures use rows of a multiple of 8 texels
static inline uint32_t geSgxTexOutStride(const GeSgxTexDesc *d) {
	return ((d->w + 7u) & ~7u) * geSgxTexOutBpp(d->format);
}

// PSP swizzle (blocks of 16 bytes x 8 rows) to linear rows
void geSgxUnswizzle(uint8_t *dst, const uint8_t *src, uint32_t rowbytes, uint32_t rows);

// CLUT entries as 0xAABBGGRR, GE_SGX_CLUT_ENTRIES of them
void geSgxClutDecode(uint32_t clut_format, const uint8_t *clut, uint32_t bytes, uint32_t *out);

/*
	src: linear texels, src_rowbytes per row (geSgxTexRowBytes), at least h
	rows. dst: d->h rows of geSgxTexOutStride bytes. clut: from
	geSgxClutDecode (CLUT formats). Returns 0 if the format is not supported.
*/
int geSgxTexConvert(const GeSgxTexDesc *d, const uint8_t *src, uint32_t src_rowbytes,
	uint8_t *dst, uint32_t dst_stride, const uint32_t *clut);

// 16 bit GE colour (0: 565, 1: 5551, 2: 4444) to 0xAABBGGRR
uint32_t geSgxDecodeColor16(uint32_t fmt, uint32_t c);

uint32_t geSgxHash(const void *data, uint32_t size);

#endif
