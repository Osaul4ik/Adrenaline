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

#ifndef __GE_SGX_TEX_H__
#define __GE_SGX_TEX_H__

#include <stdint.h>
#include <psp2/gxm.h>
#include "ge_sgx_walk.h"

typedef struct {
	uint32_t entries;   // textures cached
	uint32_t kb;        // GPU memory they use
	uint32_t uploads;   // since the last call: converted
	uint32_t upload_kb;
	uint32_t upload_us;
	uint32_t checks;    // content checks of cached textures
	uint32_t failed;    // could not be uploaded (format, memory)
	uint32_t mip_uploads; // of the uploads with mipmap levels
} GeSgxTexStats;

// Walker thread
int geSgxTexInit(void);
void geSgxTexReset(void);
int32_t geSgxTexLookup(void *user, const GeSgxTexDesc *desc); // GeSgxTextureFn
void geSgxTexFrameDone(void); // after each published frame
void geSgxTexStats(GeSgxTexStats *stats);

// Draw thread: the texture of a handle from a published frame
const SceGxmTexture *geSgxTexGet(int32_t handle);

#endif
