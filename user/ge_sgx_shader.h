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

// Fragment program key of a draw: the GE state the fragment shader depends on
uint32_t geSgxShaderKey(const GeSgxDraw *d);

/*
	The compiled program of key, or NULL while it is not there yet: the key
	is then queued for the compiler thread (libshacccg.suprx, results kept
	in ux0:data/adrenaline/sgx_shaders). Draw thread only.
*/
const SceGxmProgram *geSgxShaderGet(uint32_t key);

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
