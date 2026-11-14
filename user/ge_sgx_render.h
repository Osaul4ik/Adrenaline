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

#ifndef __GE_SGX_RENDER_H__
#define __GE_SGX_RENDER_H__

#include <stdint.h>
#include <psp2/gxm.h>
#include <vita2d.h>
#include "ge_sgx_frame.h"
#include "ge_sgx_walk.h"

typedef struct {
	uint32_t draws;    // drawn
	uint32_t vertices;
	uint32_t waiting;  // draws whose shader is still being compiled
	uint32_t other_fb; // draws into other framebuffers (not shown)
	uint32_t gpu_draws; // with the GPU transform
	uint32_t render_us;
} GeSgxRenderStats;

// Set when a GPU transform vertex program cannot be had: the walker then transforms on the CPU
extern volatile int ge_sgx_gpu_failed;
// Vertex types (attribute bits) the GPU transform cannot draw (no vertex program for them)
extern volatile uint32_t ge_sgx_gpu_bad[GE_SGX_GPU_BAD_MAX];
extern volatile int ge_sgx_gpu_nbad;

// Draw thread, with the GXM context of vita2d. Returns < 0 on error.
int geSgxRenderInit(void);

/*
	Draws the frame: each framebuffer's draws into that framebuffer's own
	rendering (twice its size), as scenes of their own (none may be open).
	Framebuffers used as textures are read from their renderings. scenes:
	how many were drawn (the first one reset vita2d's vertex pool).
	Returns the rendering of fb (or of the framebuffer copied into it,
	aliases), NULL if there is none.
*/
vita2d_texture *geSgxRenderFrame(SceGxmContext *ctx, const GeSgxFrameData *frame, uint32_t fb,
	const GeSgxAlias *aliases, int naliases, GeSgxRenderStats *stats, int *scenes);

// The rendering last shown (NULL if none): renderings may be remade, so ask every time
vita2d_texture *geSgxRenderShown(void);

#endif
