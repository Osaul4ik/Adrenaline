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
#include "ge_sgx_frame.h"
#include "ge_sgx_walk.h"

typedef struct {
	uint32_t draws;    // drawn
	uint32_t vertices;
	uint32_t waiting;  // draws whose shader is still being compiled
	uint32_t other_fb; // draws into other framebuffers (not shown)
	uint32_t render_us;
} GeSgxRenderStats;

// Draw thread, with the GXM context of vita2d. Returns < 0 on error.
int geSgxRenderInit(void);

/*
	Draws the frame's draws into the framebuffer fb (or copied into it,
	aliases) into the scene that is being drawn (960x544).
*/
void geSgxRender(SceGxmContext *ctx, const GeSgxFrameData *frame, uint32_t fb,
	const GeSgxAlias *aliases, int naliases, GeSgxRenderStats *stats);

#endif
