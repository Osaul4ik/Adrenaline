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

#ifndef __GE_SGX_H__
#define __GE_SGX_H__

#include <stdint.h>
#include "ge_sgx_walk.h"

#define GE_SGX_FRAME_RECTS 64

typedef struct {
	uint32_t flips;       // display flips seen (0: nothing yet)
	int nrects;
	GeSgxRect rects[GE_SGX_FRAME_RECTS]; // clears of the shown framebuffer
	GeSgxStats stats;     // GE work of the last frame

	// Measured over the last second
	uint32_t fps;
	uint32_t events;      // ring events per second
	uint32_t walk_us;     // time spent walking lists per second
	uint32_t walk_max_us; // longest single walk
	uint32_t invalidates; // pages fetched (cache invalidated) per second
	uint32_t resyncs;     // ring gaps since the start
	uint32_t sync_waits, sync_timeouts, sync_max_us; // PSP side, since the start

	// Memory left for the renderer, measured at the start
	uint32_t cdram_kb;
	uint32_t ram_kb;
} GeSgxFrame;

// Modes of config.sgx_renderer
#define GE_SGX_MODE_OFF  0
#define GE_SGX_MODE_WALK 1 // lists followed, nothing drawn (statistics only)
#define GE_SGX_MODE_DRAW 2 // and drawn with GXM

// Appends a line to ux0:data/adrenaline_sgx_log.txt
void GeSgxLog(const char *fmt, ...);

// Kermit command from systemctrl: start reading the ring (1) or not (0)
int GeSgxStart(uint32_t ring_addr);

// Latest frame; returns 0 when no game reported a flip in the last 500 ms
int GeSgxGetFrame(GeSgxFrame *frame);

#endif
