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
#include <psp2/types.h>
#include <psp2/kernel/sysmem.h>
#include "ge_sgx_walk.h"
#include "ge_sgx_frame.h"

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
	uint32_t invalidates; // PSP memory blocks copied per second
	uint32_t resyncs;     // ring gaps since the start
	uint32_t sync_waits, sync_timeouts, sync_max_us; // PSP side, since the start

	// Memory left for the renderer, measured at the start
	uint32_t cdram_kb;
	uint32_t ram_kb;
} GeSgxFrame;

// Modes of config.sgx_renderer
#define GE_SGX_MODE_OFF  0
#define GE_SGX_MODE_WALK 1 // lists followed, nothing drawn (statistics only)
#define GE_SGX_MODE_DRAW 2 // and drawn with GXM, shown in a corner
#define GE_SGX_MODE_FULL 3 // drawn with GXM, shown full screen

// Appends a line to ux0:data/adrenaline_sgx_log.txt
void GeSgxLog(const char *fmt, ...);

// Free memory of that type our process can still allocate (in 1 MB blocks), in KB
uint32_t GeSgxFreeKB(SceKernelMemBlockType type);

// Thread to report in the log (CPU use, core), no-op when the option is off
void GeSgxTrackThread(SceUID uid, const char *name);

// A published frame for the draw thread
typedef struct {
	int slot;
	const GeSgxFrameData *data;
	uint32_t fb;               // VRAM offset of the shown framebuffer
	const GeSgxAlias *aliases; // framebuffer copies into it
	int naliases;
} GeSgxFrameRef;

// Draw thread: the newest published frame (1) or none new (0); then GeSgxFrameRendered
int GeSgxTakeFrame(GeSgxFrameRef *ref);
void GeSgxFrameRendered(const GeSgxFrameRef *ref);
// Draw thread: once per loop (the GPU is done with frames rendered some ticks ago)
void GeSgxDrawTick(void);

// PSP memory in our address space (uncached), NULL if the range is not PSP RAM/VRAM
const uint8_t *geSgxPspPtr(uint32_t addr, uint32_t size);

// Kermit command from systemctrl: start reading the ring (1) or not (0)
int GeSgxStart(uint32_t ring_addr);

// Latest frame; returns 0 when no game reported a flip in the last 500 ms
int GeSgxGetFrame(GeSgxFrame *frame);

#endif
