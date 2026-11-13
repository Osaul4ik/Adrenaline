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
	Native resolution (960x544) rendering for PSP games.

	The PSP GE is not emulated by ScePspemu on the Vita side: it executes the
	game's display lists itself and only the finished 480x272 framebuffer is
	handed to the Vita display. The only place where the render resolution can
	be raised is therefore the display list: framebuffer, depth buffer,
	viewport, offset, scissor, drawing region and through-mode vertices are
	rewritten so that the GE rasterizes at 960x544 into real VRAM, and the
	result is copied to NATIVE_FRAMEBUFFER which the Vita side shows 1:1
	(DRAW_NATIVE).

	The approach follows TheFloW's GePatch (which already proved it on real
	hardware). Differences:
	- built into systemctrl and controlled from the Adrenaline menu;
	- viewport/offset/scissor/region are scaled from the game's own values
	  instead of being forced to full screen, so sub-viewports keep working;
	- rewritten commands remember their original value, so display lists that
	  are executed more than once are not scaled twice and draws that had to be
	  disabled for one pass are restored on the next pass;
	- scaled through-mode vertex ranges are recognised by a fingerprint of
	  their positions instead of a marker in the vertex padding, so vertex
	  layouts without padding and draws with few vertices are not scaled
	  again when a buffer is drawn more than once;
	- every list, vertex and index address is range-checked and list traversal
	  is bounded, so a bad list cannot hang or crash the kernel.
*/

#ifndef __GE_NATIVE_H__
#define __GE_NATIVE_H__

#ifdef GE_NATIVE_HOST_TEST
#include <stdint.h>
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef int16_t s16;
void *ge_host_ptr(u32 addr);
#define GE_PTR(addr) ge_host_ptr((u32)(addr))
#else
#include <psptypes.h>
#define GE_PTR(addr) ((void *)(addr))
#endif

#define GE_NATIVE_WIDTH  960
#define GE_NATIVE_HEIGHT 544
#define GE_NATIVE_PITCH  960

/* Memory layout (physical PSP addresses, normal 24 MB user memory layout only) */
#define GE_NATIVE_FAKE_VRAM       0x0A000000 /* what the game gets from sceGeEdramGetAddr */
#define GE_NATIVE_FAKE_VRAM_SIZE  0x00400000
#define GE_NATIVE_DISPLAY_BUFFER  0x0A400000 /* == NATIVE_FRAMEBUFFER, read by the Vita side */
#define GE_NATIVE_ORIG_MAP        0x0A600000 /* original values of rewritten commands */
#define GE_NATIVE_VTX_MAP         0x0A640000 /* fingerprints of scaled vertex ranges */
#define GE_NATIVE_COPY_LIST       0x0A800000 /* our own display list (framebuffer copy) */

/* Real VRAM: 960x544 RGB565 colour + 16 bit depth fill all 2 MB */
#define GE_NATIVE_VRAM_DRAW   0x04000000
#define GE_NATIVE_VRAM_DEPTH  0x04100000
#define GE_NATIVE_VRAM_DUMMY  0x041FF000 /* 4 KB scratch target for ignored framebuffers */

#define GE_NATIVE_ORIG_MAP_ENTRIES 16384 /* power of two, 16 bytes each */
#define GE_NATIVE_VTX_MAP_ENTRIES  8192  /* power of two, 20 bytes each */

/* GE commands used by the patcher */
#define GE_CMD_NOP               0x00u
#define GE_CMD_VADDR             0x01u
#define GE_CMD_IADDR             0x02u
#define GE_CMD_PRIM              0x04u
#define GE_CMD_BEZIER            0x05u
#define GE_CMD_SPLINE            0x06u
#define GE_CMD_BOUNDINGBOX       0x07u
#define GE_CMD_JUMP              0x08u
#define GE_CMD_BJUMP             0x09u
#define GE_CMD_CALL              0x0Au
#define GE_CMD_RET               0x0Bu
#define GE_CMD_END               0x0Cu
#define GE_CMD_SIGNAL            0x0Eu
#define GE_CMD_FINISH            0x0Fu
#define GE_CMD_BASE              0x10u
#define GE_CMD_VERTEXTYPE        0x12u
#define GE_CMD_OFFSETADDR        0x13u
#define GE_CMD_ORIGIN            0x14u
#define GE_CMD_REGION1           0x15u
#define GE_CMD_REGION2           0x16u
#define GE_CMD_TEXTUREMAPENABLE  0x1Eu
#define GE_CMD_ALPHABLENDENABLE  0x21u
#define GE_CMD_BONEMATRIXDATA    0x2Bu
#define GE_CMD_VIEWPORTXSCALE    0x42u
#define GE_CMD_VIEWPORTYSCALE    0x43u
#define GE_CMD_VIEWPORTXCENTER   0x45u
#define GE_CMD_VIEWPORTYCENTER   0x46u
#define GE_CMD_OFFSETX           0x4Cu
#define GE_CMD_OFFSETY           0x4Du
#define GE_CMD_FRAMEBUFPTR       0x9Cu
#define GE_CMD_FRAMEBUFWIDTH     0x9Du
#define GE_CMD_ZBUFPTR           0x9Eu
#define GE_CMD_ZBUFWIDTH         0x9Fu
#define GE_CMD_TEXADDR0          0xA0u
#define GE_CMD_TEXADDR7          0xA7u
#define GE_CMD_TEXBUFWIDTH0      0xA8u
#define GE_CMD_TEXBUFWIDTH7      0xAFu
#define GE_CMD_TRANSFERSRC       0xB2u
#define GE_CMD_TRANSFERSRCW      0xB3u
#define GE_CMD_TRANSFERDST       0xB4u
#define GE_CMD_TRANSFERDSTW      0xB5u
#define GE_CMD_FRAMEBUFPIXFORMAT 0xD2u
#define GE_CMD_CLEARMODE         0xD3u
#define GE_CMD_SCISSOR1          0xD4u
#define GE_CMD_SCISSOR2          0xD5u
#define GE_CMD_TRANSFERSTART     0xEAu
#define GE_CMD_TRANSFERSRCPOS    0xEBu
#define GE_CMD_TRANSFERDSTPOS    0xECu
#define GE_CMD_TRANSFERSIZE      0xEEu

#define GE_FORMAT_565 0

#define GE_SIGNAL_SYNC 0x08
#define GE_SIGNAL_JUMP 0x10
#define GE_SIGNAL_CALL 0x11
#define GE_SIGNAL_RET  0x12

/* Data of the NOP we put in place of a disabled draw (a real NOP has data 0) */
#define GE_NATIVE_NOP_MARKER 0x00A5A5A5

/* Vertex type */
#define GE_VTYPE_TC_SHIFT          0
#define GE_VTYPE_TC_MASK           (3 << 0)
#define GE_VTYPE_COL_SHIFT         2
#define GE_VTYPE_COL_MASK          (7 << 2)
#define GE_VTYPE_NRM_SHIFT         5
#define GE_VTYPE_NRM_MASK          (3 << 5)
#define GE_VTYPE_POS_SHIFT         7
#define GE_VTYPE_POS_MASK          (3 << 7)
#define GE_VTYPE_WEIGHT_SHIFT      9
#define GE_VTYPE_WEIGHT_MASK       (3 << 9)
#define GE_VTYPE_IDX_SHIFT         11
#define GE_VTYPE_IDX_MASK          (3 << 11)
#define GE_VTYPE_IDX_NONE          (0 << 11)
#define GE_VTYPE_IDX_8BIT          (1 << 11)
#define GE_VTYPE_IDX_16BIT         (2 << 11)
#define GE_VTYPE_IDX_32BIT         (3 << 11)
#define GE_VTYPE_WEIGHTCOUNT_SHIFT 14
#define GE_VTYPE_WEIGHTCOUNT_MASK  (7 << 14)
#define GE_VTYPE_MORPHCOUNT_SHIFT  18
#define GE_VTYPE_MORPHCOUNT_MASK   (7 << 18)
#define GE_VTYPE_THROUGH           (1 << 23)

/* Display list patcher (ge_native_list.c) */
void geNativeListInit(void);            /* clears the original-value map */
void geNativeListReset(void);           /* new display list: reset traversal state */
void geNativeListPatch(u32 list, u32 stall);
int geNativeListHasDraws(void);
u32 geNativeListTakeFrameDraws(void);   /* upscaled draws since last call */
int geNativeListBuildCopy(u32 *list);   /* fills a framebuffer copy list, returns words */

#ifdef GE_NATIVE_TRACE
extern int geNativeTraceOn;
void geNativeTrace(const char *fmt, ...);
#endif

/* Integration (ge_native.c) */
int GeNativeInit(void);

#endif
