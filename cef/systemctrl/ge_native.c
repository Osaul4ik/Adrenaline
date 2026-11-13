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
	Native resolution rendering: firmware integration.
	See ge_native.h for the overview.
*/

#include <common.h>
#include <pspge.h>

#include "main.h"
#include "ge_native.h"

/* Flips that show a new buffer without any upscaled draw before the native
   output is dropped for the game's own 480x272 framebuffer (videos and other
   CPU-drawn screens). */
#define PASSTHROUGH_FLIPS 4

static void *(* _sceGeEdramGetAddr)(void);
static unsigned int (* _sceGeEdramGetSize)(void);
static int (* _sceGeGetList)(int qid, void *list, int *flag);
static int (* _sceGeListUpdateStallAddr)(int qid, void *stall);
static int (* _sceGeListEnQueue)(const void *list, void *stall, int cbid, PspGeListArgs *arg);
static int (* _sceGeListEnQueueHead)(const void *list, void *stall, int cbid, PspGeListArgs *arg);
static int (* _sceDisplaySetFrameBuf)(void *topaddr, int bufferwidth, int pixelformat, int sync);

static u32 last_topaddr = 0;
static int idle_flips = 0;
static int passthrough = 0;

static void copyFrameBuffer(void) {
	if (passthrough)
		return;

	_sw(1, DRAW_NATIVE);

	u32 *list = (u32 *)(GE_NATIVE_COPY_LIST | 0xA0000000);
	geNativeListBuildCopy(list);
	_sceGeListEnQueue((void *)GE_NATIVE_COPY_LIST, NULL, -1, NULL);
}

static void *sceGeEdramGetAddrPatched(void) {
	return (void *)GE_NATIVE_FAKE_VRAM;
}

static unsigned int sceGeEdramGetSizePatched(void) {
	return GE_NATIVE_FAKE_VRAM_SIZE;
}

static int sceGeListUpdateStallAddrPatched(int qid, void *stall) {
	int k1 = pspSdkSetK1(0);

	char info[64];
	if (_sceGeGetList(qid, info, NULL) == 0) {
		u16 list_state = *(u16 *)(info + 0x08);
		if (list_state != 3) { // not completed
			void *list = *(void **)(info + 0x18); // previous stall
			if (!list)
				list = *(void **)(info + 0x14); // list
			if (((u32)list & 0x0FFFFFFF) < ((u32)stall & 0x0FFFFFFF))
				geNativeListPatch((u32)list, (u32)stall);
		}
	}

	sceKernelDcacheWritebackInvalidateAll();

	pspSdkSetK1(k1);
	return _sceGeListUpdateStallAddr(qid, stall);
}

static int sceGeListEnQueuePatched(const void *list, void *stall, int cbid, PspGeListArgs *arg) {
	if (geNativeListHasDraws())
		copyFrameBuffer();

	geNativeListReset();
	geNativeListPatch((u32)list, (u32)stall);
	sceKernelDcacheWritebackInvalidateAll();

	return _sceGeListEnQueue(list, stall, cbid, arg);
}

static int sceGeListEnQueueHeadPatched(const void *list, void *stall, int cbid, PspGeListArgs *arg) {
	geNativeListReset();
	geNativeListPatch((u32)list, (u32)stall);
	sceKernelDcacheWritebackInvalidateAll();

	return _sceGeListEnQueueHead(list, stall, cbid, arg);
}

static int sceDisplaySetFrameBufPatched(void *topaddr, int bufferwidth, int pixelformat, int sync) {
	u32 top = (u32)topaddr & 0x0FFFFFFF;

	if (geNativeListTakeFrameDraws()) {
		idle_flips = 0;
	} else if (top && top != last_topaddr) {
		// A different buffer is shown and the GE drew nothing into ours:
		// the picture is produced by the CPU/ME, e.g. a video.
		if (idle_flips < PASSTHROUGH_FLIPS)
			idle_flips++;
	}

	last_topaddr = top;

	if (idle_flips >= PASSTHROUGH_FLIPS) {
		passthrough = 1;
		_sw(0, DRAW_NATIVE);
	} else {
		passthrough = 0;
		copyFrameBuffer();
	}

	return _sceDisplaySetFrameBuf(topaddr, bufferwidth, pixelformat, sync);
}

static int isUserMemoryBelowFakeVram(void) {
	PspSysmemPartitionInfo info;
	memset(&info, 0, sizeof(info));
	info.size = sizeof(info);

	if (sceKernelQueryMemoryPartitionInfo661(PSP_MEMORY_PARTITION_USER, &info) < 0)
		return 0;

	return ((info.startaddr & 0x0FFFFFFF) + info.memsize) <= GE_NATIVE_FAKE_VRAM;
}

int GeNativeInit(void) {
	if (sceKernelInitKeyConfig() != PSP_INIT_KEYCONFIG_GAME)
		return 0;

	// Option from the Adrenaline menu (written by the Vita side)
	if (_lw(NATIVE_RES_CONFIG) != NATIVE_RES_MAGIC)
		return 0;

	// 0x0A000000-0x0A900000 must not belong to the game (high memory layout)
	if (!isUserMemoryBelowFakeVram()) {
		log("GeNative: disabled, high memory layout\n");
		return 0;
	}

	// The standalone plugins do the same job, do not patch twice
	if (sceKernelFindModuleByName661("GePatch") || sceKernelFindModuleByName661("GTANativeResolution")) {
		log("GeNative: disabled, native resolution plugin loaded\n");
		return 0;
	}

	// Hold L while starting the game to skip
	int (* _sceCtrlPeekBufferPositive)(SceCtrlData *pad_data, int count) = (void *)FindProc("sceController_Service", "sceCtrl_driver", 0xC4AAD55F);
	if (_sceCtrlPeekBufferPositive) {
		SceCtrlData pad;
		memset(&pad, 0, sizeof(pad));
		_sceCtrlPeekBufferPositive(&pad, 1);
		if (pad.Buttons & PSP_CTRL_LTRIGGER) {
			log("GeNative: skipped by user\n");
			return 0;
		}
	}

	_sceGeEdramGetAddr = (void *)FindProc("sceGE_Manager", "sceGe_driver", 0xE47E40E4);
	_sceGeEdramGetSize = (void *)FindProc("sceGE_Manager", "sceGe_driver", 0x1F6752AD);
	_sceGeGetList = (void *)FindProc("sceGE_Manager", "sceGe_driver", 0x67B01D8E);
	_sceGeListUpdateStallAddr = (void *)FindProc("sceGE_Manager", "sceGe_driver", 0xE0D68148);
	_sceGeListEnQueue = (void *)FindProc("sceGE_Manager", "sceGe_driver", 0xAB49E76A);
	_sceGeListEnQueueHead = (void *)FindProc("sceGE_Manager", "sceGe_driver", 0x1C0D95A6);
	_sceDisplaySetFrameBuf = (void *)FindProc("sceDisplay_Service", "sceDisplay_driver", 0x289D82FE);

	if (!_sceGeEdramGetAddr || !_sceGeEdramGetSize || !_sceGeGetList || !_sceGeListUpdateStallAddr ||
	    !_sceGeListEnQueue || !_sceGeListEnQueueHead || !_sceDisplaySetFrameBuf) {
		log("GeNative: GE/display functions not found\n");
		return -1;
	}

	geNativeListInit();
	last_topaddr = 0;
	idle_flips = 0;
	passthrough = 0;
	_sw(0, DRAW_NATIVE);

	sctrlHENPatchSyscall((u32)_sceGeEdramGetAddr, sceGeEdramGetAddrPatched);
	sctrlHENPatchSyscall((u32)_sceGeEdramGetSize, sceGeEdramGetSizePatched);
	sctrlHENPatchSyscall((u32)_sceGeListUpdateStallAddr, sceGeListUpdateStallAddrPatched);
	sctrlHENPatchSyscall((u32)_sceGeListEnQueue, sceGeListEnQueuePatched);
	sctrlHENPatchSyscall((u32)_sceGeListEnQueueHead, sceGeListEnQueueHeadPatched);
	sctrlHENPatchSyscall((u32)_sceDisplaySetFrameBuf, sceDisplaySetFrameBufPatched);

	ClearCaches();

	log("GeNative: enabled\n");

	return 1;
}
