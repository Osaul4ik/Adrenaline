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
	SGX renderer, PSP side: reports the GE work of the game to the Vita side.
	See ge_sgx_shared.h for the overview.

	Nothing the game hands to the GE is changed: the lists still run on the
	PSP GE. The hooks only record what was queued, and make DrawSync/ListSync
	wait (a few milliseconds at most) until the Vita side has read the lists,
	because after a sync the game may overwrite them.
*/

#include <common.h>
#include <pspge.h>

#include "main.h"
#include "adrenaline.h"
#include "ge_sgx.h"
#include "../../ge_sgx_shared.h"

#define RING_ENTRIES_MIN 1024

static int (* _sceGeListEnQueue)(const void *list, void *stall, int cbid, PspGeListArgs *arg);
static int (* _sceGeListEnQueueHead)(const void *list, void *stall, int cbid, PspGeListArgs *arg);
static int (* _sceGeListDeQueue)(int qid);
static int (* _sceGeListUpdateStallAddr)(int qid, void *stall);
static int (* _sceGeDrawSync)(int syncType);
static int (* _sceGeListSync)(int qid, int syncType);
static int (* _sceDisplaySetFrameBuf)(void *topaddr, int bufferwidth, int pixelformat, int sync);
static u32 (* _sceKernelGetSystemTimeLow)(void);

static volatile GeSgxRing *ring = NULL; // uncached
static u32 ring_mask = 0;
static SceUID ring_blockid = -1;

// The Vita side stopped reading: do not wait for it until it reads again
static int vita_stalled = 0;
static u32 vita_stalled_read = 0;

static volatile GeSgxAck *shared = (volatile GeSgxAck *)GE_SGX_ACK_ADDRESS; // uncached

static inline void memorySync(void) {
	asm volatile ("sync" ::: "memory");
}

static void ringPush(u32 type, u32 a, u32 b, u32 c, u32 d) {
	int intr = sceKernelCpuSuspendIntr();

	u32 w = ring->write;
	u32 read = shared->read;

	if (shared->session == ring->session && (w - read) > ring_mask) {
		// Full: the Vita side notices the gap and starts over
		ring->lost++;
	} else {
		volatile GeSgxEvent *e = &ring->events[w & ring_mask];
		e->type = type;
		e->a = a;
		e->b = b;
		e->c = c;
		e->d = d;
		memorySync();
		e->seq = w;
		memorySync();
		ring->write = w + 1;
	}

	sceKernelCpuResumeIntr(intr);
}

static u32 timeNow(void) {
	return _sceKernelGetSystemTimeLow ? _sceKernelGetSystemTimeLow() : 0;
}

// Wait until the Vita side has read every event written so far
static void waitForVita(void) {
	u32 target = ring->write;

	if (shared->session != ring->session)
		return;

	u32 read = shared->read;
	if ((int)(read - target) >= 0)
		return;

	if (vita_stalled) {
		if (read == vita_stalled_read)
			return;
		vita_stalled = 0;
	}

	u32 start = timeNow();
	u32 waited = 0;
	int loops = 0;

	while (1) {
		read = shared->read;
		if ((int)(read - target) >= 0)
			break;

		waited = _sceKernelGetSystemTimeLow ? (timeNow() - start) : (u32)loops * 100;
		if (waited >= GE_SGX_SYNC_TIMEOUT_US) {
			ring->sync_timeouts++;
			vita_stalled = 1;
			vita_stalled_read = read;
			break;
		}

		// Inside a callback or with the dispatcher off: do not spin
		if (sceKernelDelayThread(100) < 0)
			break;
		loops++;
	}

	ring->sync_waits++;
	ring->sync_wait_us += waited;
	if (waited > ring->sync_max_us)
		ring->sync_max_us = waited;
}

static int sceGeListEnQueuePatched(const void *list, void *stall, int cbid, PspGeListArgs *arg) {
	int qid = _sceGeListEnQueue(list, stall, cbid, arg);
	if (qid >= 0)
		ringPush(GE_SGX_EVENT_LIST, qid, (u32)list, (u32)stall, 0);
	return qid;
}

static int sceGeListEnQueueHeadPatched(const void *list, void *stall, int cbid, PspGeListArgs *arg) {
	int qid = _sceGeListEnQueueHead(list, stall, cbid, arg);
	if (qid >= 0)
		ringPush(GE_SGX_EVENT_LIST, qid, (u32)list, (u32)stall, 1);
	return qid;
}

static int sceGeListDeQueuePatched(int qid) {
	int res = _sceGeListDeQueue(qid);
	if (res >= 0)
		ringPush(GE_SGX_EVENT_DEQUEUE, qid, 0, 0, 0);
	return res;
}

static int sceGeListUpdateStallAddrPatched(int qid, void *stall) {
	int res = _sceGeListUpdateStallAddr(qid, stall);
	if (res >= 0)
		ringPush(GE_SGX_EVENT_STALL, qid, (u32)stall, 0, 0);
	return res;
}

static int sceGeDrawSyncPatched(int syncType) {
	int res = _sceGeDrawSync(syncType);
	if (syncType == 0)
		waitForVita();
	return res;
}

static int sceGeListSyncPatched(int qid, int syncType) {
	int res = _sceGeListSync(qid, syncType);
	if (syncType == 0)
		waitForVita();
	return res;
}

static int sceDisplaySetFrameBufPatched(void *topaddr, int bufferwidth, int pixelformat, int sync) {
	ringPush(GE_SGX_EVENT_FLIP, (u32)topaddr, bufferwidth, pixelformat, sync);
	return _sceDisplaySetFrameBuf(topaddr, bufferwidth, pixelformat, sync);
}

static int allocRing(void) {
	u32 entries;
	for (entries = GE_SGX_RING_ENTRIES; entries >= RING_ENTRIES_MIN; entries /= 2) {
		u32 size = sizeof(GeSgxRing) - (GE_SGX_RING_ENTRIES - entries) * sizeof(GeSgxEvent);
		SceUID blockid = sceKernelAllocPartitionMemory661(PSP_MEMORY_PARTITION_KERNEL, "GeSgxRing", PSP_SMEM_High, size, NULL);
		if (blockid < 0)
			continue;

		void *mem = sceKernelGetBlockHeadAddr661(blockid);
		memset(mem, 0, size);
		sceKernelDcacheWritebackInvalidateAll();

		ring_blockid = blockid;
		ring_mask = entries - 1;
		ring = (volatile GeSgxRing *)((u32)mem | 0x20000000);
		ring->entries = entries;
		return 1;
	}

	return 0;
}

static void freeRing(void) {
	if (ring_blockid >= 0)
		sceKernelFreePartitionMemory661(ring_blockid);
	ring_blockid = -1;
	ring = NULL;
}

int GeSgxInit(void) {
	if (sceKernelInitKeyConfig() != PSP_INIT_KEYCONFIG_GAME)
		return 0;

	// These rewrite the lists for the PSP GE, the Vita side needs the original ones
	if (sceKernelFindModuleByName661("GePatch") || sceKernelFindModuleByName661("GTANativeResolution")) {
		log("GeSgx: disabled, native resolution plugin loaded\n");
		return 0;
	}

	// Hold L while starting the game to skip
	int (* _sceCtrlPeekBufferPositive)(SceCtrlData *pad_data, int count) = (void *)FindProc("sceController_Service", "sceCtrl_driver", 0xC4AAD55F);
	if (_sceCtrlPeekBufferPositive) {
		SceCtrlData pad;
		memset(&pad, 0, sizeof(pad));
		_sceCtrlPeekBufferPositive(&pad, 1);
		if (pad.Buttons & PSP_CTRL_LTRIGGER) {
			log("GeSgx: skipped by user\n");
			return 0;
		}
	}

	_sceGeListEnQueue = (void *)FindProc("sceGE_Manager", "sceGe_driver", 0xAB49E76A);
	_sceGeListEnQueueHead = (void *)FindProc("sceGE_Manager", "sceGe_driver", 0x1C0D95A6);
	_sceGeListDeQueue = (void *)FindProc("sceGE_Manager", "sceGe_driver", 0x5FB86AB0);
	_sceGeListUpdateStallAddr = (void *)FindProc("sceGE_Manager", "sceGe_driver", 0xE0D68148);
	_sceGeDrawSync = (void *)FindProc("sceGE_Manager", "sceGe_driver", 0xB287BD61);
	_sceGeListSync = (void *)FindProc("sceGE_Manager", "sceGe_driver", 0x03444EB4);
	_sceDisplaySetFrameBuf = (void *)FindProc("sceDisplay_Service", "sceDisplay_driver", 0x289D82FE);
	_sceKernelGetSystemTimeLow = (void *)FindProc("sceThreadManager", "ThreadManForKernel", 0x369ED59D);

	if (!_sceGeListEnQueue || !_sceGeListEnQueueHead || !_sceGeListDeQueue || !_sceGeListUpdateStallAddr ||
	    !_sceGeDrawSync || !_sceGeListSync || !_sceDisplaySetFrameBuf) {
		log("GeSgx: GE/display functions not found\n");
		return -1;
	}

	if (!allocRing()) {
		log("GeSgx: no memory for the ring\n");
		return -1;
	}

	ring->magic = GE_SGX_MAGIC;
	ring->version = GE_SGX_VERSION;
	ring->session = shared->session + 1;
	if (ring->session == 0)
		ring->session = 1;

	// The Vita side answers 1 if the option is on and it took the ring
	// (an adrenaline_user that does not know the command answers -1)
	// Kernel (0x88...) address: the Vita side maps the whole PSP RAM for those
	int res = SendAdrenalineCmdArg(ADRENALINE_VITA_CMD_GE_SGX_START, (u32)ring & ~0x20000000);
	log("GeSgx: start = %d, ring %08X, %d entries, session %d\n", res, (unsigned int)ring, (int)(ring_mask + 1), (int)ring->session);
	if (res != 1) {
		freeRing();
		return 0;
	}

	vita_stalled = 0;

	sctrlHENPatchSyscall((u32)_sceGeListEnQueue, sceGeListEnQueuePatched);
	sctrlHENPatchSyscall((u32)_sceGeListEnQueueHead, sceGeListEnQueueHeadPatched);
	sctrlHENPatchSyscall((u32)_sceGeListDeQueue, sceGeListDeQueuePatched);
	sctrlHENPatchSyscall((u32)_sceGeListUpdateStallAddr, sceGeListUpdateStallAddrPatched);
	sctrlHENPatchSyscall((u32)_sceGeDrawSync, sceGeDrawSyncPatched);
	sctrlHENPatchSyscall((u32)_sceGeListSync, sceGeListSyncPatched);
	sctrlHENPatchSyscall((u32)_sceDisplaySetFrameBuf, sceDisplaySetFrameBufPatched);
	ClearCaches();

	log("GeSgx: enabled\n");
	return 1;
}
