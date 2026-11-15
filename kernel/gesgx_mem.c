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
	Memory for the SGX renderer of adrenaline_user, outside the emulator's budget.

	The emulator's process has a few MB of CDRAM and main RAM left: too little
	for the texture cache and the 2x render targets. The pages here are taken by
	the kernel (its own partition, not the process budget), then the same pages
	are mapped into the calling process as an ordinary user memblock, which
	sceGxmMapMemory accepts like any other.

	Two ways to map them, the first that works is used (the result says which):
	  1 a user memblock at the kernel block's physical address (HAS_PADDR),
	    when the kernel block is physically contiguous;
	  2 a user memblock mirroring the kernel block (HAS_MIRROR_BLOCKID).

	The kernel blocks of a process that ends without freeing them (the emulator
	is closed, not stopped) are freed once it is gone: marked when it exits,
	freed when the next process is created (its pages and the GPU's mapping of
	them are gone by then), or by the next call here.

	Called only from adrenaline_user's init (one thread); no lock.
*/

#include <psp2kern/kernel/sysmem.h>
#include <psp2kern/kernel/threadmgr.h>
#include <psp2kern/kernel/cpu.h>
#include <psp2kern/kernel/proc_event.h>

#include <string.h>

#include "../ge_sgx_kernel.h"

#define MAX_BLOCKS 8

typedef struct {
	SceUID pid;    // process the block is mapped into
	SceUID kuid;   // the kernel's block (owns the pages)
	SceUID uuid;   // the process's block (same pages)
	void *base;    // its address in the process
	int exited;    // the process has exited: the kernel block is freed next
} Block;

static Block blocks[MAX_BLOCKS];

// attr: SCE_KERNEL_ALLOC_MEMBLOCK_ATTR_PHYCONT, _DONT_MAP (the kernel does not need the pages mapped)
static SceUID allocKernel(SceKernelMemBlockType ktype, SceSize size, uint32_t attr) {
	SceKernelAllocMemBlockKernelOpt opt;
	memset(&opt, 0, sizeof(opt));
	opt.size = sizeof(opt);
	opt.attr = attr;
	return ksceKernelAllocMemBlock("SceGeSgxKernelPages", ktype, size, &opt);
}

static SceUID allocUser(SceUID pid, SceKernelMemBlockType type, SceSize size, uint32_t paddr, SceUID mirror) {
	SceKernelAllocMemBlockKernelOpt opt;
	memset(&opt, 0, sizeof(opt));
	opt.size = sizeof(opt);
	opt.attr = SCE_KERNEL_ALLOC_MEMBLOCK_ATTR_HAS_PID;
	opt.pid = pid;
	if (mirror >= 0) {
		opt.attr |= SCE_KERNEL_ALLOC_MEMBLOCK_ATTR_HAS_MIRROR_BLOCKID;
		opt.mirror_blockid = mirror;
	} else {
		opt.attr |= SCE_KERNEL_ALLOC_MEMBLOCK_ATTR_HAS_PADDR;
		opt.paddr = paddr;
	}
	return ksceKernelAllocMemBlock("SceGeSgxPool", type, size, &opt);
}

// Kernel blocks of processes that ended without freeing theirs (their own mapping went with them);
// all others' when pid is given (one process at a time uses these)
static void freeStale(SceUID pid) {
	int i;
	for (i = 0; i < MAX_BLOCKS; i++) {
		if (blocks[i].kuid > 0 && (blocks[i].exited || (pid && blocks[i].pid != pid))) {
			ksceKernelFreeMemBlock(blocks[i].kuid);
			memset(&blocks[i], 0, sizeof(blocks[i]));
		}
	}
}

static int onExit(SceUID pid) {
	int i;
	for (i = 0; i < MAX_BLOCKS; i++) {
		if (blocks[i].kuid > 0 && blocks[i].pid == pid)
			blocks[i].exited = 1;
	}
	return 0;
}

static int procCreate(SceUID pid, SceProcEventInvokeParam2 *param, int a3) {
	freeStale(0);
	return 0;
}

static int procExit(SceUID pid, SceProcEventInvokeParam1 *param, int a3) {
	return onExit(pid);
}

static int procKill(SceUID pid, SceProcEventInvokeParam1 *param, int a3) {
	return onExit(pid);
}

static SceUID proc_handler = -1;

void geSgxKernelMemStart(void) {
	static const SceProcEventHandler handler = {
		.size = sizeof(SceProcEventHandler),
		.create = procCreate,
		.exit = procExit,
		.kill = procKill,
	};
	proc_handler = ksceKernelRegisterProcEventHandler("AdrenalineGeSgxMem", &handler, 0);
}

void geSgxKernelMemStop(void) {
	if (proc_handler >= 0)
		ksceKernelUnregisterProcEventHandler(proc_handler);
	proc_handler = -1;
}

int kuGeSgxAllocMemBlock(unsigned int type, unsigned int size, void **base_out, GeSgxKernelMemInfo *info_out) {
	uint32_t state;
	ENTER_SYSCALL(state);

	GeSgxKernelMemInfo info;
	memset(&info, 0, sizeof(info));
	info.size = size;
	void *base = NULL;
	int res = -1;

	SceKernelMemBlockType ktype = 0;
	if (type == SCE_KERNEL_MEMBLOCK_TYPE_USER_CDRAM_RW)
		ktype = SCE_KERNEL_MEMBLOCK_TYPE_KERNEL_CDRAM_RW;
	else if (type == SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE)
		ktype = SCE_KERNEL_MEMBLOCK_TYPE_KERNEL_ROOT_PHYCONT_NC_RW;

	SceUID pid = ksceKernelGetProcessId();
	freeStale(pid);

	int slot = -1, i;
	for (i = 0; i < MAX_BLOCKS; i++) {
		if (blocks[i].kuid <= 0) {
			slot = i;
			break;
		}
	}

	if (!ktype || !size || (size & 0xFFF) || slot < 0) {
		info.kernel_err = slot < 0 ? 0x80020001 : 0x80020000; // no slot / bad arguments
		goto out;
	}

	// Contiguous and not mapped in the kernel first (its address space is small), then contiguous, then any
	SceUID kuid = allocKernel(ktype, size, SCE_KERNEL_ALLOC_MEMBLOCK_ATTR_PHYCONT | SCE_KERNEL_ALLOC_MEMBLOCK_ATTR_DONT_MAP);
	if (kuid < 0) {
		info.kernel_err = kuid;
		kuid = allocKernel(ktype, size, SCE_KERNEL_ALLOC_MEMBLOCK_ATTR_PHYCONT);
	}
	if (kuid < 0)
		kuid = allocKernel(ktype, size, 0);
	if (kuid < 0) {
		info.kernel_err2 = kuid;
		goto out;
	}

	SceKernelPARange pa;
	memset(&pa, 0, sizeof(pa));
	int r = ksceKernelGetMemBlockPARange(kuid, &pa);
	info.pa_err = r < 0 ? r : 0;
	info.paddr = pa.addr;

	SceUID uuid = -1;
	if (r >= 0 && pa.size >= size) {
		uuid = allocUser(pid, type, size, pa.addr, -1);
		if (uuid >= 0)
			info.how = 1;
		else
			info.paddr_err = uuid;
	}
	if (uuid < 0) {
		uuid = allocUser(pid, type, size, 0, kuid);
		if (uuid >= 0)
			info.how = 2;
		else
			info.mirror_err = uuid;
	}
	if (uuid < 0) {
		ksceKernelFreeMemBlock(kuid);
		goto out;
	}

	ksceKernelGetMemBlockBase(uuid, &base);
	blocks[slot].pid = pid;
	blocks[slot].kuid = kuid;
	blocks[slot].uuid = uuid;
	blocks[slot].base = base;
	res = 0;

out:
	if (base_out)
		ksceKernelMemcpyKernelToUser(base_out, &base, sizeof(base));
	if (info_out)
		ksceKernelMemcpyKernelToUser(info_out, &info, sizeof(info));
	EXIT_SYSCALL(state);
	return res;
}

int kuGeSgxFreeMemBlock(void *base) {
	uint32_t state;
	ENTER_SYSCALL(state);

	SceUID pid = ksceKernelGetProcessId();
	int res = -1, i;
	for (i = 0; i < MAX_BLOCKS; i++) {
		if (blocks[i].kuid > 0 && blocks[i].pid == pid && blocks[i].base == base) {
			ksceKernelFreeMemBlock(blocks[i].uuid);
			ksceKernelFreeMemBlock(blocks[i].kuid);
			memset(&blocks[i], 0, sizeof(blocks[i]));
			res = 0;
			break;
		}
	}

	EXIT_SYSCALL(state);
	return res;
}
