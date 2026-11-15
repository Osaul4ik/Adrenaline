/*
  Memory for the SGX renderer from adrenaline_kernel (kernel/gesgx_mem.c):
  blocks outside the emulator's budget, mapped into the calling process.
  Shared by adrenaline_kernel and adrenaline_user.
*/

#ifndef __GE_SGX_KERNEL_H__
#define __GE_SGX_KERNEL_H__

typedef struct {
	unsigned int size;
	int how;          // 1 mapped at the physical address, 2 as a mirror, 0 failed
	int kernel_err;   // kernel block, physically contiguous (0: it was)
	int kernel_err2;  // kernel block, any pages (when not contiguous either)
	int pa_err;       // its physical address (contiguous only)
	unsigned int paddr;
	int paddr_err;    // user block at the physical address
	int mirror_err;   // user block mirroring the kernel block
} GeSgxKernelMemInfo;

/*
  type: SCE_KERNEL_MEMBLOCK_TYPE_USER_CDRAM_RW or SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE,
  size a multiple of 4 KB. 0 and the block's address in *base, < 0 else; *info says
  how it went either way.
*/
int kuGeSgxAllocMemBlock(unsigned int type, unsigned int size, void **base, GeSgxKernelMemInfo *info);
int kuGeSgxFreeMemBlock(void *base);

#endif
