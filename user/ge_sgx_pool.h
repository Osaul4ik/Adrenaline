#ifndef __GE_SGX_POOL_H__
#define __GE_SGX_POOL_H__

#include <stdint.h>

// SGX renderer: GPU memory outside the emulator's budget (see ge_sgx_pool.c)

uint32_t geSgxPoolInit(void);    // once; its size in bytes, 0 without one
uint32_t geSgxPoolBytes(void);
uint32_t geSgxPoolFreeKB(void);
int geSgxPoolIsCdram(void);
void *geSgxPoolAlloc(uint32_t bytes); // 4 KB pages, already mapped for the GPU (read and write); NULL if none fit
void geSgxPoolFree(void *mem, uint32_t bytes);

#endif
