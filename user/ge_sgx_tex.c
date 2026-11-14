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
	SGX renderer: texture cache. Runs in the walker thread, while the lists
	are walked (before DrawSync returns to the game), so a texture is read
	as the GE read it.

	A cached texture is checked once per frame with a sampled hash of its
	memory (64 words plus both ends: the PSP RAM is uncached, ~125 ns per
	read). When it changed, a new entry is made: entries never change while
	a published frame may still use them, and are freed only some frames
	after they were last used (the GPU may still be reading them).
*/

#include <psp2/kernel/dmac.h>
#include <psp2/kernel/processmgr.h>
#include <psp2/kernel/sysmem.h>

#include <string.h>

#include "ge_sgx.h"
#include "ge_sgx_tex.h"
#include "ge_sgx_texconv.h"

#define MAX_TEXTURES 512
#define BUDGET_BYTES (10 * 1024 * 1024) // 12 MB left too little RAM next to the framebuffer renderings
// Lowered when memory runs out first
static uint32_t budget = BUDGET_BYTES;
#define STAGING_BYTES (1024 * 1024)
// RAM always left to the emulator
#define RESERVE_RAM_KB (2 * 1024)

// Frames an entry is kept after its last use: published frames in flight plus the GPU
#define KEEP_FRAMES 6

typedef struct {
	int used;   // slot taken (live or waiting to be freed)
	int live;   // found by lookups
	GeSgxTexDesc key;
	uint32_t sample;
	uint32_t last_used;
	uint32_t last_checked;
	uint32_t dead_since;
	uint32_t dead_seq; // freed once the GPU is done with this frame (the one being built when it died)
	SceUID mem;
	uint32_t bytes;
	SceGxmTexture tex;
} Entry;

static Entry entries[MAX_TEXTURES];
static int last_hit = -1;
static uint32_t cur_frame = 1;
static uint32_t total_bytes = 0;

static SceUID staging_mem = -1;
static uint8_t *staging[2];
static uint32_t clut_rgba[GE_SGX_CLUT_ENTRIES];

static GeSgxTexStats stats;

int geSgxTexInit(void) {
	if (staging_mem >= 0)
		return 1;
	staging_mem = sceKernelAllocMemBlock("SceGeSgxTexStaging", SCE_KERNEL_MEMBLOCK_TYPE_USER_RW, STAGING_BYTES * 2, NULL);
	if (staging_mem < 0)
		return 0;
	void *base = NULL;
	sceKernelGetMemBlockBase(staging_mem, &base);
	staging[0] = (uint8_t *)base;
	staging[1] = staging[0] + STAGING_BYTES;
	return 1;
}

static void freeMemory(Entry *e) {
	if (e->mem >= 0) {
		void *base = NULL;
		sceKernelGetMemBlockBase(e->mem, &base);
		if (base)
			sceGxmUnmapMemory(base);
		sceKernelFreeMemBlock(e->mem);
		total_bytes -= e->bytes;
	}
	e->mem = -1;
	e->bytes = 0;
}

static void kill(Entry *e) {
	if (e->live) {
		e->live = 0;
		e->dead_since = cur_frame;
		e->dead_seq = GeSgxBuildingSeq();
	}
}

void geSgxTexReset(void) {
	// Called when the frames in flight are dropped: entries may still be in
	// use by the GPU, so they only stop being found
	int i;
	for (i = 0; i < MAX_TEXTURES; i++)
		kill(&entries[i]);
	last_hit = -1;
}

static int sameKey(const GeSgxTexDesc *a, const GeSgxTexDesc *b) {
	return a->addr == b->addr && a->bufw == b->bufw && a->w == b->w && a->h == b->h &&
		a->format == b->format && a->swizzle == b->swizzle &&
		a->clut_format == b->clut_format && a->clut_hash == b->clut_hash;
}

static uint32_t sourceBytes(const GeSgxTexDesc *d) {
	return geSgxTexRowBytes(d->format, d->bufw) * geSgxTexRows(d);
}

// Hash of 64 words spread over the texture and of both ends
static uint32_t sampleHash(const uint8_t *src, uint32_t size) {
	uint32_t h = 2166136261u ^ size;
	uint32_t i, step = (size / 64) & ~3u;
	uint32_t v;

	for (i = 0; i < 64 && step; i++) {
		memcpy(&v, src + i * step, 4);
		h = (h ^ v) * 16777619u;
	}
	if (size >= 16) {
		uint32_t ends[8];
		memcpy(ends, src, 16);
		memcpy(ends + 4, src + ((size - 16) & ~3u), 16);
		for (i = 0; i < 8; i++)
			h = (h ^ ends[i]) * 16777619u;
	}
	return h;
}

static SceGxmTextureFormat gxmFormat(uint32_t format) {
	switch (format) {
		case GE_TFMT_5650: return SCE_GXM_TEXTURE_FORMAT_U5U6U5_BGR;
		case GE_TFMT_5551: return SCE_GXM_TEXTURE_FORMAT_U1U5U5U5_ABGR;
		case GE_TFMT_4444: return SCE_GXM_TEXTURE_FORMAT_U4U4U4U4_ABGR;
		default:           return SCE_GXM_TEXTURE_FORMAT_U8U8U8U8_ABGR;
	}
}

// The least recently used live entry not used for KEEP_FRAMES, NULL if none
static Entry *oldestLive(void) {
	Entry *oldest = NULL;
	int i;
	for (i = 0; i < MAX_TEXTURES; i++) {
		Entry *e = &entries[i];
		if (e->live && cur_frame - e->last_used >= KEEP_FRAMES && (!oldest || e->last_used < oldest->last_used))
			oldest = e;
	}
	return oldest;
}

// Frees the dead entries the GPU is done with; gives up live ones, oldest first, until need bytes fit
static void makeRoom(uint32_t need) {
	int i;
	uint32_t dying = 0;

	for (i = 0; i < MAX_TEXTURES; i++) {
		Entry *e = &entries[i];
		if (e->used && !e->live) {
			if ((int32_t)(ge_sgx_gpu_done_seq - e->dead_seq) >= 0) {
				freeMemory(e);
				e->used = 0;
			} else {
				dying += e->bytes;
			}
		}
	}

	// Their memory is freed once the GPU is done with them; enough are given up for need now
	while (total_bytes - dying + need > budget) {
		Entry *oldest = oldestLive();
		if (!oldest)
			break;
		kill(oldest);
		dying += oldest->bytes;
	}
}

// The first failures, for the log: why a texture could not be used
static int fail_logs = 0;
static int failed(const GeSgxTexDesc *d, const char *why) {
	stats.failed++;
	if (fail_logs < 16) {
		fail_logs++;
		GeSgxLog("GeSgx texture failed (%s): addr %08X fmt %u %ux%u bufw %u swizzle %u | cache %u KB\n", why,
			(unsigned)d->addr, (unsigned)d->format, (unsigned)d->w, (unsigned)d->h, (unsigned)d->bufw, (unsigned)d->swizzle, (unsigned)(total_bytes / 1024));
	}
	return -1;
}

static int32_t upload(const GeSgxTexDesc *d, uint32_t sample) {
	uint32_t rowbytes = geSgxTexRowBytes(d->format, d->bufw);
	uint32_t rows = geSgxTexRows(d);
	uint32_t size = rowbytes * rows;
	// A buffer narrower than the texture (480 wide pictures in 512 textures) is fine:
	// the texels past it are not used
	if (!rowbytes || !d->bufw || size > STAGING_BYTES || d->w > 512 || d->h > 512)
		return failed(d, "size");

	const uint8_t *src = geSgxPspPtr(d->addr, size);
	if (!src)
		return failed(d, "address");

	// CLUT4/8: indices (P8) and a 256 entry palette after them (64 byte aligned)
	int indexed = geSgxTexIndexed(d->format);
	uint32_t stride = indexed ? ((d->w + 7u) & ~7u) : geSgxTexOutStride(d);
	uint32_t pal_off = (stride * d->h + 63u) & ~63u;
	uint32_t data_bytes = indexed ? pal_off + 1024 : stride * d->h;
	uint32_t bytes = (data_bytes + 0xFFF) & ~0xFFFu;
	if (data_bytes > STAGING_BYTES)
		return failed(d, "size");

	makeRoom(bytes);
	if (total_bytes + bytes > budget)
		return failed(d, "budget");

	// The emulator keeps a reserve of RAM: older textures are given up for this one, which waits
	if (GeSgxFreeKB(SCE_KERNEL_MEMBLOCK_TYPE_USER_RW) < bytes / 1024 + RESERVE_RAM_KB) {
		uint32_t saved = budget;
		budget = total_bytes > bytes ? total_bytes - bytes : 0;
		makeRoom(bytes);
		budget = saved;
		return failed(d, "memory reserve");
	}

	int slot = -1, i;
	for (i = 0; i < MAX_TEXTURES; i++) {
		if (!entries[i].used) {
			slot = i;
			break;
		}
	}
	if (slot < 0) {
		// All entries taken: the oldest one is given up, its slot is free once the GPU is done with it
		Entry *oldest = oldestLive();
		if (oldest)
			kill(oldest);
		return failed(d, "entries");
	}

	SceUInt64 t0 = sceKernelGetProcessTimeWide();

	// PSP memory is uncached: one DMA for big textures, bursts for small ones
	if (size >= 16 * 1024)
		sceDmacMemcpy(staging[0], src, size);
	else
		memcpy(staging[0], src, size);

	const uint8_t *linear = staging[0];
	if (d->swizzle && d->format < GE_TFMT_DXT1) {
		geSgxUnswizzle(staging[1], staging[0], rowbytes, rows);
		linear = staging[1];
	}

	if (d->format >= GE_TFMT_CLUT4 && d->format <= GE_TFMT_CLUT32)
		geSgxClutDecode(d->clut_format, d->clut, d->clut_bytes, clut_rgba);

	Entry *e = &entries[slot];
	memset(e, 0, sizeof(*e));
	e->mem = sceKernelAllocMemBlock("SceGeSgxTexture", SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE, bytes, NULL);
	if (e->mem < 0) {
		// Out of memory before the budget: the budget comes down to what fits, then one more try
		budget = total_bytes > 1024 * 1024 ? total_bytes - 512 * 1024 : total_bytes;
		makeRoom(bytes);
		e->mem = total_bytes + bytes <= budget ? sceKernelAllocMemBlock("SceGeSgxTexture", SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE, bytes, NULL) : -1;
	}
	if (e->mem < 0) {
		e->mem = -1;
		return failed(d, "memory");
	}
	void *gpu = NULL;
	sceKernelGetMemBlockBase(e->mem, &gpu);
	if (sceGxmMapMemory(gpu, bytes, SCE_GXM_MEMORY_ATTRIB_READ) < 0) {
		sceKernelFreeMemBlock(e->mem);
		e->mem = -1;
		return failed(d, "map");
	}
	e->bytes = bytes;
	total_bytes += bytes;

	// Converted in cached memory, then one DMA into the GPU's (uncached) memory
	uint8_t *out = linear == staging[0] ? staging[1] : staging[0];
	int ok;
	if (indexed) {
		geSgxTexPalette(d, clut_rgba, (uint32_t *)(out + pal_off));
		ok = geSgxTexConvertIndices(d, linear, rowbytes, out, stride) &&
			sceGxmTextureInitLinear(&e->tex, gpu, SCE_GXM_TEXTURE_FORMAT_P8_ABGR, d->w, d->h, 0) >= 0 &&
			sceGxmTextureSetPalette(&e->tex, (uint8_t *)gpu + pal_off) >= 0;
	} else {
		if (d->bufw < d->w)
			memset(out, 0, stride * d->h);
		ok = geSgxTexConvert(d, linear, rowbytes, out, stride, clut_rgba) &&
			sceGxmTextureInitLinear(&e->tex, gpu, gxmFormat(d->format), d->w, d->h, 0) >= 0;
	}
	if (!ok) {
		freeMemory(e);
		return failed(d, "convert");
	}
	sceDmacMemcpy(gpu, out, data_bytes);

	e->key = *d;
	e->key.clut = NULL;
	e->sample = sample;
	e->last_used = cur_frame;
	e->last_checked = cur_frame;
	e->used = 1;
	__sync_synchronize();
	e->live = 1;

	stats.uploads++;
	stats.upload_kb += size / 1024;
	stats.upload_us += (uint32_t)(sceKernelGetProcessTimeWide() - t0);
	return slot;
}

/*
	Lookups are many (each state change): the recently found entries are
	tried before the whole table, and a texture that failed in this frame
	is not tried again in it.
*/
#define RECENT 16
static int recent[RECENT];
static int recent_next = 0;
static struct { GeSgxTexDesc key; uint32_t frame; } bad[RECENT];
static int bad_next = 0;

static int32_t lookupFailed(const GeSgxTexDesc *d) {
	bad[bad_next].key = *d;
	bad[bad_next].key.clut = NULL;
	bad[bad_next].frame = cur_frame + 1; // 0: unused
	bad_next = (bad_next + 1) % RECENT;
	return -1;
}

int32_t geSgxTexLookup(void *user, const GeSgxTexDesc *d) {
	(void)user;

	if (!geSgxTexRowBytes(d->format, d->bufw))
		return failed(d, "format");

	int found = -1, i;
	for (i = 0; i < RECENT; i++) {
		if (bad[i].frame == cur_frame + 1 && sameKey(&bad[i].key, d)) {
			stats.failed++;
			return -1;
		}
	}
	if (last_hit >= 0 && entries[last_hit].live && sameKey(&entries[last_hit].key, d)) {
		found = last_hit;
	} else {
		for (i = 0; i < RECENT && found < 0; i++) {
			int r = recent[i];
			if (r >= 0 && r < MAX_TEXTURES && entries[r].live && sameKey(&entries[r].key, d))
				found = r;
		}
		for (i = 0; i < MAX_TEXTURES && found < 0; i++) {
			if (entries[i].live && sameKey(&entries[i].key, d))
				found = i;
		}
		if (found >= 0) {
			recent[recent_next] = found;
			recent_next = (recent_next + 1) % RECENT;
		}
	}

	if (found >= 0) {
		Entry *e = &entries[found];
		if (e->last_checked == cur_frame) {
			e->last_used = cur_frame;
			last_hit = found;
			return found;
		}

		// First use in this frame: still the same memory contents?
		uint32_t size = sourceBytes(d);
		const uint8_t *src = geSgxPspPtr(d->addr, size);
		uint32_t sample = src ? sampleHash(src, size) : 0;
		stats.checks++;
		if (src && sample == e->sample) {
			e->last_checked = cur_frame;
			e->last_used = cur_frame;
			last_hit = found;
			return found;
		}
		kill(e);
	}

	uint32_t size = sourceBytes(d);
	const uint8_t *src = geSgxPspPtr(d->addr, size);
	if (!src) {
		failed(d, "address");
		return lookupFailed(d);
	}

	int32_t handle = upload(d, sampleHash(src, size));
	if (handle < 0)
		return lookupFailed(d);
	last_hit = handle;
	recent[recent_next] = handle;
	recent_next = (recent_next + 1) % RECENT;
	return handle;
}

void geSgxTexFrameDone(void) {
	cur_frame++;
	// A budget lowered when memory ran out comes back up while there is memory again
	if ((cur_frame & 63) == 0 && budget < BUDGET_BYTES &&
	    GeSgxFreeKB(SCE_KERNEL_MEMBLOCK_TYPE_USER_RW) >= RESERVE_RAM_KB + 2048) {
		budget += 512 * 1024;
		if (budget > BUDGET_BYTES)
			budget = BUDGET_BYTES;
	}
}

void geSgxTexStats(GeSgxTexStats *out) {
	int i;
	uint32_t n = 0;
	for (i = 0; i < MAX_TEXTURES; i++) {
		if (entries[i].live)
			n++;
	}
	stats.entries = n;
	stats.kb = total_bytes / 1024;
	*out = stats;
	stats.uploads = stats.upload_kb = stats.upload_us = stats.checks = stats.failed = 0;
}

const SceGxmTexture *geSgxTexGet(int32_t handle) {
	if (handle < 0 || handle >= MAX_TEXTURES || !entries[handle].used)
		return NULL;
	return &entries[handle].tex;
}
