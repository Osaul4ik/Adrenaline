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
static int (* _sceGeSetCallback)(PspGeCallbackData *cb);
static int (* _sceDisplaySetFrameBuf)(void *topaddr, int bufferwidth, int pixelformat, int sync);
static u32 (* _sceKernelGetSystemTimeLow)(void);
static u32 (* _sceGeGetCmd)(int cmd);
static int (* _sceGeGetMtx)(int type, void *matrix);
static int (* _sceGeRestoreContext)(const void *context);
static int (* _sceKernelCallSubIntrHandler)(int intr, int sub, int arg2, int arg3);

static volatile GeSgxRing *ring = NULL; // uncached

// Lists queued lately: where the first signal's copy starts (GE_SGX_EVENT_SIGNAL)
#define LIST_STARTS 8
static u32 list_starts[LIST_STARTS];
static u32 list_starts_n = 0;
static u32 ring_mask = 0;
static SceUID ring_blockid = -1;

// The Vita side stopped reading: do not wait for it until it reads again
static int vita_stalled = 0;
static u32 vita_stalled_read = 0;

static volatile GeSgxAck *shared = (volatile GeSgxAck *)GE_SGX_ACK_ADDRESS; // uncached

static inline void memorySync(void) {
	asm volatile ("sync" ::: "memory");
}

// Same as pspSdkDisableInterrupts/EnableInterrupts (Allegrex interrupt
// controller), without importing InterruptManagerForKernel
static inline u32 interruptsOff(void) {
	u32 flags;
	asm volatile ("mfic %0, $0\n\tmtic $0, $0\n\tsync" : "=r"(flags) :: "memory");
	return flags;
}

static inline void interruptsRestore(u32 flags) {
	asm volatile ("mtic %0, $0\n\tsync" :: "r"(flags) : "memory");
}

static void ringPush(u32 type, u32 a, u32 b, u32 c, u32 d) {
	u32 intr = interruptsOff();

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

	interruptsRestore(intr);
}

static u32 timeNow(void) {
	return _sceKernelGetSystemTimeLow ? _sceKernelGetSystemTimeLow() : 0;
}

static void serveVram(void);

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
		// The VRAM pages the walk reads, copied now (the GE idle)
		serveVram();
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

// A list with a context: the GE driver saves the GE's state into it when the list starts and restores it when it ends
static u32 listContext(const PspGeListArgs *arg) {
	return (arg && arg->size >= 8 && arg->context) ? GE_SGX_LIST_CONTEXT : 0;
}

static void noteListStart(const void *list) {
	u32 intr = interruptsOff();
	list_starts[list_starts_n % LIST_STARTS] = (u32)list & 0x0FFFFFFC;
	list_starts_n++;
	interruptsRestore(intr);
}

static int sceGeListEnQueuePatched(const void *list, void *stall, int cbid, PspGeListArgs *arg) {
	int qid = _sceGeListEnQueue(list, stall, cbid, arg);
	if (qid >= 0)
		noteListStart(list);
	if (qid >= 0)
		ringPush(GE_SGX_EVENT_LIST, qid, (u32)list, (u32)stall, listContext(arg));
	return qid;
}

static int sceGeListEnQueueHeadPatched(const void *list, void *stall, int cbid, PspGeListArgs *arg) {
	int qid = _sceGeListEnQueueHead(list, stall, cbid, arg);
	if (qid >= 0)
		noteListStart(list);
	if (qid >= 0)
		ringPush(GE_SGX_EVENT_LIST, qid, (u32)list, (u32)stall, GE_SGX_LIST_HEAD | listContext(arg));
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

/*
	A game polling whether the GE is done (syncType 1) and then writing its
	textures and lists again: the GE may be done before the Vita side has
	read them (the second game's dialog textures, rewritten as text appears,
	were read rewritten). While the Vita side has not read every event, a
	finished GE is answered "drawing" (PSP_GE_LIST_DRAWING), up to the sync
	timeout from the first such answer (then the Vita side counts as
	stalled, as for DrawSync).
*/
#define PSP_GE_LIST_DRAWING_STATE 2
static u32 hold_start = 0, hold_target = 0;
static int holding = 0;

static int vitaBehind(void) {
	u32 target = ring->write;
	if (shared->session != ring->session)
		return 0;
	u32 read = shared->read;
	if ((int)(read - target) >= 0) {
		holding = 0;
		return 0;
	}
	if (vita_stalled) {
		if (read == vita_stalled_read)
			return 0;
		vita_stalled = 0;
	}
	u32 now = timeNow();
	if (!holding || hold_target != target) {
		holding = 1;
		hold_target = target;
		hold_start = now;
	} else if (_sceKernelGetSystemTimeLow && now - hold_start >= GE_SGX_SYNC_TIMEOUT_US) {
		ring->sync_timeouts++;
		vita_stalled = 1;
		vita_stalled_read = read;
		holding = 0;
		return 0;
	}
	return 1;
}

/*
	The GE's state, read while it is idle, for the Vita side
	(GE_SGX_EVENT_STATE): the commands that are plain state (not the list
	flow, the matrix uploads, the CLUT load, the transfer), and the world,
	view, projection and texture matrices. Values sent once, then when they
	change, and all of them every GE_SGX_STATE_FULL_EVERY times.
*/
#define STATE_IDS (0xF0 - 0x10 + 4 * 16)
static u32 state_sent[STATE_IDS];
static u32 state_snapshots = 0;
static int state_full = 1;

static int stateCmd(u32 cmd) {
	if (cmd < 0x10 || cmd >= 0xF0)
		return 0;
	if (cmd == 0x13 || cmd == 0x14) // offset address, origin
		return 0;
	if (cmd >= 0x2A && cmd <= 0x2B) // bone matrix upload
		return 0;
	if (cmd >= 0x3A && cmd <= 0x41) // world, view, projection, texture matrix uploads
		return 0;
	if (cmd == 0xC4 || cmd == 0xEA) // CLUT load, transfer start
		return 0;
	return 1;
}

static u32 pend_id = 0, pend_value = 0;
static int pending = 0;

static void statePut(u32 id, u32 value, u32 slot) {
	value &= 0x00FFFFFF;
	if (!state_full && state_sent[slot] == value)
		return;
	state_sent[slot] = value;
	if (!pending) {
		pend_id = id;
		pend_value = value;
		pending = 1;
	} else {
		ringPush(GE_SGX_EVENT_STATE, pend_id, pend_value, id, value);
		pending = 0;
	}
}

static int state_busy = 0;

static void pushState(void) {
	if (!_sceGeGetCmd)
		return;
	if (shared->session != ring->session)
		return;
	// One thread at a time (the values sent and the pair being built are shared)
	u32 intr = interruptsOff();
	int busy = state_busy;
	state_busy = 1;
	interruptsRestore(intr);
	if (busy)
		return;
	if (++state_snapshots % GE_SGX_STATE_FULL_EVERY == 0)
		state_full = 1;

	int k1 = pspSdkSetK1(0);
	u32 cmd;
	for (cmd = 0x10; cmd < 0xF0; cmd++) {
		if (stateCmd(cmd))
			statePut(cmd, _sceGeGetCmd(cmd), cmd - 0x10);
	}
	if (_sceGeGetMtx) {
		u32 type;
		for (type = 8; type <= 11; type++) {
			u32 m[16];
			u32 n = type == 10 ? 16 : 12, i;
			if (_sceGeGetMtx(type, m) < 0)
				continue;
			for (i = 0; i < n; i++)
				statePut(GE_SGX_STATE_MATRIX + type * 16 + i, m[i], (0xF0 - 0x10) + (type - 8) * 16 + i); // the GE's 24 bits
		}
	}
	pspSdkSetK1(k1);

	if (pending) {
		ringPush(GE_SGX_EVENT_STATE, pend_id, pend_value, pend_id, pend_value);
		pending = 0;
	}
	state_full = 0;
	state_busy = 0;
}

// Nothing queued is left to run (sceGeDrawSync(1) answers "completed")
static int geIdle(void) {
	return _sceGeDrawSync(1) == 0;
}

static int sceGeRestoreContextPatched(const void *context) {
	int res = _sceGeRestoreContext(context);
	if (res >= 0 && geIdle()) {
		state_full = 1;
		pushState();
	}
	return res;
}

static int sceGeDrawSyncPatched(int syncType) {
	int res = _sceGeDrawSync(syncType);
	if (syncType == 0) {
		ringPush(GE_SGX_EVENT_SYNC, (u32)-1, 0, 0, 0);
		if (res >= 0)
			pushState();
		waitForVita();
	} else if (syncType == 1) {
		ring->sync_peeks++;
		if (res == 0 && vitaBehind()) {
			ring->sync_held++;
			return PSP_GE_LIST_DRAWING_STATE;
		}
	}
	return res;
}

static int sceGeListSyncPatched(int qid, int syncType) {
	int res = _sceGeListSync(qid, syncType);
	if (syncType == 0) {
		ringPush(GE_SGX_EVENT_SYNC, (u32)qid, 0, 0, 0);
		if (res >= 0 && geIdle())
			pushState();
		waitForVita();
	} else if (syncType == 1) {
		ring->sync_peeks++;
		if (res == 0 && vitaBehind()) {
			ring->sync_held++;
			return PSP_GE_LIST_DRAWING_STATE;
		}
	}
	return res;
}

/*
	The GE driver calls the game's signal callback from the GE's interrupt
	(sceKernelCallSubIntrHandler(GE, callback * 2, signal, list address after
	the SIGNAL/END), uOFW's _sceGeListInterrupt), with the GE stopped for a
	HANDLER_SUSPEND signal until it returns. Before the game's handler runs,
	the Vita side walks the lists up to that signal (GE_SGX_EVENT_SIGNAL):
	the handler then writes what comes after the signal (The 3rd Birthday:
	its HUD, fog and particles, rewritten at each of its 20 signals), and
	the next signal's walk reads it as the GE did. The waits are short and
	bounded (in the interrupt nothing else runs).
*/
#define PSP_GE_INTR 25
/*
	Time waited in signal handlers since the last flip: at most
	GE_SGX_SIGNAL_FRAME_US a frame (in the interrupt the PSP does nothing
	else: its other interrupts, the DMA's and the audio's, wait too). A wait
	that times out, or a frame's time used up, ends the waits until the
	next flip; the lists are then read when the game waits for the GE.
*/
#if GE_SGX_SIGNAL_WAIT
static u32 signal_frame_us = 0;
#endif

/*
	The commands the GE did since the last signal, copied when the next one
	(or the finish) comes, before the game's handler runs: the Vita side
	walks them as the GE read them, without the PSP waiting for it (the
	waits in the interrupt, 3 ms each, had to be given up most of the time:
	the frame rate fell and most signals were read late anyway). A ring of
	copies (the shadow), each a header word (the event's seq) and the words.
*/
#define SHADOW_MAX_SEGMENT (32 * 1024)
static SceUID shadow_blockid = -1;
static volatile u32 *shadow = NULL; // uncached
static u32 shadow_words = 0, shadow_pos = 0;
static u32 seg_after = 0;           // list address after the last signal (0: none since the list started)

static void allocShadow(void) {
	u32 size;
	// (the progresses copy a frame's commands too: room for more than a frame)
	for (size = 256 * 1024; size >= 32 * 1024; size /= 2) {
		SceUID id = sceKernelAllocPartitionMemory661(PSP_MEMORY_PARTITION_KERNEL, "GeSgxShadow", PSP_SMEM_High, size, NULL);
		if (id < 0)
			continue;
		shadow_blockid = id;
		shadow = (volatile u32 *)((u32)sceKernelGetBlockHeadAddr661(id) | 0x20000000);
		shadow_words = size / 4;
		shadow_pos = 0;
		ring->shadow = (u32)shadow & ~0x20000000;
		ring->shadow_size = size;
		return;
	}
}

static void freeShadow(void) {
	if (shadow_blockid >= 0)
		sceKernelFreePartitionMemory661(shadow_blockid);
	shadow_blockid = -1;
	shadow = NULL;
}

// The start of the segment that ends at after: the last signal's, else the start of the list it is in
static u32 segmentStart(u32 after) {
	if (seg_after && seg_after < after && after - seg_after <= SHADOW_MAX_SEGMENT)
		return seg_after;
	u32 best = 0, i;
	for (i = 0; i < list_starts_n && i < LIST_STARTS; i++) {
		u32 st = list_starts[i];
		if (st <= after && after - st <= SHADOW_MAX_SEGMENT && st > best)
			best = st;
	}
	return best;
}

/*
	A copy record in the shadow: the event's seq, the number of pieces, then
	each piece: its address, its words, the words. The commands the GE did
	since the signal before, and the sublists they call (as deep as
	SHADOW_CALL_DEPTH, each up to its RET, SHADOW_MAX_SUBLIST at most): The
	3rd Birthday's camera, HUD and fog come from sublists its handler writes
	anew, read later they were another frame's (the scene seen from another
	angle).
*/
#define SHADOW_MAX_SUBLIST (4 * 1024)
#define SHADOW_MAX_PIECES 96
#define SHADOW_CALL_DEPTH 2
#define SHADOW_MAX_RECORD 0xFFFF

static u32 rec_off, rec_len, rec_pieces, rec_ok;
static u32 piece_addr[SHADOW_MAX_PIECES];
static u32 call_base, call_offset;

static inline const volatile u32 *uncachedPtr(u32 addr) {
	return (const volatile u32 *)((addr & 0x1FFFFFFF) | 0xA0000000); // uncached: as the GE read it
}

static int validRam(u32 addr, u32 bytes) {
	addr &= 0x0FFFFFFF;
	return addr >= 0x08000000 && addr + bytes <= 0x0A000000 && !(addr & 3);
}

// One piece: copies up to words (to a RET when sublist); its CALLs copied after it, depth permitting
static void addPiece(u32 addr, u32 words, int sublist, int depth) {
	u32 k;
	addr &= 0x0FFFFFFC;
	if (!rec_ok || rec_pieces >= SHADOW_MAX_PIECES || !validRam(addr, 4))
		return;
	for (k = 0; k < rec_pieces; k++) {
		if (piece_addr[k] == addr)
			return; // copied already
	}
	if (sublist) {
		words = SHADOW_MAX_SUBLIST / 4;
		if (!validRam(addr, words * 4))
			words = (0x0A000000 - addr) / 4;
	}
	if (rec_len + 2 + words > SHADOW_MAX_RECORD || rec_off + rec_len + 2 + words > shadow_words) {
		rec_ok = 0;
		return;
	}
	const volatile u32 *src = uncachedPtr(addr);
	u32 head = rec_off + rec_len, n = 0;
	u32 calls[16], ncalls = 0;
	while (n < words) {
		u32 op = src[n];
		shadow[head + 2 + n] = op;
		n++;
		u32 cmd = op >> 24, data = op & 0x00FFFFFF;
		if (cmd == 0x10) {
			call_base = data;
		} else if (cmd == 0x13) {
			call_offset = data << 8;
		} else if (cmd == 0x0A && ncalls < 16) { // CALL
			calls[ncalls++] = ((((call_base & 0x000F0000u) << 8) | (data & 0x00FFFFFC)) + call_offset) & 0x0FFFFFFC;
		} else if (sublist && (cmd == 0x0B || cmd == 0x0C)) { // RET, END
			break;
		}
	}
	shadow[head] = addr;
	shadow[head + 1] = n;
	rec_len += 2 + n;
	piece_addr[rec_pieces++] = addr;
	if (depth < SHADOW_CALL_DEPTH) {
		for (k = 0; k < ncalls; k++)
			addPiece(calls[k], 0, 1, depth + 1);
	}
}

// The copy for an event about to be pushed (its seq: ring->write); 0 if none
static u32 captureSegment(u32 after) {
	if (!shadow || !after)
		return 0;
	after &= 0x0FFFFFFC;
	u32 start = segmentStart(after);
	if (!start || start >= after)
		return 0;
	u32 words = (after - start) / 4;
	// Room for the commands and some sublists from here, else from the shadow's start (sublists past its end are left out)
	u32 need = 2 + 2 + words + 2048;
	if (need > shadow_words)
		return 0;
	if (shadow_pos + need > shadow_words)
		shadow_pos = 0;
	rec_off = shadow_pos;
	if (rec_off > 0xFFFF)
		return 0;
	rec_len = 2;
	rec_pieces = 0;
	rec_ok = 1;
	call_base = _sceGeGetCmd ? (_sceGeGetCmd(0x10) & 0x00FFFFFF) : 0;
	call_offset = _sceGeGetCmd ? ((_sceGeGetCmd(0x13) & 0x00FFFFFF) << 8) : 0;
	addPiece(start, words, 0, 0);
	if (!rec_pieces)
		return 0;
	rec_ok = 1;
	shadow[rec_off] = ring->write;
	shadow[rec_off + 1] = rec_pieces;
	shadow_pos = rec_off + rec_len;
	return (rec_len << 16) | rec_off;
}

/*
	The GE's progress (GE_SGX_EVENT_PROGRESS), from its registers (uOFW
	hardware.h): EXEC (bit 0 running, 0x300 the call depth), LISTADDR the
	command it is at, RADR1 the address after the top list's CALL when in a
	sublist. A timer interrupt every GE_SGX_PROGRESS_US.
*/
#define HW_GE_EXEC     (*(volatile u32 *)0xBD400100)
#define HW_GE_LISTADDR (*(volatile u32 *)0xBD400108)
#define HW_GE_RADR1    (*(volatile u32 *)0xBD400110)

static SceUID (* _sceKernelSetAlarm)(SceUInt clock, SceUInt (*handler)(void *common), void *common) = NULL;
static SceUID progress_alarm = -1;
static u32 progress_last = 0;

static SceUInt progressAlarm(void *common) {
	if (!ring || !shadow || shared->session != ring->session)
		return GE_SGX_PROGRESS_US;
	u32 exec = HW_GE_EXEC;
	if (!(exec & 1))
		return GE_SGX_PROGRESS_US;
	u32 at = ((exec & 0x300) ? HW_GE_RADR1 - 4 : HW_GE_LISTADDR) & 0x0FFFFFFC;
	if (at == progress_last || !validRam(at, 4))
		return GE_SGX_PROGRESS_US;
	u32 intr_flags = interruptsOff();
	// Only with the copy: what the Vita side walks up to is what the GE read
	u32 copy = captureSegment(at);
	if (copy) {
		ringPush(GE_SGX_EVENT_PROGRESS, at, exec, 0, copy);
		seg_after = at;
		progress_last = at;
		ring->progress_events++;
	}
	interruptsRestore(intr_flags);
	return GE_SGX_PROGRESS_US;
}

static void startProgress(void) {
	if (progress_alarm >= 0 || !_sceKernelSetAlarm)
		return;
	progress_last = 0;
	progress_alarm = _sceKernelSetAlarm(GE_SGX_PROGRESS_US, progressAlarm, NULL);
	log("GeSgx: GE progress every %d us: %08X\n", GE_SGX_PROGRESS_US, (unsigned int)progress_alarm);
}

#if GE_SGX_SIGNAL_WAIT
static void waitForVitaSignal(void) {
	u32 target = ring->write;
	if (shared->session != ring->session || !_sceKernelGetSystemTimeLow)
		return;
	u32 read = shared->read;
	if ((int)(read - target) >= 0)
		return;
	if (signal_frame_us >= GE_SGX_SIGNAL_FRAME_US) {
		ring->signal_skipped++;
		return;
	}
	u32 limit = GE_SGX_SIGNAL_FRAME_US - signal_frame_us;
	if (limit > GE_SGX_SIGNAL_TIMEOUT_US)
		limit = GE_SGX_SIGNAL_TIMEOUT_US;

	u32 start = timeNow(), waited = 0;
	ring->signal_waits++;
	while (1) {
		read = shared->read;
		if ((int)(read - target) >= 0)
			break;
		waited = timeNow() - start;
		if (waited >= limit) {
			ring->signal_timeouts++;
			signal_frame_us = GE_SGX_SIGNAL_FRAME_US;
			break;
		}
	}
	signal_frame_us += waited;
	ring->signal_wait_us += waited;
	if (waited > ring->signal_max_us)
		ring->signal_max_us = waited;
}
#endif

static int sceKernelCallSubIntrHandlerPatched(int intr, int sub, int arg2, int arg3) {
	/*
		Signal callbacks (even) and finish callbacks (odd, c = 1): reported
		with the copy of the commands the GE did since the signal before
		(or the list's start), taken before the game's handler runs
	*/
	if (intr == PSP_GE_INTR && ring) {
		u32 intr_flags = interruptsOff();
		u32 copy = captureSegment((u32)arg3);
		ringPush(GE_SGX_EVENT_SIGNAL, (u32)arg2 & 0xFFFF, (u32)arg3, (u32)(sub & 1), copy);
		interruptsRestore(intr_flags);
		seg_after = (sub & 1) ? 0 : ((u32)arg3 & 0x0FFFFFFC);
#if GE_SGX_SIGNAL_WAIT
		waitForVitaSignal();
#endif
	}
	return _sceKernelCallSubIntrHandler(intr, sub, arg2, arg3);
}

// Counted only: games that learn from the GE's interrupt that a list is done
static int sceGeSetCallbackPatched(PspGeCallbackData *cb) {
	ring->callbacks++;
	return _sceGeSetCallback(cb);
}

/*
	VRAM into pspemu's window for it (see GE_SGX_VRAM_SHADOW), at every flip:
	the pages the Vita side reads (GeSgxVramWant), each run in one copy.
	With the DMA: sceDmacMemcpy is looked up in every loaded module (its
	library "sceDmac" is not where the name says), else the CPU (~120 ms).
*/
static int (* _sceDmacMemcpy)(void *dst, const void *src, SceSize size) = NULL;

// The Vita side's last VRAM copy request done, and whether one was since the last flip
static u32 vram_served = 0;
static int vram_served_since_flip = 0;

static void copyVram(void) {
	volatile GeSgxVramWant *want = (volatile GeSgxVramWant *)GE_SGX_VRAM_WANT_ADDRESS; // uncached
	if (shared->session != ring->session)
		return;

	int k1 = pspSdkSetK1(0); // kernel addresses (the flip comes from the game's syscall)
	u32 start = timeNow(), pages = 0;
	u32 p = 0;
	while (p < GE_SGX_VRAM_PAGES) {
		// A run of wanted pages, one copy
		if (!(want->bits[p / 32] & (1u << (p % 32)))) {
			p++;
			continue;
		}
		u32 first = p;
		while (p < GE_SGX_VRAM_PAGES && (want->bits[p / 32] & (1u << (p % 32))))
			p++;
		u32 off = first * GE_SGX_VRAM_PAGE, size = (p - first) * GE_SGX_VRAM_PAGE;
		if (_sceDmacMemcpy)
			_sceDmacMemcpy((void *)(GE_SGX_VRAM_SHADOW + off), (const void *)(0x44000000 + off), size);
		else
			memcpy((void *)(GE_SGX_VRAM_SHADOW + off), (const void *)(0x44000000 + off), size); // uncached both
		pages += p - first;
	}
	u32 took = timeNow() - start;
	pspSdkSetK1(k1);
	if (!pages)
		return;
	ring->vram_copies++;
	ring->vram_copy_pages += pages;
	ring->vram_copy_us += took;
	if (took > ring->vram_copy_max_us)
		ring->vram_copy_max_us = took;
}

// A copy the Vita side asked for (GeSgxAck.vram_req), while the game waits for it
static void serveVram(void) {
	u32 req = shared->vram_req;
	if (req == vram_served || shared->session != ring->session)
		return;
	copyVram();
	vram_served = req;
	vram_served_since_flip = 1;
	memorySync();
	ring->vram_done = req;
	memorySync();
}

// An export of any loaded module, by library name and NID; the module's name into *modname
static u32 findExport(const char *libname, u32 nid, const char **modname) {
	int (* getIdList)(SceUID *readbuf, int readbufsize, int *idcount) =
		(void *)FindProc("sceModuleManager", "ModuleMgrForKernel", 0x644395E2);
	if (!getIdList)
		return 0;

	static SceUID ids[128];
	int count = 0;
	if (getIdList(ids, sizeof(ids), &count) < 0)
		return 0;
	if (count > 128)
		count = 128;

	int m;
	for (m = 0; m < count; m++) {
		SceModule2 *mod = sceKernelFindModuleByUID661(ids[m]);
		if (!mod)
			continue;
		int i = 0;
		while (i < mod->ent_size) {
			SceLibraryEntryTable *entry = (SceLibraryEntryTable *)(mod->ent_top + i);
			if (entry->libname && strcmp(entry->libname, libname) == 0) {
				u32 *table = entry->entrytable;
				int total = entry->stubcount + entry->vstubcount;
				int j;
				for (j = 0; j < total; j++) {
					if (table[j] == nid) {
						*modname = mod->modname;
						return table[j + total];
					}
				}
			}
			i += entry->len * 4;
		}
	}
	return 0;
}

/*
	Registers games set once (sceGuInit, sceGuDepthBuffer, sceGuOffset...):
	the GE's values, for the ones the Vita side's walk has not seen. In
	pairs: depth buffer, framebuffer, drawing offset, region.
*/
static const u8 hw_regs[] = { 0x9E, 0x9F, 0x9C, 0x9D, 0x4C, 0x4D, 0x15, 0x16 };

static void pushRegs(void) {
	if (!_sceGeGetCmd)
		return;
	u32 i;
	for (i = 0; i + 1 < sizeof(hw_regs); i += 2)
		ringPush(GE_SGX_EVENT_REGS, hw_regs[i], _sceGeGetCmd(hw_regs[i]) & 0x00FFFFFF, hw_regs[i + 1], _sceGeGetCmd(hw_regs[i + 1]) & 0x00FFFFFF);
}

static int sceDisplaySetFrameBufPatched(void *topaddr, int bufferwidth, int pixelformat, int sync) {
	// Copied at the waits this frame: not again (the walk at the flip, without a wait, reads the copy as it is)
	if (!vram_served_since_flip)
		copyVram();
	vram_served_since_flip = 0;
	pushRegs();
#if GE_SGX_SIGNAL_WAIT
	signal_frame_us = 0;
#endif
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
	_sceGeSetCallback = (void *)FindProc("sceGE_Manager", "sceGe_driver", 0xA4FC06A4);
	_sceDisplaySetFrameBuf = (void *)FindProc("sceDisplay_Service", "sceDisplay_driver", 0x289D82FE);
	_sceKernelGetSystemTimeLow = (void *)FindProc("sceThreadManager", "ThreadManForKernel", 0x369ED59D);
	_sceKernelSetAlarm = (void *)FindProc("sceThreadManager", "ThreadManForKernel", 0x6652B8CA);
	_sceGeGetCmd = (void *)FindProc("sceGE_Manager", "sceGe_driver", 0xDC93CFEF);
	_sceGeGetMtx = (void *)FindProc("sceGE_Manager", "sceGe_driver", 0x57C8945B);
	_sceGeRestoreContext = (void *)FindProc("sceGE_Manager", "sceGe_driver", 0x0BF608FB);
	_sceKernelCallSubIntrHandler = (void *)FindProc("sceInterruptManager", "InterruptManagerForKernel", 0x0C5F7AE3);
	u32 subintr_stub = sctrlHENFindImport("sceGE_Manager", "InterruptManagerForKernel", 0x0C5F7AE3);
	if (!subintr_stub)
		subintr_stub = sctrlHENFindImport("sceGE_Manager", "InterruptManagerForKernel", 0xCDC86B64);
	// VRAM copied with the DMA (1) or the CPU (2)
	const char *dmac_module = "";
	_sceDmacMemcpy = (void *)findExport("sceDmac", 0x617F3FE6, &dmac_module);
	u32 how = _sceDmacMemcpy ? 1 : 2;
	log("GeSgx: VRAM copied with %d, GE registers %s, matrices %s, context restore %s\n", (int)how, _sceGeGetCmd ? "read" : "not found",
		_sceGeGetMtx ? "read" : "not found", _sceGeRestoreContext ? "hooked" : "not found");
	log("GeSgx: GE signal callbacks %s\n", (_sceKernelCallSubIntrHandler && subintr_stub) ? "hooked" : "not found");

	if (!_sceGeListEnQueue || !_sceGeListEnQueueHead || !_sceGeListDeQueue || !_sceGeListUpdateStallAddr ||
	    !_sceGeDrawSync || !_sceGeListSync || !_sceDisplaySetFrameBuf) {
		log("GeSgx: GE/display functions not found\n");
		return -1;
	}

	if (!allocRing()) {
		log("GeSgx: no memory for the ring\n");
		return -1;
	}

	ring->vram_copy_how = how;
	strncpy((char *)ring->vram_copy_module, dmac_module, sizeof(ring->vram_copy_module) - 1);
	allocShadow();
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
		freeShadow();
		freeRing();
		return 0;
	}

	vita_stalled = 0;
	vram_served = 0; // the Vita side's requests start over with the ring
	vram_served_since_flip = 0;
	ring->vram_done = 0;
	state_full = 1;
	state_snapshots = 0;
	pending = 0;
	state_busy = 0;

	sctrlHENPatchSyscall((u32)_sceGeListEnQueue, sceGeListEnQueuePatched);
	sctrlHENPatchSyscall((u32)_sceGeListEnQueueHead, sceGeListEnQueueHeadPatched);
	sctrlHENPatchSyscall((u32)_sceGeListDeQueue, sceGeListDeQueuePatched);
	sctrlHENPatchSyscall((u32)_sceGeListUpdateStallAddr, sceGeListUpdateStallAddrPatched);
	sctrlHENPatchSyscall((u32)_sceGeDrawSync, sceGeDrawSyncPatched);
	sctrlHENPatchSyscall((u32)_sceGeListSync, sceGeListSyncPatched);
	if (_sceGeSetCallback)
		sctrlHENPatchSyscall((u32)_sceGeSetCallback, sceGeSetCallbackPatched);
	if (_sceGeRestoreContext)
		sctrlHENPatchSyscall((u32)_sceGeRestoreContext, sceGeRestoreContextPatched);
	seg_after = 0;
	list_starts_n = 0;
	if (_sceKernelCallSubIntrHandler && subintr_stub)
		MAKE_JUMP(subintr_stub, sceKernelCallSubIntrHandlerPatched);
	sctrlHENPatchSyscall((u32)_sceDisplaySetFrameBuf, sceDisplaySetFrameBufPatched);
	ClearCaches();
	startProgress();

	log("GeSgx: enabled\n");
	return 1;
}
