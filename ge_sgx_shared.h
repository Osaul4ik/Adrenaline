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
	SGX renderer: data shared by systemctrl (PSP side) and adrenaline_user
	(Vita side).

	The PSP GE keeps executing every display list itself, so VRAM, interrupts
	and list synchronisation stay exactly as on real hardware. systemctrl
	reports what the game hands to the GE (lists, stall updates, display
	flips) as events in a ring that the Vita side reads from PSP RAM, so the
	same lists can be rendered again on the Vita's GPU.

	Direction of the data:
	- the ring (events, write counter, PSP statistics) is written only by the
	  PSP side, with uncached stores; it lives in the PSP kernel partition and
	  its address is given to the Vita side with ADRENALINE_VITA_CMD_GE_SGX_START.
	- the Vita side never writes into the ring: its read counter goes into a
	  cache line of its own at the end of the SceAdrenaline block (GeSgxAck),
	  which stays valid across PSP reboots. So a stale ring after a game exit
	  is only ever read, and no field of the PSP side shares the line.
*/

#ifndef __GE_SGX_SHARED_H__
#define __GE_SGX_SHARED_H__

#define GE_SGX_MAGIC   0x58475347 // 'GSGX'
#define GE_SGX_VERSION 3 // the ring's layout: both sides from the same build

#define GE_SGX_RING_ENTRIES 4096 // power of two
#define GE_SGX_RING_MASK (GE_SGX_RING_ENTRIES - 1)

enum GeSgxEventTypes {
	GE_SGX_EVENT_NONE,
	GE_SGX_EVENT_LIST,      // a = queue id, b = list address, c = stall address (0: none), d = GE_SGX_LIST_* flags
	GE_SGX_EVENT_STALL,     // a = queue id, b = new stall address
	GE_SGX_EVENT_FLIP,      // a = top address, b = buffer width, c = pixel format, d = sync
	GE_SGX_EVENT_DEQUEUE,   // a = queue id
	/*
		GE registers as the GE has them (sceGeGetCmd), at a flip: a = command,
		b = its value (24 bits), c = another command, d = its value. The Vita
		side takes them for the registers its walk never saw set (games set
		the depth buffer once, in a list before the walker's state starts).
	*/
	GE_SGX_EVENT_REGS,
	/*
		The game waited for the GE (sceGeDrawSync / sceGeListSync with 0,
		after it returned): the GE is done, the lists' memory and what they
		use are as the GE read them. a = queue id (-1: all lists)
	*/
	GE_SGX_EVENT_SYNC,
	/*
		The GE's state as the GE has it, read while it is idle (after a
		sceGeDrawSync(0), a sceGeListSync(0) that left nothing running, a
		sceGeRestoreContext): a = id, b = its value (24 bits), c = id, d =
		value. Ids below 0x100 are commands (sceGeGetCmd); 0x100 + type * 16
		+ index a matrix value (sceGeGetMtx, type 8 world, 9 view, 10
		projection, 11 texture; 24 bits, as the GE keeps it).
		The Vita side walks the lists the GE has done, then takes these over
		its own: the GE's state is what the next lists start from (state the
		walk never saw change: a sceGeRestoreContext, a list of the kernel's).
		Only the values that changed since the last ones sent, and all of
		them every GE_SGX_STATE_FULL_EVERY times.
	*/
	GE_SGX_EVENT_STATE,
	/*
		The GE stopped at a signal whose handler is about to run (the GE
		driver's call of the game's signal callback, sceKernelCallSubIntrHandler):
		a = the signal's 16 bits, b = the list address after its SIGNAL/END
		(0: not given). The PSP side waits (GE_SGX_SIGNAL_TIMEOUT_US at most)
		until the Vita side has walked the lists up to that signal, then lets
		the handler run: what the handler writes for the commands after the
		signal is read after it ran, as the GE reads it.
	*/
	GE_SGX_EVENT_SIGNAL,
};

#define GE_SGX_LIST_HEAD    1 // queued at the head
#define GE_SGX_LIST_CONTEXT 2 // queued with a context (PspGeListArgs): the GE's state is saved at its start, restored at its end

#define GE_SGX_STATE_MATRIX 0x100
#define GE_SGX_STATE_FULL_EVERY 16

typedef struct {
	unsigned int type;
	unsigned int a;
	unsigned int b;
	unsigned int c;
	unsigned int d;
	unsigned int seq; // position of the event, written last
	unsigned int reserved[2];
} GeSgxEvent; // 0x20

typedef struct {
	unsigned int magic;
	unsigned int version;
	unsigned int session;      // changes with every game start
	unsigned int entries;      // GE_SGX_RING_ENTRIES
	unsigned int reserved0[12];

	// Written by the PSP side (own 64 byte line)
	volatile unsigned int write; // events written so far
	volatile unsigned int lost;  // events dropped because the ring was full
	unsigned int reserved1[14];

	// PSP side statistics (own 64 byte line)
	volatile unsigned int sync_waits;    // DrawSync/ListSync calls that waited for the Vita side
	volatile unsigned int sync_timeouts; // ... and gave up
	volatile unsigned int sync_wait_us;  // total time waited
	volatile unsigned int sync_max_us;   // longest wait
	volatile unsigned int vram_copies;      // VRAM copies into the window (see GE_SGX_VRAM_SHADOW)
	volatile unsigned int vram_copy_us;     // their total time
	volatile unsigned int vram_copy_pages;  // pages copied in all
	volatile unsigned int vram_copy_max_us; // the longest
	volatile unsigned int vram_copy_how;    // 1: the DMA, 2: the CPU
	char vram_copy_module[16];              // the module whose sceDmacMemcpy is used
	volatile unsigned int sync_peeks;   // DrawSync/ListSync(1) calls (polling whether the GE is done)
	volatile unsigned int sync_held;    // ... answered "drawing" because the Vita side had not read the lists yet
	volatile unsigned int callbacks;    // sceGeSetCallback calls (games told by the GE's interrupt that it is done)
	volatile unsigned int signal_waits;    // signal handlers that waited for the Vita side's walk (GE_SGX_EVENT_SIGNAL)
	volatile unsigned int signal_timeouts; // ... and gave up
	volatile unsigned int signal_wait_us;  // total time waited
	volatile unsigned int signal_max_us;   // longest wait
	volatile unsigned int signal_skipped;  // not waited for: the frame's time for it used up (GE_SGX_SIGNAL_FRAME_US)

	GeSgxEvent events[GE_SGX_RING_ENTRIES];
} GeSgxRing;

// Written by the Vita side only, one cache line
typedef struct {
	volatile unsigned int session; // session of the ring being read
	volatile unsigned int read;    // events of that ring consumed so far
	unsigned int reserved[14];
} GeSgxAck;

#define GE_SGX_ACK_ADDRESS 0xABCDFFC0 // last line of ADRENALINE_ADDRESS/ADRENALINE_SIZE

/*
	PSP VRAM is the GE's own memory inside the PSP part of the chip: the Vita
	side cannot read it. pspemu has a window for it in PSP RAM (PSP 0x0BE00000,
	SCE_PSPEMU_VRAM, what ScePspemuConvertAddress gives for 0x04000000), which
	the PSP kernel fills only when it suspends (that is how savestates keep
	VRAM). The PSP side copies VRAM into that window at every display flip,
	with the DMA, so the Vita side reads VRAM there as it is.
*/
#define GE_SGX_VRAM_SHADOW 0xABE00000 // uncached kernel address of PSP 0x0BE00000
#define GE_SGX_VRAM_SIZE   0x00200000

/*
	Only the VRAM pages the Vita side reads (textures, CLUTs; not the
	framebuffers, which it renders itself) are copied: it sets their bits in
	this bitmap, one per 4 KB page, the first time it reads them. Written by
	the Vita side only, in the line before GeSgxAck, for the session there;
	bits are only ever set (all cleared when the Vita side takes a ring).
*/
#define GE_SGX_VRAM_PAGE 4096
#define GE_SGX_VRAM_PAGES (GE_SGX_VRAM_SIZE / GE_SGX_VRAM_PAGE) // 512

typedef struct {
	volatile unsigned int bits[GE_SGX_VRAM_PAGES / 32]; // of the session in GeSgxAck
} GeSgxVramWant; // one 64 byte line

#define GE_SGX_VRAM_WANT_ADDRESS 0xABCDFF80

/*
	Longest time a DrawSync/ListSync waits for the Vita side to read the lists.
	After DrawSync the game writes its next lists over these: a wait cut short
	leaves the Vita side reading new commands in old lists (the 'bad list'
	ones, objects missing and stripes). Long enough for the heaviest frames
	(Need for Speed Carbon's race: 50-60 ms), the game waits for the walker;
	only a Vita side that stopped reading (then not waited for again until it
	reads) costs it once.
*/
#define GE_SGX_SYNC_TIMEOUT_US 100000

// Longest wait for the Vita side in a signal handler (the GE's interrupt: nothing else runs on the PSP meanwhile)
#define GE_SGX_SIGNAL_TIMEOUT_US 3000
// ... and in all of a frame's signal handlers (until the next flip): past it, the lists are read at the game's wait for the GE
#define GE_SGX_SIGNAL_FRAME_US 6000

#endif
