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
#define GE_SGX_VERSION 1

#define GE_SGX_RING_ENTRIES 4096 // power of two
#define GE_SGX_RING_MASK (GE_SGX_RING_ENTRIES - 1)

enum GeSgxEventTypes {
	GE_SGX_EVENT_NONE,
	GE_SGX_EVENT_LIST,      // a = queue id, b = list address, c = stall address (0: none), d = 1 if queued at the head
	GE_SGX_EVENT_STALL,     // a = queue id, b = new stall address
	GE_SGX_EVENT_FLIP,      // a = top address, b = buffer width, c = pixel format, d = sync
	GE_SGX_EVENT_DEQUEUE,   // a = queue id
};

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
	unsigned int reserved2[12];

	GeSgxEvent events[GE_SGX_RING_ENTRIES];
} GeSgxRing;

// Written by the Vita side only, one cache line
typedef struct {
	volatile unsigned int session; // session of the ring being read
	volatile unsigned int read;    // events of that ring consumed so far
	unsigned int reserved[14];
} GeSgxAck;

#define GE_SGX_ACK_ADDRESS 0xABCDFFC0 // last line of ADRENALINE_ADDRESS/ADRENALINE_SIZE

// Longest time a DrawSync/ListSync waits for the Vita side to read the lists
#define GE_SGX_SYNC_TIMEOUT_US 3000

#endif
