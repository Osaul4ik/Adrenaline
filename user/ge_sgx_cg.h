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
	SGX renderer: the Cg shaders of the GE states and their compiler
	(SceShaccCg, libshacccg.suprx). Shared by adrenaline_user and the
	Adrenaline app (bubble), which compiles the programs before the
	emulator starts where the emulator itself cannot load the compiler.
*/

#ifndef __GE_SGX_CG_H__
#define __GE_SGX_CG_H__

#include <stdint.h>
#include <psp2/gxm.h>

// Bump when the generated sources change: older cached programs are not used
#define GE_SGX_CG_VERSION 1

#define GE_SGX_CG_DIR "ux0:data/adrenaline/sgx_shaders"
// Written by adrenaline_user when it could not compile: the app compiles all programs at its next start
#define GE_SGX_CG_REQUEST GE_SGX_CG_DIR "/compile_request"

// Key of the vertex program (fragment program keys are below 0x800)
#define GE_SGX_SHADER_VERTEX 0xFFFFu

// Cg source of a program
int geSgxCgSource(uint32_t key, char *buf, int size);

// Every program key there is, returns their number
int geSgxCgAllKeys(uint32_t *keys, int max);

void geSgxCgPath(uint32_t key, char *path, int size);
SceGxmProgram *geSgxCgLoadCached(uint32_t key); // malloc'd, NULL if not there
void geSgxCgSave(uint32_t key, const SceGxmProgram *program, uint32_t size);

/*
	Loads libshacccg.suprx. load_fn loads a module (sceKernelLoadStartModule
	or a wrapper), alloc/free are the compiler's heap. Returns 1, or the
	error of the module load.
*/
int geSgxCgInit(SceUID (* load_fn)(const char *path), void *(* alloc)(unsigned int), void (* release)(void *));
void geSgxCgTerm(void);

// Compiles a program (malloc'd); log receives the compiler's messages
SceGxmProgram *geSgxCgCompile(uint32_t key, uint32_t *size, void (* log)(uint32_t key, int line, const char *msg));

#endif
