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

#include <psp2/appmgr.h>
#include <psp2/ctrl.h>
#include <psp2/sysmodule.h>
#include <psp2/io/fcntl.h>
#include <psp2/io/devctl.h>
#include <psp2/io/dirent.h>
#include <psp2/io/stat.h>
#include <psp2/kernel/processmgr.h>
#include <psp2/kernel/modulemgr.h>
#include <psp2/net/http.h>
#include <psp2/net/net.h>
#include <psp2/net/netctl.h>
#include <taihen.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utils.h"
#include "pspdebug.h"

#include "../adrenaline_compat.h"
#include "../user/ge_sgx_cg.h"
#include "../user/main.h"
#include <stddef.h>

#define printf psvDebugScreenPrintf

#define EBOOT_URL "http://de01.psp.update.playstation.org/update/psp/image/eu/2014_1212_6be8878f475ac5b1a499b95ab2f7d301/EBOOT.PBP"
#define ADRENALINE_USER_AGENT "Adrenaline/1.00 libhttp/1.1"

static int downloadFile(const char *src, const char *dst) {
	int res;
	int statusCode;
	int tmplId = -1, connId = -1, reqId = -1;
	SceUID fd = -1;
	int ret = 1;

	res = sceHttpCreateTemplate(ADRENALINE_USER_AGENT, SCE_HTTP_VERSION_1_1, SCE_TRUE);
	if (res < 0)
		goto ERROR_EXIT;

	tmplId = res;

	res = sceHttpCreateConnectionWithURL(tmplId, src, SCE_TRUE);
	if (res < 0)
		goto ERROR_EXIT;

	connId = res;

	res = sceHttpCreateRequestWithURL(connId, SCE_HTTP_METHOD_GET, src, 0);
	if (res < 0)
		goto ERROR_EXIT;

	reqId = res;

	res = sceHttpSendRequest(reqId, NULL, 0);
	if (res < 0)
		goto ERROR_EXIT;

	res = sceHttpGetStatusCode(reqId, &statusCode);
	if (res < 0)
		goto ERROR_EXIT;

	if (statusCode == 200) {
		uint8_t buf[4096];
		uint64_t size = 0;
		uint32_t value = 0;

		res = sceHttpGetResponseContentLength(reqId, &size);
		if (res < 0)
			goto ERROR_EXIT;

		res = sceIoOpen(dst, SCE_O_WRONLY | SCE_O_CREAT | SCE_O_TRUNC, 0777);
		if (res < 0)
			goto ERROR_EXIT;

		fd = res;

		printf("Downloading...");

		int x = psvDebugScreenGetX();
		int y = psvDebugScreenGetY();

		while (1) {
			int read = sceHttpReadData(reqId, buf, sizeof(buf));

			if (read < 0) {
				res = read;
				break;
			}

			if (read == 0)
				break;

			int written = sceIoWrite(fd, buf, read);

			if (written < 0) {
				res = written;
				break;
			}

			value += read;

			psvDebugScreenSetXY(x, y);
			printf("%u%%", size ? (unsigned)((uint64_t)value * 100 / size) : 0u);
		}
	}

ERROR_EXIT:
	if (fd >= 0)
		sceIoClose(fd);

	if (reqId >= 0)
		sceHttpDeleteRequest(reqId);

	if (connId >= 0)
		sceHttpDeleteConnection(connId);

	if (tmplId >= 0)
		sceHttpDeleteTemplate(tmplId);

	if (res < 0)
		return res;

	return ret;
}

static SceUID loadModule(const char *path) {
	return sceKernelLoadStartModule(path, 0, NULL, 0, NULL, NULL);
}

static void *cgAlloc(unsigned int size) {
	return malloc(size);
}

static void cgFree(void *ptr) {
	free(ptr);
}

static void compileLog(uint32_t key, int line, const char *msg) {
	char buf[256];
	int n = snprintf(buf, sizeof(buf), "%04X line %d: %s\n", (unsigned)key, line, msg);
	printf("  %s", buf);
	SceUID fd = sceIoOpen(GE_SGX_CG_DIR "/compile_log.txt", SCE_O_WRONLY | SCE_O_CREAT | SCE_O_APPEND, 0777);
	if (fd >= 0) {
		sceIoWrite(fd, buf, n > 0 && n < (int)sizeof(buf) ? n : 0);
		sceIoClose(fd);
	}
}

// The SGX renderer option of the Adrenaline menu (adrenaline.bin), 0 if not set
static int sgxRendererOption(void) {
	int value = 0;
	SceUID fd = sceIoOpen("ux0:app/" ADRENALINE_TITLEID "/adrenaline.bin", SCE_O_RDONLY, 0);
	if (fd < 0)
		return 0;
	if (sceIoLseek(fd, offsetof(AdrenalineConfig, sgx_renderer), SCE_SEEK_SET) < 0 ||
	    sceIoRead(fd, &value, sizeof(value)) != sizeof(value))
		value = 0;
	sceIoClose(fd);
	return value;
}

/*
	A program the compiler failed on (or crashed on: its .try mark left
	behind) gets <path>.fail with the generator's revision in it; it is
	not tried again until the generator changes (GE_SGX_CG_GEN_REV)
*/
static void writeFail(const char *path) {
	char fail[104], rev[16];
	snprintf(fail, sizeof(fail), "%s.fail", path);
	int len = snprintf(rev, sizeof(rev), "%d", GE_SGX_CG_GEN_REV);
	SceUID fd = sceIoOpen(fail, SCE_O_WRONLY | SCE_O_CREAT | SCE_O_TRUNC, 0777);
	if (fd >= 0) {
		sceIoWrite(fd, rev, len);
		sceIoClose(fd);
	}
}

static int failedBefore(const char *path) {
	char fail[104], rev[16], now[16];
	snprintf(fail, sizeof(fail), "%s.fail", path);
	SceUID fd = sceIoOpen(fail, SCE_O_RDONLY, 0);
	if (fd < 0)
		return 0;
	int len = sceIoRead(fd, rev, sizeof(rev) - 1);
	sceIoClose(fd);
	rev[len > 0 ? len : 0] = '\0';
	snprintf(now, sizeof(now), "%d", GE_SGX_CG_GEN_REV);
	return strcmp(rev, now) == 0;
}

/*
	The cache's stamp: the programs' version, the generator's revision and
	the list of keys it was made for. The same as this build's: the cache is
	complete, nothing looked at (every start looked at every program's file,
	thousands of them: seconds). Written after a pass that went through the
	whole list.
*/
#define GE_SGX_CG_STAMP GE_SGX_CG_DIR "/ready"

static void stampText(const uint32_t *keys, int n, char *out, int size) {
	uint32_t h = 2166136261u;
	int i;
	for (i = 0; i < n; i++)
		h = (h ^ keys[i]) * 16777619u;
	snprintf(out, size, "version %d vs %d revision %d keys %d %08X\n", GE_SGX_CG_VERSION, GE_SGX_CG_VS_VERSION, GE_SGX_CG_GEN_REV, n, (unsigned)h);
}

static int stampMatches(const char *want) {
	char have[96];
	SceUID fd = sceIoOpen(GE_SGX_CG_STAMP, SCE_O_RDONLY, 0);
	if (fd < 0)
		return 0;
	int len = sceIoRead(fd, have, sizeof(have) - 1);
	sceIoClose(fd);
	have[len > 0 ? len : 0] = '\0';
	return strcmp(have, want) == 0;
}

static void writeStamp(const char *text) {
	SceUID fd = sceIoOpen(GE_SGX_CG_STAMP, SCE_O_WRONLY | SCE_O_CREAT | SCE_O_TRUNC, 0777);
	if (fd >= 0) {
		sceIoWrite(fd, text, strlen(text));
		sceIoClose(fd);
	}
}

/*
	SGX renderer: libshacccg.suprx cannot be loaded inside the emulator
	(ScePspemu), only in a normal app like this one. With the renderer on
	(or when adrenaline_user asked for it, GE_SGX_CG_REQUEST), compile every
	program that is not cached yet. This app runs at the first start of
	Adrenaline after a reboot (afterwards the system starts the emulator
	directly).
*/
static void compileSgxShaders(void) {
	SceIoStat stat;
	int requested = sceIoGetstat(GE_SGX_CG_REQUEST, &stat) >= 0;
	if (!requested && sgxRendererOption() < 2)
		return;

	static uint32_t keys[4096];
	int n = geSgxCgAllKeys(keys, 4096 - 256);
	if (n > 4096 - 256)
		n = 4096 - 256;
	char stamp[96];
	stampText(keys, n, stamp, sizeof(stamp));
	int complete = stampMatches(stamp);
	if (complete && !requested)
		return; // made for this build already
	// Complete: only what the renderer asked for is looked at
	if (complete)
		n = 0;
	// The keys the renderer asked for (one per line, hex), not in the list
	if (requested) {
		static char req[4096];
		SceUID rfd = sceIoOpen(GE_SGX_CG_REQUEST, SCE_O_RDONLY, 0);
		int len = rfd >= 0 ? sceIoRead(rfd, req, sizeof(req) - 1) : 0;
		if (rfd >= 0)
			sceIoClose(rfd);
		req[len > 0 ? len : 0] = '\0';
		char *s = req;
		while (*s && n < 4096) {
			char *end;
			uint32_t k = (uint32_t)strtoul(s, &end, 16);
			if (end == s) {
				s++;
				continue;
			}
			s = end;
			int j, have = 0;
			for (j = 0; j < n && !have; j++)
				have = keys[j] == k;
			if (!have && k != 0 && k != 1)
				keys[n++] = k;
		}
	}
	int i, missing = 0;
	char path[96], mark[104];
	/*
		A program the compiler could not make, or that crashed it (its
		.try mark left from the try), is not tried again with this
		generator: <path>.fail
	*/
	for (i = 0; i < n; i++) {
		geSgxCgPath(keys[i], path, sizeof(path));
		snprintf(mark, sizeof(mark), "%s.try", path);
		if (sceIoGetstat(mark, &stat) >= 0) {
			// The .try holds the revision it was tried with: a crash of an older generator is tried again
			char fail[104];
			snprintf(fail, sizeof(fail), "%s.fail", path);
			sceIoRemove(fail);
			sceIoRename(mark, fail);
		}
	}
	for (i = 0; i < n; i++) {
		geSgxCgPath(keys[i], path, sizeof(path));
		if (sceIoGetstat(path, &stat) < 0 && !failedBefore(path))
			missing++;
	}

	{
		char line[160];
		int len = snprintf(line, sizeof(line), "--- shaders version %d: %d programs, %d not compiled yet%s\n", GE_SGX_CG_VERSION, n, missing,
			requested ? " (requested by adrenaline_user)" : "");
		sceIoMkdir("ux0:data/adrenaline", 0777);
		sceIoMkdir(GE_SGX_CG_DIR, 0777);
		SceUID fd = sceIoOpen(GE_SGX_CG_DIR "/compile_log.txt", SCE_O_WRONLY | SCE_O_CREAT | SCE_O_APPEND, 0777);
		if (fd >= 0) {
			sceIoWrite(fd, line, len);
			sceIoClose(fd);
		}
	}

	if (missing > 0) {
		printf("SGX renderer: compiling %d shaders (once)...\n", missing);
		int res = geSgxCgInit(loadModule, cgAlloc, cgFree);
		if (res != 1) {
			printf("libshacccg.suprx could not be loaded (0x%08X).\n", res);
			printf("It is expected in ur0:data/ or ur0:data/external/.\n");
			sceKernelDelayThread(3 * 1000 * 1000);
			return;
		}

		int done = 0, failed = 0;
		int x = psvDebugScreenGetX(), y = psvDebugScreenGetY();
		for (i = 0; i < n; i++) {
			geSgxCgPath(keys[i], path, sizeof(path));
			if (sceIoGetstat(path, &stat) >= 0 || failedBefore(path))
				continue;
			// The .try mark (with the generator's revision): if the compiler crashes on this one, the next start skips it
			snprintf(mark, sizeof(mark), "%s.try", path);
			SceUID fd = sceIoOpen(mark, SCE_O_WRONLY | SCE_O_CREAT | SCE_O_TRUNC, 0777);
			if (fd >= 0) {
				char rev[16];
				int len = snprintf(rev, sizeof(rev), "%d", GE_SGX_CG_GEN_REV);
				sceIoWrite(fd, rev, len);
				sceIoClose(fd);
			}
			uint32_t size = 0;
			SceGxmProgram *program = geSgxCgCompile(keys[i], &size, compileLog);
			if (program) {
				geSgxCgSave(keys[i], program, size);
				free(program);
				sceIoRemove(mark);
			} else {
				writeFail(path);
				sceIoRemove(mark);
				failed++;
			}
			done++;
			psvDebugScreenSetXY(x, y);
			printf("%d / %d", done, missing);
		}
		geSgxCgTerm();
		printf("\n");
		if (!complete)
			writeStamp(stamp);
		{
			char line[96];
			int len = snprintf(line, sizeof(line), "--- compiled %d, failed %d\n", done - failed, failed);
			SceUID fd = sceIoOpen(GE_SGX_CG_DIR "/compile_log.txt", SCE_O_WRONLY | SCE_O_CREAT | SCE_O_APPEND, 0777);
			if (fd >= 0) {
				sceIoWrite(fd, line, len);
				sceIoClose(fd);
			}
		}
		if (failed) {
			printf("%d shaders did not compile.\n", failed);
			sceKernelDelayThread(3 * 1000 * 1000);
		}
	}

	// Nothing to make: the list's programs are all there (or known to fail)
	if (missing == 0 && !complete)
		writeStamp(stamp);
	sceIoRemove(GE_SGX_CG_REQUEST);
}

static void initNet() {
	static char memory[16 * 1024];

	sceSysmoduleLoadModule(SCE_SYSMODULE_NET);
	sceSysmoduleLoadModule(SCE_SYSMODULE_HTTPS);

	SceNetInitParam param;
	param.memory = memory;
	param.size = sizeof(memory);
	param.flags = 0;

	sceNetInit(&param);
	sceNetCtlInit();

	sceHttpInit(40 * 1024);
}

int main() {
	int res;

	psvDebugScreenInit();

	// Safe mode
	if (sceIoDevctl("ux0:", 0x3001, NULL, 0, NULL, 0) == 0x80010030) {
		printf("Please enable unsafe homebrew first before using this software.");
		while (1);
	}

	// Enable write access
	sceAppMgrUmount("app0:");

	SceIoStat stat;
	memset(&stat, 0, sizeof(SceIoStat));
	if (sceIoGetstat("ux0:app/" ADRENALINE_TITLEID "/flash0", &stat) < 0 &&
		sceIoGetstat("ux0:app/" ADRENALINE_TITLEID "/661.PBP", &stat) < 0) {
		printf("The 6.61 firmware has not been installed yet and 661.PBP does not\nexist.\n");
		printf("Press X to download the PSP 6.61 firmware.\n");
		printf("Press any other button to ignore it (but you need to manually\nput 661.PBP to ux0:app/" ADRENALINE_TITLEID "/661.PBP" ").\n\n");

		while (1) {
			SceCtrlData pad;
			sceCtrlReadBufferPositive(0, &pad, 1);

			if (pad.buttons & SCE_CTRL_CROSS) {
				initNet();

				res = downloadFile(EBOOT_URL, "ux0:app/" ADRENALINE_TITLEID "/661.PBP");
				if (res < 0) {
					printf("Error 0x%08X downloading file.\n", res);
					while (1);
				}

				break;
			} else if (pad.buttons != 0) {
				break;
			}
		}
	}

	// SGX renderer: the emulator could not load the shader compiler
	compileSgxShaders();

	// Load kernel module
	res = taiLoadStartKernelModule("ux0:app/" ADRENALINE_TITLEID "/sce_module/adrenaline_kernel.skprx", 0, NULL, 0);
	if (res < 0) {
		printf("Could not load adrenaline_kernel.skprx. Please uninstall the old Adrenaline first.");
		while (1);
	}

	return 0;
}