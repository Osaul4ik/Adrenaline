/* Placeholder definitions for the symbols named in kernel/exports.yml.
   Only used to generate import stubs, never run. */

int module_start(unsigned int args, void *argp) {
  return 0;
}

int module_stop(unsigned int args, void *argp) {
  return 0;
}

int kuCtrlPeekBufferPositive(int port, void *pad_data, int count) {
  return 0;
}

void _start() __attribute__ ((weak, alias ("module_start")));
