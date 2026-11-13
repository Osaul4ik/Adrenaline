#!/usr/bin/env python3
"""Sanity check of a packed kernel PRX (~PSP + gzip), as built by `make -C cef _systemctrl`.

Verifies the ~PSP header, that the payload decompresses to the PRX ELF, and that
the import stub tables are consistent (every library has exactly the NIDs it
declares). Optionally checks that given NIDs are imported:

    check_prx.py systemctrl.prx [LIB:NID ...]
"""

import gzip
import struct
import sys



def fail(msg):
    print("FAIL:", msg)
    sys.exit(1)


def sections(elf):
    shoff = struct.unpack_from("<I", elf, 0x20)[0]
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", elf, 0x2E)
    secs = [struct.unpack_from("<10I", elf, shoff + i * shentsize) for i in range(shnum)]
    strtab = secs[shstrndx][4]

    def name(off):
        end = elf.index(b"\0", strtab + off)
        return elf[strtab + off:end].decode()

    return {name(s[0]): s for s in secs}, secs


def va_to_off(secs, va):
    for s in secs:
        # s: name type flags addr offset size ...
        if s[3] and s[3] <= va < s[3] + s[5] and s[1] != 8:  # not NOBITS
            return s[4] + va - s[3]
    return None


def imports(elf):
    named, secs = sections(elf)
    stub = named.get(".lib.stub")
    if not stub:
        fail("no .lib.stub section")

    result = {}
    spans = []
    for i in range(stub[5] // 20):
        name_va, _ver, _attr, _len, _vcount, fcount, nids_va, _funcs = struct.unpack_from("<IHHBBHII", elf, stub[4] + i * 20)
        o = va_to_off(secs, name_va)
        lib = elf[o:elf.index(b"\0", o)].decode()
        no = va_to_off(secs, nids_va)
        nids = [struct.unpack_from("<I", elf, no + 4 * j)[0] for j in range(fcount)]
        result.setdefault(lib, []).append(nids)
        spans.append((nids_va, fcount, lib))
    return result, spans


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)

    data = open(sys.argv[1], "rb").read()
    required = sys.argv[2:]

    if data[:4] != b"~PSP":
        fail("missing ~PSP header")

    comp_attr = struct.unpack_from("<H", data, 6)[0]
    if not comp_attr & 1:
        fail("payload not compressed")

    tag = struct.unpack_from("<I", data, 0xD0)[0]
    if tag != 0xDADADAF0:
        fail("unexpected tag 0x%08X (kernel module tag is 0xDADADAF0)" % tag)

    elf_size = struct.unpack_from("<I", data, 0x28)[0]
    elf = gzip.decompress(data[0x150:])
    if len(elf) != elf_size:
        fail("decompressed size %d != header elf_size %d" % (len(elf), elf_size))
    if elf[:4] != b"\x7fELF":
        fail("payload is not an ELF")

    libs, spans = imports(elf)

    # NID tables of different libraries must not overlap (a merged stub whose
    # count was raised without its own table growing reads its neighbour's NIDs)
    spans.sort()
    for (a, n, lib), (b, _, nxt) in zip(spans, spans[1:]):
        if a + 4 * n > b:
            fail("NID table of %s overlaps %s" % (lib, nxt))

    # Every NID table must stay inside .rodata.sceNid: a stub library merged from
    # two sources with a short NID table would make imports resolve to wrong functions.
    named, _ = sections(elf)
    nid_sec = named.get(".rodata.sceNid")
    all_nids = {n for tables in libs.values() for t in tables for n in t}
    if nid_sec:
        nid_words = {struct.unpack_from("<I", elf, nid_sec[4] + 4 * k)[0] for k in range(nid_sec[5] // 4)}
        stray = all_nids - nid_words
        if stray:
            fail("NIDs outside .rodata.sceNid: %s" % ", ".join("0x%08X" % n for n in sorted(stray)))

    for lib, tables in libs.items():
        if len(tables) > 1:
            fail("library %s imported by %d separate stub tables" % (lib, len(tables)))

    for req in required:
        lib, nid = req.split(":")
        nid = int(nid, 16)
        if nid not in (libs.get(lib) or [[]])[0]:
            fail("%s does not import 0x%08X" % (lib, nid))

    print("OK: %s, ELF %d bytes, %d import libraries" % (sys.argv[1], len(elf), len(libs)))
    for lib in sorted(libs):
        print("  %-28s %d" % (lib, len(libs[lib][0])))


if __name__ == "__main__":
    main()
