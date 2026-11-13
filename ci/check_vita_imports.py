#!/usr/bin/env python3
"""Check the imports of a Vita module against firmware 3.60 and a reference build.

    check_vita_imports.py NEW_MODULE [REFERENCE_MODULE] --db VITA_HEADERS_DB_DIR

Modules may be ELF or (f)SELF. Every imported function must exist in the
vita-headers NID database of firmware 3.60 (the oldest firmware Adrenaline
supports), otherwise the module would fail to load on that firmware.
Libraries that are not in the database (taiHEN, Adrenaline's own kubridge)
are only listed. With a reference module, imports that were not there before
are printed as well, and a library the reference module did not import must be
weak: the user module is loaded before most system libraries are, and a strong
import of a library that is not loaded yet fails with 0x8002D003 (NO_LIB).
"""

import glob
import struct
import sys
import zlib

import yaml


def self_to_elf(d):
    h = struct.unpack_from("<IIHHIQQQQQQQQQQQQQ", d, 0)
    elf_off, phdr_off, secinfo_off = h[11], h[12], h[14]
    eh = bytearray(d[elf_off:elf_off + 52])
    phnum = struct.unpack_from("<H", eh, 0x2C)[0]
    phs = [list(struct.unpack_from("<8I", d, phdr_off + 32 * i)) for i in range(phnum)]
    struct.pack_into("<I", eh, 0x1C, 52)
    out = eh + bytearray(32 * phnum)
    for i, p in enumerate(phs):
        off, ln, comp, _enc = struct.unpack_from("<QQQQ", d, secinfo_off + 32 * i)
        data = d[off:off + ln]
        if comp == 2:
            data = zlib.decompress(data)
        pos = (len(out) + 15) & ~15
        out += b"\0" * (pos - len(out))
        p[1] = pos
        out += data
        struct.pack_into("<8I", out, 52 + 32 * i, *p)
    return bytes(out)


def load(path):
    d = open(path, "rb").read()
    if d[:4] == b"SCE\0":
        d = self_to_elf(d)
    if d[:4] != b"\x7fELF":
        sys.exit("not an ELF/SELF: " + path)
    return d


def imports(d):
    phnum = struct.unpack_from("<H", d, 0x2C)[0]
    ph = [struct.unpack_from("<8I", d, 52 + 32 * i) for i in range(phnum)]

    def rd(va, n):
        for p in ph:
            if p[0] == 1 and p[2] <= va < p[2] + p[4]:
                o = p[1] + va - p[2]
                return d[o:o + n]
        raise ValueError(hex(va))

    def u32(va):
        return struct.unpack("<I", rd(va, 4))[0]

    def cstr(va):
        s = b""
        while rd(va, 1) != b"\0":
            s += rd(va, 1)
            va += 1
        return s.decode()

    e = struct.unpack_from("<I", d, 0x18)[0]
    seg, off = e >> 30, e & 0x3FFFFFFF
    base, mi = ph[seg][2], ph[seg][1] + off
    it, ib = struct.unpack_from("<II", d, mi + 0x2C)

    result = {}
    weak = set()
    va = base + it
    while va < base + ib:
        size = rd(va, 1)[0]
        flags = struct.unpack("<H", rd(va + 4, 2))[0]
        if size == 0x34:
            nf = struct.unpack("<H", rd(va + 6, 2))[0]
            lnid, lname = struct.unpack("<II", rd(va + 0x10, 8))
            fnids = u32(va + 0x1C)
        elif size == 0x24:
            nf = struct.unpack("<H", rd(va + 6, 2))[0]
            lnid, lname, fnids = struct.unpack("<III", rd(va + 0x0C, 12))
        else:
            sys.exit("unknown import entry size 0x%x" % size)
        name = cstr(lname)
        result.setdefault((name, lnid), set()).update(u32(fnids + 4 * i) for i in range(nf))
        if flags & 0x8:
            weak.add(name)
        va += size
    return result, weak


def load_db(db_dir):
    libs = {}  # lib nid -> (name, {func nid: name})
    for f in glob.glob(db_dir + "/360/*.yml"):
        y = yaml.safe_load(open(f))
        for mod in (y.get("modules") or {}).values():
            for lname, lib in (mod.get("libraries") or {}).items():
                funcs = {v: k for k, v in (lib.get("functions") or {}).items()}
                funcs.update({v: k for k, v in (lib.get("variables") or {}).items()})
                libs[lib.get("nid")] = (lname, funcs)
    return libs


def main():
    args = sys.argv[1:]
    if "--db" not in args:
        sys.exit(__doc__)
    i = args.index("--db")
    db_dir = args[i + 1]
    paths = args[:i] + args[i + 2:]

    db = load_db(db_dir)
    new, new_weak = imports(load(paths[0]))
    ref = imports(load(paths[1]))[0] if len(paths) > 1 else None

    missing = []
    for (lname, lnid), nids in sorted(new.items()):
        known = db.get(lnid)
        if not known:
            print("  %-28s 0x%08X  %3d imports%s (not in the 3.60 database)" % (lname, lnid, len(nids), " weak" if lname in new_weak else ""))
            continue
        for nid in sorted(nids):
            if nid not in known[1]:
                missing.append((lname, nid))
        print("  %-28s 0x%08X  %3d imports%s" % (lname, lnid, len(nids), " weak" if lname in new_weak else ""))

    if ref is not None:
        ref_pairs = {(l, n) for (l, _), ns in ref.items() for n in ns}
        added = [(l, n) for (l, ln), ns in sorted(new.items()) for n in sorted(ns) if (l, n) not in ref_pairs]
        print("imports not in the reference module: %d" % len(added))
        for l, n in added:
            known = db.get(next(ln for (ll, ln) in new if ll == l))
            fname = known[1].get(n, "?") if known else "?"
            print("  + %s 0x%08X %s" % (l, n, fname))

    strong_new_libs = []
    if ref is not None:
        ref_libs = {l for (l, _) in ref}
        strong_new_libs = [l for (l, _) in new if l not in ref_libs and l not in new_weak]
        for l in strong_new_libs:
            print("FAIL: new library %s is imported strongly (link its _stub_weak)" % l)

    if missing:
        print("FAIL: imports that firmware 3.60 does not export:")
        for l, n in missing:
            print("  %s 0x%08X" % (l, n))
        sys.exit(1)

    if strong_new_libs:
        sys.exit(1)

    print("OK: all imports of known libraries exist in firmware 3.60")


if __name__ == "__main__":
    main()
