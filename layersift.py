"""layersift core — exact, local reading of container image archives.

Three tools over the tar that ``docker save`` (or ``podman save``) writes, or an OCI
image archive (``skopeo copy ... oci-archive:``, ``buildah push ... oci-archive:``):

``image_summary``  What the image is made of and where its size goes: every layer
                   with the Dockerfile instruction that made it, the size of the
                   final filesystem, the bytes that ship but cannot be seen (files
                   deleted or overwritten by a later layer), caches and downloaded
                   archives left behind, and the biggest directories and files.
``image_layer``    One layer: the biggest files it adds, what it deletes or
                   overwrites below it, and how much of it later layers waste.
``image_compare``  Two images (before and after a Dockerfile change): size, layers
                   shared with the base, and the directories that grew or shrank.

A layer is a tar of the files it adds or changes, plus whiteout entries
(``.wh.<name>``, ``.wh..wh..opq``) for what it deletes. Layers are applied in
order, as a container runtime does, to get the final filesystem; a file that a
later layer deletes or replaces is still shipped in the layer that added it.

Standard library only. No network, no Docker daemon, no registry, no credentials,
no subprocess (it never runs docker: pass the saved tar), no extraction to disk, no
background work, no self-update. The only side effect is reading the path given.
"""

from __future__ import annotations

import bz2
import gzip
import io
import json
import lzma
import os
import posixpath
import re
import stat
import sys
import tarfile
from collections import defaultdict
from typing import Any, Optional

# --- Bounds ---------------------------------------------------------------------------
MAX_ARCHIVE_BYTES = 64 * 1024 ** 3       # the saved tar itself
MAX_JSON_BYTES = 8 * 1024 * 1024         # manifest, index and config documents
MAX_UNPACKED_BYTES = 64 * 1024 ** 3      # contents declared per archive and across its layers (~2.5 s/GiB to read)
MAX_ENTRIES = 3_000_000                  # tar entries across all layers
MAX_LAYERS = 1_000
MAX_META_BYTES = 64 * 1024 * 1024        # all JSON-looking members kept from one archive
MAX_HEADER_BYTES = 1024 * 1024           # one long-name or pax extended header
DEFAULT_TOP = 15
HARD_TOP = 200
MIB = 1024 * 1024
FINDING_MIN = 1 * MIB                    # smaller leftovers are not worth a finding


# --- Paths (same guards as Hermes' read_file) -------------------------------------
def _read_block_error(abs_path: str) -> Any:
    """Honour Hermes' read denylist when running inside the agent.

    The host refuses reads of its credential stores, internal skill-hub caches
    and project ``.env`` files; layersift must not become a side channel around
    that guard. It defers to the same check Core's ``read_file`` uses (which
    resolves symlinks itself). Outside a Hermes runtime the module is absent and
    there is no denylist to apply, so the check no-ops. If the check itself fails,
    the read is refused: a denylist that could not be consulted is not a pass.
    """
    try:
        from agent.file_safety import get_read_block_error
    except Exception:
        return None
    try:
        return get_read_block_error(abs_path)
    except Exception as exc:
        return f"Refusing to read {abs_path}: the read denylist check failed ({type(exc).__name__})"


# Like Hermes' read_file, never read device, fd or process files: they can block
# forever (/dev/zero, a FIFO) or leak process state (/proc/<pid>/environ).
_REFUSED_ROOTS = ("/proc", "/dev")


def _under_refused_root(path: str) -> bool:
    candidates = [os.path.abspath(path)]
    try:
        candidates.append(os.path.realpath(path))
    except (OSError, ValueError):
        pass
    return any(c == r or c.startswith(r + "/") for c in candidates for r in _REFUSED_ROOTS)


_GUIDE = "Save the image with: docker save -o image.tar <image> (or podman save), and pass image.tar."


def _resolve(raw: Any, name: str = "path") -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"'{name}' must be a non-empty string")
    path = os.path.expanduser(raw.strip())
    # The raw string is checked too, as Core does: abspath() turns an NT namespace
    # path such as \??\UNC\host\share into C:\??\..., which the check no longer
    # recognises, while stat() on the raw path would still reach the network.
    blocked = _read_block_error(path) or _read_block_error(os.path.abspath(path))
    if blocked:
        raise ValueError(blocked)
    if _under_refused_root(path):
        raise ValueError(f"Refusing to read {path}: device and /proc paths are not image archives")
    # Windows device and namespace paths (\\.\pipe\x, \\?\..., \??\...): stat() alone
    # would connect to a named pipe server
    if os.name == "nt" and re.match(r"^[\\/]{2}[.?][\\/]|^\\\?\?\\", path):
        raise ValueError(f"Refusing to read {path}: device and namespace paths are not image archives")
    try:
        st = os.stat(path)
    except FileNotFoundError:
        raise ValueError(f"Path not found: {path}") from None
    except OSError as exc:
        raise ValueError(f"Cannot access {path}: {exc.strerror or exc}") from None
    if stat.S_ISDIR(st.st_mode):
        raise ValueError(f"{path} is a directory. {_GUIDE}")
    if not stat.S_ISREG(st.st_mode):
        raise ValueError(f"Refusing to read {path}: not a regular file")
    if st.st_size > MAX_ARCHIVE_BYTES:
        raise ValueError(f"{path} is {st.st_size} bytes, over the {MAX_ARCHIVE_BYTES // 1024 ** 3} GiB limit")
    return path


# --- Archive reading ---------------------------------------------------------------------
def _compression(head: bytes) -> str:
    if head.startswith(b"\x1f\x8b"):
        return "gzip"
    if head.startswith(b"\x28\xb5\x2f\xfd"):
        return "zstd"
    if head.startswith(b"BZh"):
        return "bzip2"
    if head.startswith(b"\xfd7zXZ\x00"):
        return "xz"
    return "none"


def _decompressed(fileobj: Any, comp: str) -> Any:
    """A reader of the decompressed bytes. tarfile's own "r|gz" streams decompress in
    small Python-level steps (8 MB/s on very compressible data); these readers run in C."""
    if comp == "gzip":
        return gzip.GzipFile(fileobj=fileobj, mode="rb")
    if comp == "bzip2":
        return bz2.BZ2File(fileobj, mode="rb")
    if comp == "xz":
        return lzma.LZMAFile(fileobj, mode="rb")
    if comp == "zstd":
        from compression import zstd  # Python 3.14+, checked by _check_codec
        return zstd.ZstdFile(fileobj, mode="rb")
    return fileobj


def _check_codec(comp: str, what: str) -> None:
    if comp == "zstd" and sys.version_info < (3, 14):
        raise ValueError(f"{what} is zstd-compressed, which Python reads from 3.14 on; "
                         "decompress it first (zstd -d) or save the image uncompressed")


class _Prefixed(io.RawIOBase):
    """A stream with its first bytes already read (to sniff the compression) put back."""

    def __init__(self, head: bytes, rest: Any) -> None:
        self.head, self.rest = head, rest

    def readable(self) -> bool:
        return True

    def readinto(self, b: Any) -> int:
        if self.head:
            n = min(len(b), len(self.head))
            b[:n] = self.head[:n]
            self.head = self.head[n:]
            return n
        data = self.rest.read(len(b))
        b[:len(data)] = data
        return len(data)


class _TarInfo(tarfile.TarInfo):
    """tarfile reads a GNU long name or a pax header whole, at whatever size its header
    declares, before handing the member back. Refuse oversized ones before that read."""

    def _proc_member(self, tarfile_: Any) -> Any:
        if self.size < 0:
            raise ValueError("a tar header declares a negative size; refusing the archive")
        if self.type in (tarfile.GNUTYPE_LONGNAME, tarfile.GNUTYPE_LONGLINK, tarfile.XHDTYPE,
                         tarfile.XGLTYPE, tarfile.SOLARIS_XHDTYPE) and self.size > MAX_HEADER_BYTES:
            raise ValueError(f"a tar header declares {self.size} bytes of long name or attributes "
                             f"(the limit is {MAX_HEADER_BYTES}); refusing the archive")
        if self.type == tarfile.GNUTYPE_SPARSE:
            raise ValueError("a tar header is a GNU sparse file, which container images do not use; refusing")
        return super()._proc_member(tarfile_)


def _tar(fileobj: Any, mode: str) -> tarfile.TarFile:
    return tarfile.open(fileobj=fileobj, mode=mode, tarinfo=_TarInfo)


class _Budget:
    """Caps the work a crafted archive can cause: entries, and the file bytes headers
    declare, charged at each header before its data is read or decompressed. There is no
    ratio test: a real layer of zeros (a sparse lastlog) compresses about 1000:1, as a
    bomb does; the absolute cap is what bounds the time a stream can take."""

    def __init__(self, what: str) -> None:
        self.what = what
        self.entries = 0
        self.bytes = 0

    def charge(self, size: int) -> None:
        if size < 0:
            raise ValueError("a tar header declares a negative size; refusing the archive")
        self.entries += 1
        self.bytes += size
        if self.entries > MAX_ENTRIES:
            raise ValueError(f"more than {MAX_ENTRIES} entries in {self.what}; refusing to read further")
        if self.bytes > MAX_UNPACKED_BYTES:
            raise ValueError(f"{self.what} declares more than {MAX_UNPACKED_BYTES // 1024 ** 3} GiB of "
                             "contents; refusing to read further")


def _norm(name: str) -> str:
    p = posixpath.normpath("/" + name.replace("\\", "/")).lstrip("/")
    return "" if p == "." else p


class Layer:
    def __init__(self, index: int, ref: str) -> None:
        self.index = index            # 1-based, bottom layer first
        self.ref = ref                # member name of the layer in the archive
        self.digest: Optional[str] = None
        self.instruction: Optional[str] = None
        self.created: Optional[str] = None
        self.archive_bytes = 0        # as stored in the archive (compressed or not)
        self.compression = "none"
        self.adds: list = []          # (path, size, kind, hardlink target) kind: f file, d dir, l symlink, h hardlink, o other
        self.diff_id: Optional[str] = None   # digest of the uncompressed layer, from the image config
        self.whiteouts: list = []     # paths this layer deletes from lower layers
        self.opaque: list = []        # directories whose lower contents this layer hides
        self.size = 0                 # bytes of regular files this layer adds
        self.files = 0

    def copy_from(self, other: "Layer") -> None:
        for k in ("archive_bytes", "compression", "adds", "whiteouts", "opaque", "size", "files"):
            setattr(self, k, getattr(other, k))


def _scan_layer(fobj: Any, layer: Layer, budget: _Budget) -> None:
    head = fobj.read(6)
    if not head:
        return
    comp = layer.compression = _compression(head)
    _check_codec(comp, f"layer {layer.index}")
    try:
        seekable = bool(fobj.seekable())
    except (AttributeError, OSError):   # a member of a compressed archive read as a stream
        seekable = False
    if comp == "none" and seekable:
        fobj.seek(0)   # uncompressed and seekable: skip file contents instead of reading them
        tf = _tar(fobj, "r:")
    else:
        tf = _tar(_decompressed(io.BufferedReader(_Prefixed(head, fobj), 1 << 20), comp), "r|")
    with tf:
        for m in tf:
            # TarFile keeps every header it has read; a layer can hold hundreds of
            # thousands. Nothing here needs them again, so do not let them pile up.
            tf.members.clear()
            budget.charge(m.size)
            p = _norm(m.name)
            if not p:
                continue
            parent, base = posixpath.split(p)
            if base.startswith(".wh..wh."):
                if base == ".wh..wh..opq":
                    layer.opaque.append(parent)
                continue   # other .wh..wh. names are AUFS bookkeeping (.wh..wh.plnk), not deletions
            if base.startswith(".wh."):
                if base[4:] not in ("", ".", ".."):
                    layer.whiteouts.append(posixpath.join(parent, base[4:]))
                continue
            target = None
            if m.isreg():
                kind, size = "f", m.size
                layer.size += size
                layer.files += 1
            elif m.isdir():
                kind, size = "d", 0
            elif m.issym():
                kind, size = "l", 0
            elif m.islnk():
                kind, size = "h", 0   # shares data with its target: no bytes of its own
                target = _norm(m.linkname)
            else:
                kind, size = "o", 0
            layer.adds.append((p, size, kind, target))


def _open_file(path: str) -> tuple[Any, str]:
    """Open once, without blocking, and check what was opened: the path may have been
    swapped for a FIFO since it was checked. Both passes read this same handle."""
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0))
    fh = os.fdopen(fd, "rb")
    try:
        st = os.fstat(fh.fileno())
        if not stat.S_ISREG(st.st_mode):
            raise ValueError(f"Refusing to read {path}: not a regular file")
        head = fh.read(512)
        comp = _compression(head)
        _check_codec(comp, path)
        if comp == "none" and head.lstrip()[:1] in (b"{", b"["):
            raise ValueError(f"{path} is JSON, not an image archive. {_GUIDE}")
        return fh, comp
    except BaseException:
        fh.close()
        raise


def _outer_tar(fh: Any, comp: str, path: str) -> tarfile.TarFile:
    fh.seek(0)
    try:
        return _tar(fh, "r:") if comp == "none" else _tar(_decompressed(fh, comp), "r|")
    except tarfile.TarError as exc:
        raise ValueError(f"{path} is not a tar archive ({exc}). {_GUIDE}") from None


def _link_target(m: tarfile.TarInfo) -> str:
    return _norm(posixpath.join(posixpath.dirname(_norm(m.name)), m.linkname)) if m.issym() else _norm(m.linkname)


# --- Image metadata --------------------------------------------------------------------------
def _instruction(created_by: Any) -> Optional[str]:
    if not isinstance(created_by, str) or not created_by.strip():
        return None
    s = re.sub(r"\s+# buildkit\s*$", "", created_by.strip())
    m = re.match(r"^/bin/sh -c #\(nop\)\s*(.*)$", s, re.S)
    if m:
        s = m.group(1)
    else:
        # classic builder: "/bin/sh -c apt-get ..." or "|2 A=1 B=2 /bin/sh -c ...";
        # BuildKit: "RUN /bin/sh -c apt-get ..." or "RUN |1 A=1 /bin/sh -c ..."
        m = re.match(r"^(?:RUN )?(?:\|\d+(?: \S+=\S*)* )?/bin/(?:ba)?sh -c (.*)$", s, re.S)
        if m:
            s = "RUN " + m.group(1)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:300] + ("..." if len(s) > 300 else "")


def _blob_path(digest: Any) -> Optional[str]:
    if isinstance(digest, str) and re.fullmatch(r"[a-z0-9]+:[a-f0-9]{32,}", digest):
        alg, hexd = digest.split(":", 1)
        return f"blobs/{alg}/{hexd}"
    return None


def _platform(p: Any) -> Optional[str]:
    if isinstance(p, dict) and p.get("os") and p.get("architecture"):
        return "/".join(str(x) for x in (p["os"], p["architecture"], p.get("variant")) if x)
    return None


class _Docs:
    """JSON members kept as bytes in pass 1 and parsed only when an image refers to them,
    so a crafted archive full of large JSON files costs a byte cap, not a parse each."""

    def __init__(self) -> None:
        self.raw: dict = {}
        self.parsed: dict = {}
        self.dropped: set = set()
        self.total = 0

    def keep(self, name: str, data: bytes) -> None:
        # Past the cap, later documents are not kept (attestation and SBOM blobs can be
        # large and are never read); it is an error only if an image then needs one.
        if self.total + len(data) > MAX_META_BYTES:
            self.dropped.add(name)
            return
        self.total += len(data)
        self.raw[name] = data
        self.parsed.pop(name, None)

    def get(self, name: Any) -> Any:
        if isinstance(name, str) and name in self.dropped and name not in self.raw:
            raise ValueError(f"the archive holds more than {MAX_META_BYTES // MIB} MiB of JSON documents and "
                             f"{name}, which the image needs, is past that limit; refusing")
        if not isinstance(name, str) or name not in self.raw:
            return None
        if name not in self.parsed:
            try:
                self.parsed[name] = json.loads(self.raw[name])
            except (ValueError, RecursionError):
                self.parsed[name] = None
        return self.parsed[name]


def _images(docs: _Docs) -> list:
    """Every image the archive holds: tags, config member, layer members (bottom first)."""
    out = []
    manifest = docs.get("manifest.json")
    if isinstance(manifest, list):   # docker save, podman save (docker-archive)
        for e in manifest:
            if isinstance(e, dict) and isinstance(e.get("Layers"), list):
                out.append({"tags": [t for t in e.get("RepoTags") or [] if isinstance(t, str)],
                            "config": _norm(e["Config"]) if isinstance(e.get("Config"), str) else None,
                            "layers": [_norm(x) for x in e["Layers"] if isinstance(x, str)],
                            "layer_digests": [None] * len(e["Layers"])})
        if out:
            return out
    index = docs.get("index.json")
    if isinstance(index, dict):      # OCI image layout archive
        todo = [(d, None) for d in index.get("manifests") or [] if isinstance(d, dict)]
        seen = 0
        while todo and seen < 256:
            seen += 1
            desc, tag = todo.pop(0)
            ann = desc.get("annotations") if isinstance(desc.get("annotations"), dict) else {}
            tag = ann.get("io.containerd.image.name") or ann.get("org.opencontainers.image.ref.name") or tag
            doc = docs.get(_blob_path(desc.get("digest")))
            if not isinstance(doc, dict):
                continue
            if isinstance(doc.get("manifests"), list):   # an index: one manifest per platform
                todo.extend((d, tag) for d in doc["manifests"] if isinstance(d, dict))
                continue
            if isinstance(doc.get("layers"), list):
                cfg = doc.get("config") if isinstance(doc.get("config"), dict) else {}
                plat = _platform(desc.get("platform"))
                if plat == "unknown/unknown":   # buildx attestation manifests
                    continue
                layers = [d for d in doc["layers"] if isinstance(d, dict)]
                out.append({"tags": [tag] if isinstance(tag, str) else [], "platform": plat,
                            "config": _blob_path(cfg.get("digest")),
                            "layers": [_blob_path(d.get("digest")) or "" for d in layers],
                            "layer_digests": [d.get("digest") for d in layers]})
        if out:
            return out
    raise ValueError(f"no manifest.json or OCI index.json describing an image. {_GUIDE}")


class Image:
    """The layers of one image applied in order, with what each one hides."""

    def __init__(self, path: str, meta: dict, layers: list, config: Optional[dict]) -> None:
        self.path = path
        self.meta = meta
        self.layers = layers
        self.config = config if isinstance(config, dict) else {}
        self.fs: dict = {}                      # path -> (size, layer index, kind)
        self.children: dict = defaultdict(set)  # dir path -> child paths
        self.waste: list = []                   # (path, size, added in, removed by, how)
        self.overwrites_by: dict = defaultdict(list)   # layer index -> waste rows it causes
        self.link_of: dict = {}                 # hardlink path -> the file path holding its data
        self.links_to: dict = defaultdict(set)  # file path -> hardlinks sharing its data
        self._map_history()
        for layer in layers:
            self._apply(layer)
        self.removed_from: dict = defaultdict(int)     # layer index -> its bytes later layers removed
        for w in self.waste:
            self.removed_from[w[2]] += w[1]
        self.final = sum(e[0] for e in self.fs.values())
        self.shipped = sum(L.size for L in layers)

    def _map_history(self) -> None:
        hist = self.config.get("history")
        if not isinstance(hist, list):
            return
        real = [h for h in hist if isinstance(h, dict) and not h.get("empty_layer")]
        if len(real) != len(self.layers):
            return   # cannot tell which instruction made which layer
        for layer, h in zip(self.layers, real):
            layer.instruction = _instruction(h.get("created_by"))
            layer.created = h.get("created") if isinstance(h.get("created"), str) else None

    def _drop(self, path: str, by: int, how: str) -> None:
        stack = [path]
        while stack:
            p = stack.pop()
            stack.extend(self.children.pop(p, ()))
            if p == path:   # descendants' parent sets were popped above
                self.children.get(posixpath.dirname(p), set()).discard(p)
            e = self.fs.pop(p, None)
            if e is None:
                continue
            if e[2] == "h":
                held = self.link_of.pop(p, None)
                if held is not None:
                    self.links_to[held].discard(p)
                continue
            # Removing one name of a hardlinked file removes no bytes while another name
            # survives: that name now holds the data (even an empty file's, so no link is
            # left pointing at a path a later, different file will take).
            alive = sorted(x for x in self.links_to.pop(p, ()) if self.fs.get(x, (0, 0, ""))[2] == "h")
            if alive:
                heir, rest = alive[0], alive[1:]
                self.link_of.pop(heir, None)
                self.fs[heir] = (e[0], e[1], "f")
                for x in rest:
                    self.link_of[x] = heir
                    self.links_to[heir].add(x)
                continue
            if e[0] <= 0:
                continue
            row = (p, e[0], e[1], by, how)
            self.waste.append(row)
            self.overwrites_by[by].append(row)

    def _apply(self, layer: Layer) -> None:
        i = layer.index
        for d in layer.opaque:
            for child in list(self.children.get(d, ())):
                self._drop(child, i, "hidden")
        for p in layer.whiteouts:
            self._drop(p, i, "deleted")
        for p, size, kind, target in layer.adds:
            # a/b replaces a lower-layer file named a, even when this layer has no entry for a
            a = posixpath.dirname(p)
            while a:
                e = self.fs.get(a)
                if e is not None:
                    if e[2] not in ("d", "l"):
                        self._drop(a, i, "overwritten")
                    break
                a = posixpath.dirname(a)
            old = self.fs.get(p)
            if old is not None:
                if old[2] == "d" and kind == "d":
                    continue   # a directory listed again (its metadata changed): nothing shipped twice
                self._drop(p, i, "overwritten")
            self.fs[p] = (size, i, kind)
            self._link(p)
            if kind == "h" and target:
                held = self.link_of.get(target, target)   # a link to a link shares the first file's data
                if self.fs.get(held, (0, 0, ""))[2] == "f":
                    self.link_of[p] = held
                    self.links_to[held].add(p)

    def _link(self, p: str) -> None:
        """Index p under its parent, and each missing ancestor under its own: a layer may
        list app/sub/b without an entry for app/sub, and deleting app must still reach b."""
        while p:
            parent = posixpath.dirname(p)
            kids = self.children[parent]
            if p in kids:
                return
            kids.add(p)
            p = parent

    def layer_label(self, index: int) -> str:
        L = self.layers[index - 1]
        return f"layer {index}" + (f" ({L.instruction[:120]})" if L.instruction else "")


def _ref(name: str) -> tuple:
    """(repository, tag or None) as Docker normalises a name: docker.io/library/ and
    docker.io/ are implied, and a colon after the last slash starts the tag."""
    name = name.split("@", 1)[0]
    repo, tag = name, None
    i = name.rfind(":")
    if i > name.rfind("/"):
        repo, tag = name[:i], name[i + 1:] or None
    for prefix in ("docker.io/", "index.docker.io/"):
        if repo.startswith(prefix):
            repo = repo[len(prefix):]
            break
    if repo.startswith("library/") and repo.count("/") == 1:
        repo = repo[len("library/"):]
    return repo, tag


def _pick(images: list, want: Any, path: str) -> dict:
    """The image `want` names: an exact tag, then tag:latest, then a repository name, then
    a platform. More than one match is an error naming them, never a silent first pick."""
    if want is None:
        return images[0]
    if not isinstance(want, str) or not want.strip():
        raise ValueError("'image' must be a non-empty string (a tag or platform from the archive)")
    w = want.strip()
    wr, wt = _ref(w)
    tiers = (
        lambda im: _image_name(im) == w,
        lambda im: w in im["tags"],
        # as Docker reads names: myapp:1 is docker.io/library/myapp:1, myapp is myapp:latest
        lambda im: any(_ref(t) == (wr, wt or "latest") for t in im["tags"]),
        lambda im: wt is None and any(_ref(t)[0] == wr for t in im["tags"]),
        lambda im: im.get("platform") == w,
    )
    names = "; ".join(_image_name(im) for im in images)
    for match in tiers:
        hits = [im for im in images if match(im)]
        if len(hits) == 1:
            return hits[0]
        if hits:
            raise ValueError(f"'{w}' matches several images in {path}: "
                             + "; ".join(_image_name(im) for im in hits)
                             + ". Pass one of these names exactly as written.")
    raise ValueError(f"no image matching '{w}' in {path}; it holds: {names}")


_JSONISH = re.compile(r"(?:\.json|^repositories|^blobs/[^/]+/[0-9a-f]+)$")


def _load(raw: Any, name: str = "path", want: Any = None) -> tuple[Image, list]:
    path = _resolve(raw, name)
    docs = _Docs()
    members: dict = {}
    links: dict = {}
    seen: dict = defaultdict(int)
    fh, comp = _open_file(path)
    tf = None
    try:
        outer = _Budget(path)
        # pass 1: names, and the JSON documents (manifest, index, configs) as bytes
        tf = _outer_tar(fh, comp, path)
        for m in tf:
            tf.members.clear()
            outer.charge(m.size)
            n = _norm(m.name)
            seen[n] += 1
            if m.issym() or m.islnk():
                links[n] = _link_target(m)
                continue
            if not m.isreg():
                continue
            members[n] = m
            if m.size <= MAX_JSON_BYTES and _JSONISH.search(n):
                f = tf.extractfile(m)
                head = f.read(64)
                if head.lstrip()[:1] in (b"{", b"["):
                    docs.keep(n, head + f.read(MAX_JSON_BYTES))
        images = _images(docs)
        chosen = _pick(images, want, path)
        others = [_image_name(im) for im in images if im is not chosen]
        if len(chosen["layers"]) > MAX_LAYERS:
            raise ValueError(f"the image has {len(chosen['layers'])} layers, over the {MAX_LAYERS} limit")

        def resolve(n: str) -> str:
            for _ in range(8):
                if n in members:
                    if seen[n] > 1:
                        raise ValueError(f"{n} appears {seen[n]} times in {path}; refusing an ambiguous archive")
                    return n
                if n not in links:
                    break
                n = links[n]
            raise ValueError(f"layer {n or '(unnamed)'} is not in {path}: the archive is incomplete. {_GUIDE}")

        layers = [Layer(i + 1, resolve(ref)) for i, ref in enumerate(chosen["layers"])]
        for L, d in zip(layers, chosen["layer_digests"]):
            L.digest = d if isinstance(d, str) else None
            L.archive_bytes = members[L.ref].size
        by_ref: dict = defaultdict(list)
        for L in layers:
            by_ref[L.ref].append(L)

        # pass 2: the layers themselves, from the same open file
        budget = _Budget("the layers")
        done: set = set()

        def scan(ref: str, fobj: Any) -> None:
            group = by_ref[ref]
            _scan_layer(fobj, group[0], budget)
            for L in group[1:]:
                L.copy_from(group[0])
            done.add(ref)

        if comp == "none":
            for ref in by_ref:
                scan(ref, tf.extractfile(members[ref]))
        else:   # a compressed archive is a stream: read it again from the start
            tf.close()
            tf = _outer_tar(fh, comp, path)
            for m in tf:
                tf.members.clear()
                n = _norm(m.name)
                if n in by_ref and n not in done:
                    scan(n, tf.extractfile(m))
        if len(done) != len(by_ref):
            raise ValueError(f"{path} changed while it was read; try again")

        config = docs.get(chosen["config"])
        if isinstance(config, dict) and isinstance(config.get("rootfs"), dict):
            diff_ids = config["rootfs"].get("diff_ids")
            if isinstance(diff_ids, list) and len(diff_ids) == len(layers):
                for L, d in zip(layers, diff_ids):
                    if isinstance(d, str):
                        L.diff_id = d
                        L.digest = L.digest or d
        meta = {"tags": chosen["tags"], "platform": chosen.get("platform")}
        if isinstance(config, dict) and not meta["platform"]:
            meta["platform"] = _platform(config)
        return Image(path, meta, layers, config), others
    except tarfile.TarError as exc:
        raise ValueError(f"{path} could not be read as an image archive ({exc}). {_GUIDE}") from None
    except (EOFError, OSError, lzma.LZMAError) as exc:
        raise ValueError(f"{path} is truncated or unreadable ({exc}). {_GUIDE}") from None
    finally:
        if tf is not None:
            tf.close()
        fh.close()


def _image_name(im: dict) -> str:
    name = ", ".join(im["tags"]) or "(untagged)"
    return name + (f" [{im['platform']}]" if im.get("platform") else "")


# --- Findings ---------------------------------------------------------------------------------
_HOME = r"(?:root|home/[^/]+)"
_LEFTOVERS = [
    ("apt package lists", re.compile(r"^var/lib/apt/lists/."),
     "Remove them in the same RUN as apt-get install: && rm -rf /var/lib/apt/lists/*"),
    ("apt package cache", re.compile(r"^var/cache/apt/.+\.(?:deb|bin)$"),
     "Run apt-get clean in the same RUN, or install with --no-install-recommends and clean up"),
    ("apk package cache", re.compile(r"^var/cache/apk/."), "Use apk add --no-cache"),
    ("yum/dnf cache", re.compile(r"^var/cache/(?:yum|dnf)/."), "Run yum clean all / dnf clean all in the same RUN"),
    ("pip cache", re.compile(rf"^{_HOME}/\.cache/pip/."),
     "Use pip install --no-cache-dir, or RUN --mount=type=cache,target=/root/.cache/pip"),
    ("uv cache", re.compile(rf"^{_HOME}/\.cache/uv/."), "Use uv's --no-cache, or a RUN --mount=type=cache"),
    ("npm cache", re.compile(rf"^{_HOME}/\.npm/_cacache/."),
     "Run npm cache clean --force in the same RUN, or use RUN --mount=type=cache,target=/root/.npm"),
    ("yarn cache", re.compile(rf"^(?:usr/local/share/\.cache/yarn|{_HOME}/\.cache/yarn)/."),
     "Run yarn cache clean in the same RUN"),
    ("Go build cache", re.compile(rf"^{_HOME}/\.cache/go-build/."), "Build in a separate stage, or use RUN --mount=type=cache"),
    ("Cargo registry", re.compile(rf"^{_HOME}/\.cargo/registry/."), "Build in a separate stage and copy only the binary"),
    ("files in /tmp", re.compile(r"^(?:tmp|var/tmp)/."), "Delete them in the RUN that created them"),
    (".git directory", re.compile(r"(?:^|/)\.git/."), "Add .git to .dockerignore"),
]
_ARCHIVE = re.compile(r"\.(?:tar|tgz|tar\.gz|tar\.xz|tar\.bz2|tar\.zst|zip|whl|deb|rpm|apk|jar)$")
_TOOLCHAIN = re.compile(r"^usr/(?:include/|lib/gcc/|libexec/gcc/|lib/llvm-)")
TOOLCHAIN_MIN = 50 * MIB
ARCHIVE_MIN = 5 * MIB


def _hsize(n: float) -> str:
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if abs(n) < 1000 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000
    return f"{n:.1f} TB"


def _sz(n: int) -> dict:
    return {"bytes": n, "size": _hsize(n)}


def _findings(img: Image) -> list:
    out = []
    # bytes shipped in one layer and removed by a later one
    groups: dict = defaultdict(lambda: [0, 0, []])
    for p, size, added, removed, how in img.waste:
        g = groups[(added, removed, how)]
        g[0] += size
        g[1] += 1
        g[2].append((size, p))
    for (added, removed, how), (size, count, paths) in sorted(groups.items(), key=lambda kv: -kv[1][0]):
        if size < FINDING_MIN:
            continue
        verb = {"deleted": "deletes", "overwritten": "replaces", "hidden": "hides (opaque directory)"}[how]
        later = img.layers[removed - 1]
        f = {"kind": "shipped_but_" + ("deleted" if how != "overwritten" else "overwritten"),
             **_sz(size), "files": count, "added_in": img.layer_label(added), "removed_in": img.layer_label(removed),
             "examples": [p for _, p in sorted(paths, reverse=True)[:5]],
             "detail": f"{img.layer_label(removed)} {verb} {count} file(s) that {img.layer_label(added)} added; "
                       f"they are invisible in the container but still downloaded and stored in layer {added}."}
        instr = (later.instruction or "").lower()
        if how == "overwritten" and re.search(r"\b(?:chmod|chown|chgrp)\b", instr):
            f["fix"] = ("A separate RUN chmod/chown copies every file it touches into a new layer. Use "
                        "COPY --chmod/--chown, or change permissions in the RUN that created the files.")
        elif how == "deleted":
            f["fix"] = "Delete the files in the same RUN that created them, or build them in a separate stage."
        out.append(f)
    # leftovers in the final filesystem
    hits: dict = defaultdict(lambda: [0, 0, defaultdict(int), []])
    toolchain = [0, defaultdict(int)]
    archives = []
    for p, (size, layer, kind) in img.fs.items():
        if kind != "f" or size == 0:
            continue
        for label, rx, _ in _LEFTOVERS:
            if rx.search(p):
                h = hits[label]
                h[0] += size
                h[1] += 1
                h[2][layer] += size
                h[3].append((size, p))
                break
        else:
            if _ARCHIVE.search(p) and size >= ARCHIVE_MIN:
                archives.append((size, p, layer))
        if _TOOLCHAIN.match(p):
            toolchain[0] += size
            toolchain[1][layer] += size
    for label, rx, fix in _LEFTOVERS:
        h = hits.get(label)
        if not h or h[0] < FINDING_MIN:
            continue
        top_layer = max(h[2].items(), key=lambda kv: kv[1])[0]
        out.append({"kind": "leftover", "what": label, **_sz(h[0]), "files": h[1],
                    "mostly_from": img.layer_label(top_layer),
                    "examples": [p for _, p in sorted(h[3], reverse=True)[:3]], "fix": fix})
    for size, p, layer in sorted(archives, reverse=True)[:10]:
        out.append({"kind": "archive_left_in_image", "path": p, **_sz(size), "added_in": img.layer_label(layer),
                    "fix": "If it was downloaded only to be unpacked or installed, delete it in the same RUN."})
    if toolchain[0] >= TOOLCHAIN_MIN:
        top_layer = max(toolchain[1].items(), key=lambda kv: kv[1])[0]
        out.append({"kind": "build_toolchain", **_sz(toolchain[0]), "mostly_from": img.layer_label(top_layer),
                    "detail": "Compiler and header files (/usr/include, gcc, llvm) are in the final image.",
                    "fix": "If they are only needed to build, a multi-stage build can leave them behind."})
    return out


def _dir_sizes(fs: dict, max_depth: int = 3) -> dict:
    sizes: dict = defaultdict(int)
    for p, (size, _, kind) in fs.items():
        if kind != "f" or not size:
            continue
        parts = p.split("/")
        for depth in range(1, min(max_depth, len(parts) - 1) + 1):
            sizes["/".join(parts[:depth])] += size
    return sizes


def _buckets(fs: dict, depth: int = 3) -> dict:
    """Each file counted once, under its directory cut to `depth` levels (usr/bin, usr/lib/python3.12)."""
    sizes: dict = defaultdict(int)
    for p, (size, _, kind) in fs.items():
        if kind == "f" and size:
            sizes["/".join(posixpath.dirname(p).split("/")[:depth])] += size
    return sizes


def _largest_dirs(sizes: dict, top: int) -> list:
    """Directories two or three levels down, skipping one that is mostly a single child."""
    biggest_child: dict = defaultdict(int)
    for d, size in sizes.items():
        if d.count("/") == 2:
            parent = d.rsplit("/", 1)[0]
            biggest_child[parent] = max(biggest_child[parent], size)
    rows = []
    for d, size in sizes.items():
        depth = d.count("/") + 1
        if depth not in (2, 3):
            continue
        if depth == 2 and biggest_child.get(d, 0) >= 0.8 * size:
            continue
        rows.append((size, d))
    rows.sort(reverse=True)
    return [{"dir": "/" + d, **_sz(s)} for s, d in rows[:top]]


def _layer_row(img: Image, L: Layer) -> dict:
    row = {"layer": L.index, "instruction": L.instruction, **_sz(L.size), "files": L.files}
    wasted = img.removed_from.get(L.index, 0)
    if wasted:
        row["removed_later"] = _sz(wasted)
    if L.compression != "none":
        row["compressed"] = _sz(L.archive_bytes)
    return row


def _header(img: Image, others: list) -> dict:
    shipped, final = img.shipped, img.final
    out: dict = {"ok": True, "file": img.path, "image": ", ".join(img.meta["tags"]) or "(untagged)"}
    if img.meta.get("platform"):
        out["platform"] = img.meta["platform"]
    out.update({"layers": len(img.layers), "shipped": _sz(shipped), "final_filesystem": _sz(final),
                "removed_by_later_layers": _sz(shipped - final)})
    if shipped:
        out["removed_share"] = round((shipped - final) / shipped, 3)
    if any(L.compression != "none" for L in img.layers):
        out["compressed"] = _sz(sum(L.archive_bytes for L in img.layers))
    if not any(L.instruction for L in img.layers):
        out["note"] = ("The image config's history does not line up with its layers, so layers are shown "
                       "without the instruction that made them.")
    if others:
        out["other_images_in_archive"] = others
    return out


def _err(message: str, **extra: Any) -> str:
    return json.dumps({"ok": False, "error": message, **extra})


def _int_arg(args: dict, key: str, default: int, cap: int, minimum: int = 1) -> int:
    v = args.get(key, default)
    if isinstance(v, bool) or not isinstance(v, int) or v < minimum:
        v = default
    return min(v, cap)


# --- Schemas -----------------------------------------------------------------------------------
_PATH_DESC = ("A saved container image: the tar from docker save -o image.tar <image> or podman save, "
              "or an OCI image archive (skopeo/buildah oci-archive). gzip, bzip2 and xz compression are read.")
_IMAGE_DESC = "When the archive holds several images or platforms: the tag or platform (linux/arm64) to read."

IMAGE_SUMMARY = {
    "name": "image_summary",
    "description": (
        "Read a local saved container image (docker save tar or OCI archive) and say where its size goes: "
        "every layer with the Dockerfile instruction that made it and its size, the size of the final "
        "filesystem, bytes that ship but are deleted or overwritten by a later layer (with the layer that "
        "added them and the one that removed them), caches and downloaded archives left in the image (apt "
        "lists, pip/npm/yarn caches, /tmp, .git, compilers) with the fix, and the biggest directories and "
        "files. Never runs docker or contacts a registry."
    ),
    "parameters": {"type": "object", "properties": {
        "path": {"type": "string", "description": _PATH_DESC},
        "image": {"type": "string", "description": _IMAGE_DESC},
        "top": {"type": "integer", "description": f"Directories and files to list. Default {DEFAULT_TOP}, capped at {HARD_TOP}."},
    }, "required": ["path"]},
}

IMAGE_LAYER = {
    "name": "image_layer",
    "description": (
        "Show one layer of a local saved container image: its instruction and size, the biggest files and "
        "directories it adds, what it deletes or overwrites in the layers below, and how much of it later "
        "layers delete or overwrite."
    ),
    "parameters": {"type": "object", "properties": {
        "path": {"type": "string", "description": _PATH_DESC},
        "layer": {"type": "integer", "description": "Layer number as image_summary lists them, 1 = the bottom (base) layer."},
        "image": {"type": "string", "description": _IMAGE_DESC},
        "top": {"type": "integer", "description": f"Files and directories to list. Default {DEFAULT_TOP}, capped at {HARD_TOP}."},
    }, "required": ["path", "layer"]},
}

IMAGE_COMPARE = {
    "name": "image_compare",
    "description": (
        "Compare two local saved container images (e.g. before and after a Dockerfile change): size of the "
        "final filesystem and of what ships, bytes removed by later layers, how many bottom layers they "
        "share, and the directories that grew, shrank, appeared or went."
    ),
    "parameters": {"type": "object", "properties": {
        "base": {"type": "string", "description": "The image before the change. " + _PATH_DESC},
        "head": {"type": "string", "description": "The image after the change."},
        "image": {"type": "string", "description": "Tag or platform to read from both archives when they hold several."},
        "top": {"type": "integer", "description": f"Directories to list per change. Default {DEFAULT_TOP}, capped at {HARD_TOP}."},
    }, "required": ["base", "head"]},
}


# --- Handlers (always return a JSON string) ----------------------------------------------------
def _files(fs_items: Any, top: int) -> list:
    rows = sorted(((e[0], p, e[1]) for p, e in fs_items if e[2] == "f"), reverse=True)[:top]
    return [{"path": "/" + p, **_sz(s), "layer": layer} for s, p, layer in rows]


def image_summary(args: dict, **kwargs) -> str:
    try:
        img, others = _load(args.get("path"), want=args.get("image"))
        top = _int_arg(args, "top", DEFAULT_TOP, HARD_TOP)
        out = _header(img, others)
        out["layer_list"] = [_layer_row(img, L) for L in img.layers]
        out["findings"] = _findings(img)
        sizes = _dir_sizes(img.fs)
        final = img.final
        out["by_top_dir"] = [{"dir": "/" + d, **_sz(s), "share": round(s / final, 3) if final else None}
                             for d, s in sorted(((d, s) for d, s in sizes.items() if "/" not in d),
                                                key=lambda kv: -kv[1])[:top]]
        out["largest_dirs"] = _largest_dirs(sizes, top)
        out["largest_files"] = _files(img.fs.items(), top)
        return json.dumps(out)
    except Exception as exc:  # the tool contract is a JSON string, never a raise
        return _err(str(exc))


def image_layer(args: dict, **kwargs) -> str:
    try:
        img, others = _load(args.get("path"), want=args.get("image"))
        top = _int_arg(args, "top", DEFAULT_TOP, HARD_TOP)
        n = args.get("layer")
        if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= len(img.layers):
            return _err(f"'layer' must be a layer number from 1 to {len(img.layers)}")
        L = img.layers[n - 1]
        out: dict = {"ok": True, "file": img.path, "image": ", ".join(img.meta["tags"]) or "(untagged)",
                     **_layer_row(img, L)}
        if L.digest:
            out["digest"] = L.digest
        if L.created:
            out["created"] = L.created
        adds = {p: (s, n, k) for p, s, k, _ in L.adds}
        out["largest_files"] = _files(adds.items(), top)
        out["largest_dirs"] = _largest_dirs(_dir_sizes(adds), top)
        later = defaultdict(lambda: [0, 0])
        for p, size, added, removed, how in img.waste:
            if added == n:
                later[(removed, how)][0] += size
                later[(removed, how)][1] += 1
        out["removed_by_later_layers"] = [
            {"by": img.layer_label(r), "how": how, **_sz(s), "files": c}
            for (r, how), (s, c) in sorted(later.items(), key=lambda kv: -kv[1][0])]
        own = sorted(img.overwrites_by.get(n, []), key=lambda w: -w[1])
        if own:
            out["removes_from_lower_layers"] = {
                **_sz(sum(w[1] for w in own)), "files": len(own),
                "largest": [{"path": "/" + p, **_sz(s), "how": how, "added_in": f"layer {a}"}
                            for p, s, a, _, how in own[:top]]}
        if L.whiteouts or L.opaque:
            out["deletes"] = ["/" + p for p in L.whiteouts[:top]] + [f"/{d}/* (opaque)" for d in L.opaque[:top]]
        if others:
            out["other_images_in_archive"] = others
        return json.dumps(out)
    except Exception as exc:
        return _err(str(exc))


def _shared_bottom(b: Image, h: Image) -> int:
    """Identical bottom layers. diff_ids (digests of the uncompressed layers, from each
    image's config) match across formats; blob digests differ once a layer is compressed."""
    n = 0
    for x, y in zip(b.layers, h.layers):
        if x.diff_id and y.diff_id:
            same = x.diff_id == y.diff_id
        else:
            same = bool(x.digest) and x.digest == y.digest
        if same:
            n += 1
        else:
            break
    return n


def image_compare(args: dict, **kwargs) -> str:
    try:
        b, _ = _load(args.get("base"), "base", want=args.get("image"))
        h, _ = _load(args.get("head"), "head", want=args.get("image"))
        top = _int_arg(args, "top", DEFAULT_TOP, HARD_TOP)
        out: dict = {"ok": True}
        for side, img in (("base", b), ("head", h)):
            hd = _header(img, [])
            hd.pop("ok", None)
            out[side] = hd
        for key, fb, fh in (("final_filesystem", b.final, h.final), ("shipped", b.shipped, h.shipped),
                            ("removed_by_later_layers", b.shipped - b.final, h.shipped - h.final)):
            out[key + "_change"] = {**_sz(fh - fb), "ratio": round(fh / fb, 3) if fb else None}
        shared = _shared_bottom(b, h)
        out["shared_bottom_layers"] = shared
        if shared:
            out["shared_note"] = f"The first {shared} layer(s) are identical (same digest): usually the base image."
        sb, sh = _buckets(b.fs), _buckets(h.fs)
        changes = []
        for d in set(sb) | set(sh):
            x, y = sb.get(d, 0), sh.get(d, 0)
            if x != y:
                changes.append((y - x, d, x, y))
        changes.sort()
        def rows(items: list) -> list:
            return [{"dir": "/" + d, "base": _sz(x)["size"], "head": _sz(y)["size"], **_sz(delta)}
                    for delta, d, x, y in items[:top]]
        out["grew"] = rows([c for c in reversed(changes) if c[0] > 0])
        out["shrank"] = rows([c for c in changes if c[0] < 0])
        fb_kinds = {f.get("what") or f["kind"]: f["bytes"] for f in _findings(b)}
        fh_kinds = {f.get("what") or f["kind"]: f["bytes"] for f in _findings(h)}
        fixed = sorted(set(fb_kinds) - set(fh_kinds))
        new = sorted(set(fh_kinds) - set(fb_kinds))
        if fixed:
            out["findings_gone"] = fixed
        if new:
            out["findings_new"] = new
        return json.dumps(out)
    except Exception as exc:
        return _err(str(exc))
