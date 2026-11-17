# layersift

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) plugin that reads a
saved container image and says where its size goes, with the Dockerfile
instruction behind each byte.

An image is a stack of layers. Each layer is a tar of the files one instruction
added or changed, and deleting a file in a later layer only hides it: the bytes
still ship in the layer that added them. `docker history` shows layer sizes but
not what is inside them, and nothing says which bytes are dead weight. layersift
applies the layers in order, as a container runtime does, and answers exactly.

## Tools

| Tool | What it answers |
|---|---|
| `image_summary` | Every layer with its instruction and size; the final filesystem size; bytes that ship but a later layer deletes or overwrites (which layer added them, which removed them); caches and archives left in the image (apt lists, apk/yum/dnf caches, pip/uv/npm/yarn caches, Go and Cargo caches, `/tmp`, `.git`, compilers and headers), each with the fix; the biggest directories and files. |
| `image_layer` | One layer: its biggest files and directories, what it deletes or overwrites below it, and how much of it later layers remove. |
| `image_compare` | Two images, before and after a Dockerfile change: size of what ships and of the final filesystem, layers shared at the bottom (the base image), directories that grew or shrank, findings fixed or introduced. |

Example, from a real `debian:bookworm-slim` image built with common mistakes:

```
layer 3 (RUN head -c 30000000 /dev/urandom > /tmp/blob.bin)  30.0 MB  removed later: 30.0 MB
layer 4 (RUN rm /tmp/blob.bin)                                0 B
findings:
  shipped_but_deleted  30.0 MB  added in layer 3, deleted in layer 4
                       fix: delete the files in the same RUN that created them
  leftover  apt package lists  19.9 MB  from layer 2 (RUN apt-get update && apt-get install ...)
                       fix: && rm -rf /var/lib/apt/lists/* in the same RUN
  leftover  .git directory  3.0 MB  from layer 5 (COPY app /app)
                       fix: add .git to .dockerignore
```

## Input

Save the image first; layersift never talks to Docker or a registry:

```
docker save -o image.tar myimage:tag
```

It reads:

- `docker save` output, both the classic layout (`<id>/layer.tar`) and the OCI-style
  layout Docker writes from version 25 (`blobs/sha256/...` with `index.json`);
- `podman save`;
- OCI image archives (`skopeo copy ... oci-archive:image.tar`, `buildah push`),
  including multi-platform indexes (pick one with `image: linux/arm64`);
- archives with several images (`docker save a b`; pick one with `image: a:tag`);
- gzip, bzip2 and xz compression of the archive or of its layers (zstd on Python 3.14+).

## Install

```
hermes plugins install rodricksz4h5/layersift --enable
```

## Limits

- Sizes are the bytes of regular files, as `docker export` would give them. They do
  not include filesystem block overhead, so `docker images` (which reports disk
  use) can show a somewhat larger number.
- Reading a layer means reading every tar header in it with Python's `tarfile`:
  about a second for a typical image, around ten seconds per hundred thousand files.
- Leftover findings are path patterns; a cache directory the image needs at run time
  will still be reported.

## Safety

Standard library only, no dependencies. No network, never runs `docker` or any other
program, no registry, no credentials, no background work, no self-update. Archives
are read, never extracted: nothing is written to disk. Only regular files are read;
`/proc` and `/dev` paths, FIFOs, sockets and Windows device paths are refused, and
every path goes through Hermes' read denylist. Archive size, tar entries and the
file bytes declared across layers are capped, so a crafted archive cannot make it
read without end.

## License

MIT
