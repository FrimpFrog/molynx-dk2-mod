"""A minimal glTF 2.0 binary (GLB) writer: meshes with positions, normals, UVs, optional vertex
colours (AO), uint32 indices and PBR materials with embedded images (JPEG/PNG bytes)."""
import json
import struct

import numpy as np


class GLB:
    def __init__(self):
        self.bin = bytearray()
        self.j = {"asset": {"version": "2.0", "generator": "molynx_cinematic"}, "scene": 0,
                  "scenes": [{"nodes": []}], "nodes": [], "meshes": [], "accessors": [],
                  "bufferViews": [], "materials": [], "textures": [], "images": [], "samplers": [
                      {"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}]}

    def _view(self, data: bytes, target=None):
        while len(self.bin) % 4:
            self.bin.append(0)
        off = len(self.bin)
        self.bin += data
        v = {"buffer": 0, "byteOffset": off, "byteLength": len(data)}
        if target:
            v["target"] = target
        self.j["bufferViews"].append(v)
        return len(self.j["bufferViews"]) - 1

    def _acc(self, arr, kind, comp, target, minmax=False):
        arr = np.ascontiguousarray(arr)
        view = self._view(arr.tobytes(), target)
        a = {"bufferView": view, "componentType": comp, "count": int(arr.shape[0]), "type": kind}
        if minmax:
            a["min"] = arr.min(0).astype(float).tolist()
            a["max"] = arr.max(0).astype(float).tolist()
        self.j["accessors"].append(a)
        return len(self.j["accessors"]) - 1

    def image(self, data: bytes, mime: str):
        view = self._view(data)
        self.j["images"].append({"bufferView": view, "mimeType": mime})
        self.j["textures"].append({"sampler": 0, "source": len(self.j["images"]) - 1})
        return len(self.j["textures"]) - 1

    def material(self, name, base_tex=None, mr_tex=None, normal_tex=None, emissive_tex=None,
                 color=(1, 1, 1, 1), metallic=0.0, roughness=0.5, emissive=(0, 0, 0), extras=None,
                 normal_scale=1.0, double=False):
        pbr = {"baseColorFactor": list(color), "metallicFactor": metallic, "roughnessFactor": roughness}
        if base_tex is not None:
            pbr["baseColorTexture"] = {"index": base_tex}
        if mr_tex is not None:
            pbr["metallicRoughnessTexture"] = {"index": mr_tex}
        m = {"name": name, "pbrMetallicRoughness": pbr, "doubleSided": double}
        if normal_tex is not None:
            m["normalTexture"] = {"index": normal_tex, "scale": normal_scale}
        if emissive_tex is not None:
            m["emissiveTexture"] = {"index": emissive_tex}
        if any(emissive):
            m["emissiveFactor"] = list(emissive)
        if extras:
            m["extras"] = extras
        self.j["materials"].append(m)
        return len(self.j["materials"]) - 1

    def mesh(self, name, P, N, idx, material, UV=None, color=None, glow=None):
        attrs = {"POSITION": self._acc(P.astype(np.float32), "VEC3", 5126, 34962, True),
                 "NORMAL": self._acc(N.astype(np.float32), "VEC3", 5126, 34962)}
        if UV is not None:
            attrs["TEXCOORD_0"] = self._acc(UV.astype(np.float32), "VEC2", 5126, 34962)
        if color is not None:
            attrs["COLOR_0"] = self._acc(color.astype(np.float32), "VEC3", 5126, 34962)
        if glow is not None:
            attrs["_GLOW"] = self._acc(np.asarray(glow, np.float32).reshape(-1), "SCALAR", 5126, 34962)
        ind = self._acc(np.asarray(idx, np.uint32).reshape(-1), "SCALAR", 5125, 34963)
        self.j["meshes"].append({"name": name, "primitives": [{"attributes": attrs, "indices": ind, "material": material}]})
        self.j["nodes"].append({"name": name, "mesh": len(self.j["meshes"]) - 1})
        self.j["scenes"][0]["nodes"].append(len(self.j["nodes"]) - 1)

    def save(self, path):
        for k in ("textures", "images"):
            if not self.j[k]:
                del self.j[k]
        if "textures" not in self.j:
            del self.j["samplers"]
        while len(self.bin) % 4:
            self.bin.append(0)
        self.j["buffers"] = [{"byteLength": len(self.bin)}]
        js = json.dumps(self.j, separators=(",", ":")).encode()
        js += b" " * ((4 - len(js) % 4) % 4)
        out = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(self.bin))
        out += struct.pack("<II", len(js), 0x4E4F534A) + js
        out += struct.pack("<II", len(self.bin), 0x004E4942) + bytes(self.bin)
        with open(path, "wb") as f:
            f.write(out)
        return len(out)
