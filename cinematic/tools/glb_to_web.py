"""Repackage the GLB for the artifact host: model.json (materials, texture names, part files),
part_<group>.json (geometry as base64) and tex<N>.jpg. Groups: body (everything not a head),
head_grin, head_closed - so the viewer loads one expression at a time."""
import base64, json, struct, sys
import numpy as np
from pathlib import Path
src, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
for f in list(out.glob('part_*.json')) + list(out.glob('tex*.*')) + list(out.glob('model.json')): f.unlink()
b = src.read_bytes()
jlen = struct.unpack_from('<I', b, 12)[0]
j = json.loads(b[20:20 + jlen]); bin0 = 20 + jlen + 8
def view(i):
    v = j['bufferViews'][i]; o = bin0 + v.get('byteOffset', 0); return b[o:o + v['byteLength']]
texfiles = []
for k, t in enumerate(j.get('textures', [])):
    img = j['images'][t['source']]; ext = 'jpg' if img['mimeType'] == 'image/jpeg' else 'png'
    name = f'tex{k}.{ext}'; (out / name).write_bytes(view(img['bufferView'])); texfiles.append(name)
groups = {}
for n in j['nodes']:
    me = j['meshes'][n['mesh']]
    grp = 'head_grin' if me['name'].startswith('head_grin_') else 'head_closed' if me['name'].startswith('head_closed_') else 'body'
    for pr in me['primitives']:
        attrs = {}
        for key, ai in pr['attributes'].items():
            a = j['accessors'][ai]; arr = np.frombuffer(view(a['bufferView']), np.float32).reshape(a['count'], -1)
            if key == 'POSITION':
                lo, hi = arr.min(0), arr.max(0); sc = (hi - lo) / 65535 + 1e-12
                q = np.round((arr - lo) / sc).astype(np.uint16)
                attrs[key] = {'type': a['type'], 'enc': 'u16', 'lo': lo.tolist(), 'sc': sc.tolist(), 'data': base64.b64encode(q.tobytes()).decode()}
            elif key == 'NORMAL':
                attrs[key] = {'type': a['type'], 'enc': 'i8', 'data': base64.b64encode(np.round(np.clip(arr, -1, 1) * 127).astype(np.int8).tobytes()).decode()}
            elif key == 'COLOR_0':
                attrs[key] = {'type': a['type'], 'enc': 'u8', 'data': base64.b64encode(np.round(np.clip(arr, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8).tobytes()).decode()}
            else:
                attrs[key] = {'type': a['type'], 'data': base64.b64encode(arr.tobytes()).decode()}
        ia = j['accessors'][pr['indices']]
        groups.setdefault(grp, []).append({'name': me['name'], 'material': pr['material'], 'attributes': attrs, 'indices': base64.b64encode(view(ia['bufferView'])).decode()})
parts = {}
for g, meshes in groups.items():
    js = json.dumps({'meshes': meshes}, separators=(',', ':')); (out / f'part_{g}.json').write_text(js); parts[g] = {'file': f'part_{g}.json', 'bytes': len(js)}
    print(g, len(js))
(out / 'model.json').write_text(json.dumps({'materials': j['materials'], 'textures': texfiles, 'parts': parts, 'extras': j.get('extras', {})}))
print('textures', len(texfiles))
