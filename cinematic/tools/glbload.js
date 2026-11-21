// Minimal GLB loader for three.js: meshes, normals, UVs, vertex colours, PBR materials with
// embedded images, plus Molynx-specific material extras (clearcoat, bump map, sheen, emissive power).
import * as THREE from './three.module.js';

const COMP = { 5126: Float32Array, 5125: Uint32Array, 5123: Uint16Array, 5121: Uint8Array };
const SIZE = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 };
// ember glow inside the rock: a per-vertex glow amount (attribute _GLOW) added as emission
function addGlow(mat, color, power) {
  mat.userData.glow = true;
  mat.onBeforeCompile = (sh) => {
    sh.uniforms.glowColor = { value: new THREE.Color().fromArray(color).multiplyScalar(power) };
    sh.vertexShader = 'attribute float glow;\nvarying float vGlow;\n' + sh.vertexShader.replace('#include <begin_vertex>', '#include <begin_vertex>\nvGlow = glow;');
    sh.fragmentShader = 'uniform vec3 glowColor;\nvarying float vGlow;\n' + sh.fragmentShader.replace('#include <emissivemap_fragment>', '#include <emissivemap_fragment>\ntotalEmissiveRadiance += glowColor * vGlow;');
  };
  mat.customProgramCacheKey = () => 'glow';
}


export async function loadGLB(url) {
  const buf = await (await fetch(url)).arrayBuffer();
  const dv = new DataView(buf);
  const jlen = dv.getUint32(12, true);
  const json = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 20, jlen)));
  const binOff = 20 + jlen + 8;
  const view = (i) => { const v = json.bufferViews[i]; return [binOff + (v.byteOffset || 0), v.byteLength]; };
  const acc = (i) => { const a = json.accessors[i]; const [o] = view(a.bufferView); const T = COMP[a.componentType]; return new T(buf, o, a.count * SIZE[a.type]); };
  const textures = await Promise.all((json.textures || []).map(async (t) => {
    const img = json.images[t.source]; const [o, l] = view(img.bufferView);
    const bmp = await createImageBitmap(new Blob([new Uint8Array(buf, o, l)], { type: img.mimeType }), { imageOrientation: 'none' });
    const tex = new THREE.Texture(bmp); tex.flipY = false; tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.anisotropy = 8; tex.needsUpdate = true; return tex;
  }));
  const srgb = (t) => { t.colorSpace = THREE.SRGBColorSpace; return t; };
  const mats = json.materials.map((m) => {
    const p = m.pbrMetallicRoughness || {}; const x = m.extras || {};
    const mat = new THREE.MeshPhysicalMaterial({
      name: m.name, color: new THREE.Color().fromArray(p.baseColorFactor || [1, 1, 1]),
      metalness: p.metallicFactor ?? 0, roughness: p.roughnessFactor ?? 0.5,
      side: m.doubleSided ? THREE.DoubleSide : THREE.FrontSide,
    });
    if (p.baseColorTexture) mat.map = srgb(textures[p.baseColorTexture.index].clone());
    if (p.metallicRoughnessTexture) { mat.roughnessMap = mat.metalnessMap = textures[p.metallicRoughnessTexture.index]; }
    if (m.normalTexture && !x.bump) { mat.normalMap = textures[m.normalTexture.index]; const s = m.normalTexture.scale ?? 1; mat.normalScale.set(s, -s); }
    if (m.emissiveTexture) mat.emissiveMap = srgb(textures[m.emissiveTexture.index].clone());
    if (m.emissiveFactor) mat.emissive = new THREE.Color().fromArray(m.emissiveFactor);
    if (x.emissiveIntensity) mat.emissiveIntensity = x.emissiveIntensity;
    if (x.bump !== undefined) { mat.bumpMap = textures[x.bump]; mat.bumpScale = x.bumpScale ?? 1; }
    if (x.clearcoat) { mat.clearcoat = x.clearcoat; mat.clearcoatRoughness = x.clearcoatRoughness ?? 0.2; }
    if (x.sheen) { mat.sheen = x.sheen; mat.sheenColor = new THREE.Color().fromArray(x.sheenColor || [1, 1, 1]); mat.sheenRoughness = x.sheenRoughness ?? 0.5; }
    if (x.specularIntensity !== undefined) mat.specularIntensity = x.specularIntensity;
    if (x.vertexColors) mat.vertexColors = true;
    if (x.glowColor) addGlow(mat, x.glowColor, x.glowPower ?? 1);
    if (x.envMapIntensity !== undefined) mat.envMapIntensity = x.envMapIntensity;
    return mat;
  });
  const group = new THREE.Group();
  for (const n of json.nodes) {
    const me = json.meshes[n.mesh];
    for (const pr of me.primitives) {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(acc(pr.attributes.POSITION), 3));
      g.setAttribute('normal', new THREE.BufferAttribute(acc(pr.attributes.NORMAL), 3));
      if (pr.attributes.TEXCOORD_0 !== undefined) g.setAttribute('uv', new THREE.BufferAttribute(acc(pr.attributes.TEXCOORD_0), 2));
      if (pr.attributes.COLOR_0 !== undefined) g.setAttribute('color', new THREE.BufferAttribute(acc(pr.attributes.COLOR_0), 3));
      if (pr.attributes._GLOW !== undefined) g.setAttribute('glow', new THREE.BufferAttribute(acc(pr.attributes._GLOW), 1));
      g.setIndex(new THREE.BufferAttribute(acc(pr.indices), 1));
      const mesh = new THREE.Mesh(g, mats[pr.material]); mesh.name = me.name;
      mesh.castShadow = true; mesh.receiveShadow = true;
      group.add(mesh);
    }
  }
  group.userData = json.extras || {};
  return group;
}
