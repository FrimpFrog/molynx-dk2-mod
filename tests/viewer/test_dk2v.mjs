// usage: node test_dk2v.mjs <file.dk2v> <expected.json>; exits non-zero on mismatch
import { readFileSync } from 'node:fs';
import { parseDk2v, framePositions, blend, textureFile } from '../../viewer/dk2v.js';

const [file, expectedFile] = process.argv.slice(2);
const buf = readFileSync(file);
const model = parseDk2v(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
const want = JSON.parse(readFileSync(expectedFile, 'utf8'));
const fail = (m) => { console.error('FAIL', m); process.exit(1); };

const walk = model.animations.walk;
if (walk.parts.length !== want.part_count) fail(`part count ${walk.parts.length} != ${want.part_count}`);
walk.parts.forEach((p, i) => { if (p.vertexCount !== want.vertex_counts[i]) fail(`vertex count part ${i}`); });
const got = framePositions(walk.parts[0], want.frame, model.bbox);
want.first_positions.forEach((v, i) => { if (Math.abs(got[i] - v) > 1e-4) fail(`position ${i}: ${got[i]} vs ${v}`); });
const mid = blend(new Float32Array([0, 10]), new Float32Array([10, 20]), 0.5);
if (mid[0] !== 5 || mid[1] !== 15) fail('blend midpoint');
if (textureFile('GUI\\Creatures\\reaperMM0') !== 'GUI__Creatures__reaperMM0.png') fail('textureFile');
if (textureFile('#ADD#King-CleaveMM0') !== '_ADD_King-CleaveMM0.png') fail('textureFile #');
if (textureFile('Creature Bed_Goblin INMM0') !== 'Creature_Bed_Goblin_INMM0.png') fail('textureFile space');
console.log('ok');
