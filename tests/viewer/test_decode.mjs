// usage: node test_decode.mjs <file.dk2v.txt> <expected.json>
import { readFileSync } from 'node:fs';
import { decodeDk2vText, parseDk2v, framePositions } from '../../viewer/dk2v.js';

const [file, expectedFile] = process.argv.slice(2);
const want = JSON.parse(readFileSync(expectedFile, 'utf8'));
const model = parseDk2v(await decodeDk2vText(readFileSync(file, 'utf8')));
const walk = model.animations.walk;
if (walk.parts.length !== want.part_count) { console.error('FAIL part count'); process.exit(1); }
const got = framePositions(walk.parts[0], want.frame, model.bbox);
want.first_positions.forEach((v, i) => { if (Math.abs(got[i] - v) > 1e-4) { console.error('FAIL position', i); process.exit(1); } });
console.log('ok');
