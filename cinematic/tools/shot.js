const { chromium } = require('playwright');
const http = require('http'), fs = require('fs'), path = require('path');
(async () => {
  const root = process.argv[2], page_ = process.argv[3], out = process.argv[4], w = +(process.argv[5]||400), h = +(process.argv[6]||300);
  const srv = http.createServer((q, s) => { const p = path.join(root, decodeURIComponent(q.url.split('?')[0])); fs.readFile(p, (e, d) => { if (e) { s.writeHead(404); return s.end(); } const ext = path.extname(p); s.writeHead(200, {'Content-Type': ext === '.js' ? 'text/javascript' : ext === '.html' ? 'text/html' : 'application/octet-stream'}); s.end(d); }); }).listen(0);
  const port = srv.address().port;
  const b = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const p = await b.newPage({ viewport: { width: w, height: h } });
  p.on('console', m => console.log('console:', m.text()));
  p.on('pageerror', e => console.log('pageerror:', e.message));
  await p.goto(`http://127.0.0.1:${port}/${page_}`);
  await p.waitForFunction(() => document.title.startsWith('ok') || document.title.startsWith('err'), null, { timeout: 600000 });
  console.log('title:', await p.title());
  await p.screenshot({ path: out, timeout: 400000 });
  await b.close(); srv.close();
})();
