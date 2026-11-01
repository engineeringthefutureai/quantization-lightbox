const SAMPLE = "sample.png", WIZARD = "wizard.png";
const params = new URLSearchParams(window.location.search);
const hideOrangeFrame =
  params.get("frame") === "0" ||
  params.get("frame") === "false" ||
  params.get("frame") === "none" ||
  params.get("frame") === "off" ||
  params.get("frame") === "hidden" ||
  params.get("frame") === "hide" ||
  params.get("orangeframe") === "0" ||
  params.get("orangeframe") === "false" ||
  params.get("orangeframe") === "none" ||
  params.get("orangeframe") === "off" ||
  params.get("orangeframe") === "hidden" ||
  params.get("orangeframe") === "hide" ||
  (params.has("hideframe") && params.get("hideframe") !== "0" && params.get("hideframe") !== "false") ||
  (params.has("noframe") && params.get("noframe") !== "0" && params.get("noframe") !== "false") ||
  (params.has("hideorangeframe") && params.get("hideorangeframe") !== "0" && params.get("hideorangeframe") !== "false");
const MAXW = 900, WIN = 64, ERRMAX = 0.15;
const METHODS = [
  {id:"orig", name:"Original", tag:"8-bit source", analog:"The full-precision weights, standing in for FP16 or BF16."},
  {id:"naive", name:"Naive uniform grid", tag:"one range for everything", analog:"Round-to-nearest over one global range. Faint values collapse to zero, like small weights crushed by outliers."},
  {id:"block", name:"Block-wise scale", tag:"≈ Q4_1 / Q5_1", analog:"Each block gets its own scale and minimum, so dim regions get a fine grid of their own."},
  {id:"diff", name:"Block + error diffusion", tag:"≈ GPTQ", analog:"Same grid, but each rounding error is pushed onto values not yet quantized, so averages survive."},
  {id:"mixed", name:"Mixed precision", tag:"≈ Q4_K_M", analog:"Bits follow detail: busy blocks get one more, flat blocks one fewer, same average budget."},
  {id:"mixdiff", name:"Mixed + diffusion", tag:"≈ EXL2", analog:"Both ideas together. Usually the lowest error per bit of anything here."},
];
// LLM counterpart labels follow the bit width: b-bit codes here correspond to Qb in GGUF
const TAGS = {
  orig: () => "8-bit source",
  naive: b => `per-tensor INT${b}`,
  block: b => ({1: "no GGUF peer (IQ1 uses codebooks)", 2: "≈ Q2_K", 3: "≈ Q3_K", 4: "≈ Q4_1", 5: "≈ Q5_1", 6: "≈ Q6_K", 7: "no 7-bit GGUF type", 8: "≈ Q8_0"})[b],
  diff: b => b === 1 ? "no 1-bit GPTQ" : `≈ GPTQ ${b}-bit`,
  mixed: b => b === 1 ? "≈ UD-IQ1 dynamic" : b === 2 ? "≈ Q2_K mix" : b <= 5 ? `≈ Q${b}_K_M` : "no GGUF mix at this width",
  mixdiff: (b, bpv) => `≈ EXL2 ${bpv}bpw`,
};
const state = {bits:2, B:8, view:"img", W:0, H:0, ch:null, out:{}, cost:{}, focus:null};
const $ = id => document.getElementById(id);
const fmt = n => n.toLocaleString("en-US");
const kb = bits => (bits / 8 / 1024).toLocaleString("en-US", {maximumFractionDigits: 1}) + " KiB";

// ---------- UI scaffold ----------
const grid = $("grid"), strip = $("strip"), cards = {};
for (const m of METHODS) {
  const el = document.createElement("article"); el.className = "card";
  el.innerHTML = `<canvas aria-label="${m.name}"></canvas>
    <div class="head"><h3 class="name">${m.name}</h3><span class="tag">${m.tag}</span></div>
    <p class="analog">${m.analog}</p>
    <div class="stats"><div><span>Bits / value</span><b data-k="bpv">–</b></div><div><span>RMSE</span><b data-k="rmse">–</b></div><div><span>PSNR</span><b data-k="psnr">–</b></div></div>`;
  grid.appendChild(el);
  const cv = el.querySelector("canvas");
  cv.addEventListener("click", e => {
    const r = cv.getBoundingClientRect();
    state.focus = [Math.min(state.W - 1, Math.floor((e.clientX - r.left) / r.width * state.W)), Math.min(state.H - 1, Math.floor((e.clientY - r.top) / r.height * state.H))];
    drawMarkers(); drawStrip(); inspect();
  });
  const fig = document.createElement("figure");
  fig.innerHTML = `<canvas width="${WIN}" height="${WIN}"></canvas><figcaption>${m.name}</figcaption>`;
  strip.appendChild(fig);
  cards[m.id] = {cv, ctx: cv.getContext("2d"), stats: el, zcv: fig.querySelector("canvas")};
}

function segment(id, key, parse) {
  const seg = $(id);
  seg.addEventListener("click", e => {
    const b = e.target.closest("button"); if (!b) return;
    seg.querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", x === b));
    state[key] = parse(b.dataset.v); key === "view" ? render() : key === "mch" ? microscope() : schedule();
  });
}
segment("blockSeg", "B", Number);
segment("viewSeg", "view", String);
$("bits").addEventListener("input", () => { state.bits = +$("bits").value; $("bitsOut").textContent = state.bits; schedule(); });
$("pick").onclick = () => $("file").click();
$("file").onchange = e => e.target.files[0] && loadFile(e.target.files[0]);
$("sample").onclick = () => loadURL(SAMPLE);
$("wizard").onclick = () => loadURL(WIZARD);
addEventListener("dragover", e => e.preventDefault());
addEventListener("drop", e => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f && f.type.startsWith("image/")) loadFile(f); });

// ---------- loading ----------
function loadFile(f) { const u = URL.createObjectURL(f); loadURL(u, () => URL.revokeObjectURL(u)); }
function loadURL(url, done) {
  const img = new Image();
  img.onload = () => {
    const s = Math.min(1, MAXW / img.naturalWidth);
    const W = Math.round(img.naturalWidth * s), H = Math.round(img.naturalHeight * s);
    const c = document.createElement("canvas"); c.width = W; c.height = H;
    const x = c.getContext("2d"); x.drawImage(img, 0, 0, W, H);
    const src = x.getImageData(0, 0, W, H), d = src.data, n = W * H;
    const ch = [new Float32Array(n), new Float32Array(n), new Float32Array(n)];
    for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) ch[k][i] = d[i * 4 + k] / 255;
    Object.assign(state, {W, H, ch, src, focus: null});
    for (const m of METHODS) { cards[m.id].cv.width = W; cards[m.id].cv.height = H; }
    done && done(); compute();
  };
  img.src = url;
}

// ---------- quantizers (operate on one channel, values in [0,1]) ----------
function naive(v, bits) {
  const L = (1 << bits) - 1, o = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) o[i] = Math.round(v[i] * L) / L;
  return o;
}
function blockGrid(v, W, H, B, bitsOf) {
  const bx = Math.ceil(W / B), by = Math.ceil(H / B), n = bx * by;
  const mn = new Float32Array(n), st = new Float32Array(n), Ls = new Int32Array(n);
  for (let b = 0; b < n; b++) {
    const x0 = (b % bx) * B, y0 = Math.floor(b / bx) * B;
    let lo = 1, hi = 0;
    for (let y = y0; y < Math.min(y0 + B, H); y++) for (let x = x0; x < Math.min(x0 + B, W); x++) { const t = v[y * W + x]; if (t < lo) lo = t; if (t > hi) hi = t; }
    const L = (1 << bitsOf[b]) - 1;
    if (L === 0) { mn[b] = fp((lo + hi) / 2); st[b] = 0; } // zero-bit block: one value, the midpoint
    else { mn[b] = fp(lo); st[b] = fp((hi - lo) / L); }
    Ls[b] = L;
  }
  return {bx, mn, st, Ls};
}
function blockQuant(v, W, H, B, bitsOf, diffuse) {
  const g = blockGrid(v, W, H, B, bitsOf), o = new Float32Array(v.length);
  const w = diffuse ? Float32Array.from(v) : v;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = y * W + x, b = Math.floor(y / B) * g.bx + Math.floor(x / B);
    const t = w[i], s = g.st[b];
    const q = s > 0 ? Math.min(g.Ls[b], Math.max(0, Math.round((t - g.mn[b]) / s))) : 0;
    const r = g.mn[b] + q * s; o[i] = r;
    if (diffuse) {
      const e = t - r;
      if (x + 1 < W) w[i + 1] += e * 7 / 16;
      if (y + 1 < H) {
        if (x > 0) w[i + W - 1] += e * 3 / 16;
        w[i + W] += e * 5 / 16;
        if (x + 1 < W) w[i + W + 1] += e / 16;
      }
    }
  }
  if (diffuse) o.w = w;
  return o;
}
// bit allocation: rank blocks by luminance std; top quarter +1 bit, bottom quarter -1
function allocate(W, H, B, bits) {
  const bx = Math.ceil(W / B), by = Math.ceil(H / B), n = bx * by, sd = new Float32Array(n), cnt = new Int32Array(n);
  const [R, G, Bl] = state.ch;
  for (let b = 0; b < n; b++) {
    const x0 = (b % bx) * B, y0 = Math.floor(b / bx) * B; let s = 0, s2 = 0, c = 0;
    for (let y = y0; y < Math.min(y0 + B, H); y++) for (let x = x0; x < Math.min(x0 + B, W); x++) {
      const i = y * W + x, l = .2126 * R[i] + .7152 * G[i] + .0722 * Bl[i]; s += l; s2 += l * l; c++;
    }
    sd[b] = Math.sqrt(Math.max(0, s2 / c - (s / c) ** 2)); cnt[b] = c;
  }
  const order = [...sd.keys()].sort((a, b) => sd[a] - sd[b]);
  const out = new Int32Array(n).fill(bits), q = Math.floor(n / 4);
  for (let k = 0; k < q; k++) { out[order[k]] = bits - 1; out[order[n - 1 - k]] = bits + 1; }
  return {bitsOf: out, cnt};
}

// ---------- compute ----------
let pending = 0;
function schedule() { cancelAnimationFrame(pending); pending = requestAnimationFrame(compute); }
function stat(q) {
  const {ch, W, H} = state, n = W * H, err = new Float32Array(n); let se = 0;
  for (let i = 0; i < n; i++) {
    let a = 0;
    for (let k = 0; k < 3; k++) { const d = q[k][i] - ch[k][i]; se += d * d; a += Math.abs(d); }
    err[i] = a / 3;
  }
  return {err, rmse: Math.sqrt(se / (n * 3))};
}
function setStats(el, bpv, rmse) {
  const set = (k, t) => el.querySelector(`[data-k=${k}]`).textContent = t;
  set("bpv", bpv); set("rmse", rmse.toFixed(4)); set("psnr", rmse === 0 ? "∞" : (20 * Math.log10(1 / rmse)).toFixed(1) + " dB");
}
function compute() {
  if (!state.ch) return;
  const {W, H, B, bits, ch} = state, N = W * H * 3;
  const nb = Math.ceil(W / B) * Math.ceil(H / B), uni = new Int32Array(nb).fill(bits);
  const {bitsOf: mix, cnt} = allocate(W, H, B, bits);
  let mixPayload = 0; for (let b = 0; b < nb; b++) mixPayload += mix[b] * cnt[b] * 3;
  const blockMeta = nb * 3 * 32;
  state.cost = {
    N, nb,
    orig: {payload: N * 8, meta: 0},
    naive: {payload: N * bits, meta: 0},
    block: {payload: N * bits, meta: blockMeta},
    diff: {payload: N * bits, meta: blockMeta},
    mixed: {payload: mixPayload, meta: blockMeta + nb * 2},
    mixdiff: {payload: mixPayload, meta: blockMeta + nb * 2},
  };
  state.mix = mix;
  const runs = {
    orig: () => ch,
    naive: () => ch.map(v => naive(v, bits)),
    block: () => ch.map(v => blockQuant(v, W, H, B, uni, false)),
    diff: () => ch.map(v => blockQuant(v, W, H, B, uni, true)),
    mixed: () => ch.map(v => blockQuant(v, W, H, B, mix, false)),
    mixdiff: () => ch.map(v => blockQuant(v, W, H, B, mix, true)),
  };
  for (const m of METHODS) {
    const q = runs[m.id](), {err, rmse} = stat(q), c = state.cost[m.id];
    state.out[m.id] = {q, err, rmse};
    setStats(cards[m.id].stats, ((c.payload + c.meta) / N).toFixed(2), rmse);
    cards[m.id].stats.querySelector(".tag").textContent = TAGS[m.id](bits, ((c.payload + c.meta) / N).toFixed(2));
  }
  if (!state.focus) { // default to the 8x8 patch where block scaling helps most over naive
    const {W, H} = state, a = state.out.naive.err, b = state.out.block.err; let best = -1, bp = null;
    for (let y = 0; y + 8 <= H; y += 8) for (let x = 0; x + 8 <= W; x += 8) {
      let d = 0; for (let j = 0; j < 8; j++) for (let i = 0; i < 8; i++) { const p = (y + j) * W + x + i; d += a[p] - b[p]; }
      if (d > best) { best = d; bp = [x + 4, y + 4]; }
    }
    state.focus = bp;
  }
  render(); explain(); inspect(); costTable();
}

// ---------- rendering ----------
const STOPS = [[0,0,4],[40,11,84],[101,21,110],[159,42,99],[212,72,66],[245,125,21],[250,193,39],[252,255,164]];
function cmap(t) {
  t = Math.min(1, Math.max(0, t)) * (STOPS.length - 1);
  const i = Math.min(STOPS.length - 2, Math.floor(t)), f = t - i, a = STOPS[i], b = STOPS[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}
$("legendBar").style.background = `linear-gradient(90deg, ${STOPS.map(c => `rgb(${c})`).join(",")})`;
function toImage({q, err}) {
  const {W, H, view} = state, img = new ImageData(W, H), d = img.data;
  for (let i = 0; i < W * H; i++) {
    if (view === "err") { const c = cmap(err[i] / ERRMAX); d[i*4] = c[0]; d[i*4+1] = c[1]; d[i*4+2] = c[2]; }
    else { d[i*4] = q[0][i] * 255; d[i*4+1] = q[1][i] * 255; d[i*4+2] = q[2][i] * 255; }
    d[i*4+3] = 255;
  }
  return img;
}
const images = {};
function render() {
  if (!state.ch) return;
  $("legend").hidden = state.view !== "err";
  for (const m of METHODS) images[m.id] = toImage(state.out[m.id]);
  drawMarkers(); drawStrip();
}
function focusPt() { return state.focus || [state.W >> 1, state.H >> 1]; }
function focusBox() {
  const {W, H} = state, f = focusPt();
  return [Math.max(0, Math.min(W - WIN, f[0] - WIN / 2)), Math.max(0, Math.min(H - WIN, f[1] - WIN / 2))];
}
function drawMarkers() {
  const [x, y] = focusBox();
  $("where").textContent = state.focus ? `x ${x + WIN/2}, y ${y + WIN/2}` : "center";
  for (const m of METHODS) {
    const {ctx} = cards[m.id]; ctx.putImageData(images[m.id], 0, 0);
    if (!hideOrangeFrame) {
      ctx.strokeStyle = "#f2b44c"; ctx.lineWidth = Math.max(2, state.W / 300); ctx.strokeRect(x + .5, y + .5, WIN, WIN);
    }
  }
}
function drawStrip() {
  const [x, y] = focusBox();
  for (const m of METHODS) cards[m.id].zcv.getContext("2d").putImageData(images[m.id], -x, -y, x, y, WIN, WIN);
}

// ---------- math sections ----------
function explain() {
  const {W, H, bits} = state, N = W * H * 3, L = (1 << bits) - 1;
  $("rawEq").textContent =
`${fmt(W)} × ${fmt(H)} pixels = ${fmt(W * H)} pixels
${fmt(W * H)} pixels × 3 channels = ${fmt(N)} values
${fmt(N)} values × 8 bits = ${fmt(N * 8)} bits = ${fmt(N)} bytes ≈ ${kb(N * 8)}`;
  const levels = []; for (let q = 0; q <= Math.min(L, 7); q++) levels.push((q / L * 255).toFixed(0));
  $("naiveGrid").textContent = `at ${bits} bits, ${L + 1} levels: ${levels.join(", ")}${L > 7 ? ", …" : ""} on the 0–255 scale`;
}

function costTable() {
  const c = state.cost, N = c.N, raw = N * 8;
  const rows = [
    ["Raw 8-bit RGB", "orig", "one byte per value", true],
    ["Raw FP16", null, "what an LLM starts from", true],
    ["Naive uniform grid", "naive", "range fixed at 0–1"],
    ["Block-wise scale", "block", `${fmt(c.nb)} blocks × 3 channels × 32 bits`],
    ["Block + error diffusion", "diff", "identical storage to block-wise"],
    ["Mixed precision", "mixed", "block scales + 2-bit width tag per block"],
    ["Mixed + diffusion", "mixdiff", "identical storage to mixed"],
  ];
  let html = `<thead><tr><th>Version</th><th>Codes</th><th>Metadata</th><th>Total</th><th>Bits / value</th><th>vs raw 8-bit</th></tr></thead><tbody>`;
  for (const [label, id, note, ref] of rows) {
    const p = id ? c[id].payload : N * 16, m = id ? c[id].meta : 0, t = p + m;
    html += `<tr class="${ref ? "ref" : ""}"><td>${label}<span class="sub">${note}</span></td><td>${kb(p)}</td><td>${m ? kb(m) : "0"}</td><td>${kb(t)}</td><td>${(t / N).toFixed(3)}</td><td>${(t / raw * 100).toFixed(1)}%</td></tr>`;
  }
  $("costTable").innerHTML = html + "</tbody>";
}

function f16(v) { // float32 -> IEEE half bits
  const f = new Float32Array([v]), u = new Uint32Array(f.buffer)[0];
  const s = (u >>> 16) & 0x8000, e = ((u >>> 23) & 0xff) - 127 + 15, m = u & 0x7fffff;
  if (e <= 0) { if (e < -10) return s; const mm = (m | 0x800000) >> (1 - e); return s | ((mm + 0x1000) >> 13); }
  if (e >= 31) return s | 0x7c00;
  return s | (e << 10) | ((m + 0x1000) >> 13);
}
function h2f(h) {
  const sg = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
  if (e === 0) return sg * Math.pow(2, -14) * m / 1024;
  if (e === 31) return m ? NaN : sg * Infinity;
  return sg * Math.pow(2, e - 15) * (1 + m / 1024);
}
function fp(v) { return h2f(f16(v)); }
const hex16 = v => "0x" + f16(v).toString(16).padStart(4, "0");
function inspect() {
  if (!state.ch) return;
  const {W, H, B, bits} = state, v = state.ch[0], L = (1 << bits) - 1;
  const [fx, fy] = focusPt(), bxi = Math.floor(fx / B), byi = Math.floor(fy / B);
  const x0 = bxi * B, y0 = byi * B, x1 = Math.min(x0 + B, W), y1 = Math.min(y0 + B, H);
  let lo = 1, hi = 0;
  for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) { const t = v[y * W + x]; lo = Math.min(lo, t); hi = Math.max(hi, t); }
  lo = fp(lo); hi = Math.max(hi, lo);
  const s = fp((hi - lo) / L), count = (x1 - x0) * (y1 - y0);
  const bIdx = byi * Math.ceil(W / B) + bxi, mixBits = state.mix[bIdx];
  $("blockKv").innerHTML = `
    <dt>Block</dt><dd>column ${bxi}, row ${byi}</dd>
    <dt>Pixels</dt><dd>x ${x0}–${x1 - 1}, y ${y0}–${y1 - 1} (${count} values)</dd>
    <dt>Bits per code</dt><dd>b = ${bits}, so L = ${L}</dd>
    <dt>Minimum m</dt><dd style="color:var(--meta)">${lo.toFixed(4)} (byte ${Math.round(lo * 255)}) · fp16 ${hex16(lo)}</dd>
    <dt>Maximum M</dt><dd>${hi.toFixed(4)} (byte ${Math.round(hi * 255)})</dd>
    <dt>Step s</dt><dd style="color:var(--meta)">(M − m) / L = ${s.toFixed(5)} · fp16 ${hex16(s)}</dd>
    <dt>Step in bytes</dt><dd>${(s * 255).toFixed(2)} vs ${(255 / L).toFixed(2)} for the naive grid</dd>
    <dt>Storage</dt><dd>${count} × ${bits} + 32 = ${count * bits + 32} bits, ${((count * bits + 32) / count).toFixed(3)} per value</dd>
    <dt>Mixed precision</dt><dd>this block gets ${mixBits} bit${mixBits > 1 ? "s" : ""} (${mixBits > bits ? "among the busiest" : mixBits < bits ? "among the flattest" : "middle"} quarter${mixBits === bits ? "s" : ""})</dd>`;
  const codeBits = count * bits, total = codeBits + 32;
  $("layout").innerHTML = `<div class="ls" style="flex:16">s · 16</div><div class="lm" style="flex:16">m · 16</div><div class="lc" style="flex:${codeBits}">${count} codes × ${bits} = ${codeBits} bits</div>`;
  const n = Math.min(8, x1 - x0); let rows = "", packed = [];
  for (let k = 0; k < n; k++) {
    const t = v[y0 * W + x0 + k], u = s > 0 ? (t - lo) / s : 0, q = Math.min(L, Math.max(0, Math.round(u))), r = lo + q * s;
    const nq = Math.round(t * L) / L;
    const bin = q.toString(2).padStart(bits, "0"); packed.push(bin);
    rows += `<tr><td>${x0 + k}</td><td>${Math.round(t * 255)}</td><td>${u.toFixed(2)}</td><td style="color:var(--code)">${q}</td><td style="color:var(--code)">${bin}</td><td>${Math.round(r * 255)}</td><td>${Math.round((r - t) * 255) >= 0 ? "+" : ""}${Math.round((r - t) * 255)}</td><td>${Math.round(nq * 255)}</td></tr>`;
  }
  $("rowLabel").textContent = `Top row of the block, first ${n} values (red channel, 0–255 scale)`;
  $("rowTable").innerHTML = `<thead><tr><th>x</th><th>Byte</th><th>(x − m) / s</th><th>Code q</th><th>Binary</th><th>Decoded</th><th>Error</th><th>Naive decodes</th></tr></thead><tbody>${rows}</tbody>`;
  microscope();
  $("packed").innerHTML = packed.map(b => `<span>${b}</span>`).join(" ") + `<span class="dim" style="color:var(--muted)"> … ${count - n} more codes</span>`;
}


// ---------- microscope ----------
state.mch = 0; state.sel = null;
segment("chanSeg", "mch", Number);
const CH = ["red", "green", "blue"];
const mixc = (a, b, t) => `rgb(${a.map((x, i) => Math.round(x + (b[i] - x) * t)).join(",")})`;
const PANEL = [22, 20, 28], AMBER = [242, 180, 76], NEG = [127, 178, 255], POS = [255, 122, 89];
const esc = t => String(t).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
function chanCell(val, k) {
  const c = [0, 0, 0]; c[k] = Math.round(val);
  const lum = .2126 * c[0] + .7152 * c[1] + .0722 * c[2];
  return {bg: `rgb(${c})`, fg: lum > 110 ? "#000" : "#fff"};
}
function matrix(title, vals, cellFn, tsv, mid, cls) {
  const cells = vals.map((v, i) => {
    const c = cellFn(v, i), k = cls ? cls(i) : "";
    return `<div class="cell ${k}" style="background:${c.bg};color:${c.fg}"${mid ? ` data-mid="${mid}" data-i="${i}"` : ""}${c.tip ? ` title="${c.tip}"` : ""}><span class="v">${c.text}</span>${c.bin != null ? `<span class="bn">${c.bin}</span>` : ""}</div>`;
  }).join("");
  return `<figure class="mat"><figcaption><span class="lbl">${title}</span><button class="copy" data-tsv="${tsv}">Copy</button></figcaption><div class="cells${state.B >= 16 ? " dense" : ""}" style="grid-template-columns:repeat(${state.B},1fr)">${cells}</div></figure>`;
}
const binHTML = (x, n) => { const b = x.toString(2).padStart(n, "0"); if (n <= 4) return b; const h = Math.ceil(n / 2); return b.slice(0, h) + "<br>" + b.slice(h); };
const toTSV = a => { const P = state.B, r = []; for (let y = 0; y < P; y++) r.push(a.slice(y * P, y * P + P).join("\t")); return r.join("&#10;"); };
const sgn = n => (n > 0 ? "+" : "") + n;
const n5 = v => v.toFixed(5);

// one fp16 value -> sign | exponent | mantissa fields, decoded inline
function halfHTML(h, name, hl) {
  const sg = h >> 15, e = (h >> 10) & 31, M = h & 1023, val = h2f(h), bin = (x, n) => x.toString(2).padStart(n, "0");
  const cells = (str, c) => [...str].map(b => `<i class="${c}">${b}</i>`).join("");
  let expL, manL, dec;
  if (e === 0 && M === 0) { expL = "0"; manL = "0"; dec = `${name} = 0`; }
  else if (e === 0) { expL = "0 → subnormal, 2^−14"; manL = `${M}/1024 = ${(M / 1024).toFixed(4)}`; dec = `${name} = ${sg ? "−" : "+"} ${(M / 1024).toFixed(4)} × 2^−14 = ${val.toPrecision(5)}`; }
  else { expL = `${e} − 15 = ${e - 15}`; manL = `1 + ${M}/1024 = ${(1 + M / 1024).toFixed(4)}`; dec = `${name} = ${sg ? "−" : "+"} ${(1 + M / 1024).toFixed(4)} × 2^${e - 15} = ${val.toPrecision(5)}`; }
  return `<div class="half ${hl}">
    <div class="fields"><span class="hname">${name}</span>
      <span class="fld"><span class="fb">${cells(bin(sg, 1), "b-sign")}</span><span class="fl">${sg ? "−" : "+"}</span></span>
      <span class="fld"><span class="fb">${cells(bin(e, 5), "b-exp")}</span><span class="fl">${expL}</span></span>
      <span class="fld"><span class="fb">${cells(bin(M, 10), "b-man")}</span><span class="fl">${manL}</span></span></div>
    <div class="hdec">${dec} <span class="dim">· ${(val * 255).toFixed(3)} on the 0–255 scale</span></div></div>`;
}
function metaBlockHTML(g, b, bxCount, mixed, baseBits, hl) {
  let html = halfHTML(f16(g.st[b]), "s", hl) + halfHTML(f16(g.mn[b]), "m", hl);
  if (mixed) {
    const t = state.mix[b] - (baseBits - 1), tb = t.toString(2).padStart(2, "0");
    html += `<div class="half ${hl}"><div class="fields"><span class="hname">tag</span><span class="fld"><span class="fb">${[...tb].map(c => `<i class="b-tag">${c}</i>`).join("")}</span><span class="fl">${t} → ${baseBits} − 1 + ${t} = ${state.mix[b]}-bit codes</span></span></div></div>`;
  }
  return `<div class="mblock"><span class="lbl">Block col ${b % bxCount}, row ${Math.floor(b / bxCount)}</span><div class="halves">${html}</div></div>`;
}
const FP16_NOTE = `<p class="note"><b>Why fp16.</b> A block's step and minimum are arbitrary real numbers, so they have to be stored as floats, and 16 bits is the smallest standard float: IEEE 754 half precision, value = ± (1 + mantissa/1024) × 2^(exponent − 15). The 10-bit mantissa gives about three significant digits, plenty for a scale, and the exponent lets the same 16 bits describe a step of 0.0001 or 100. A 32-bit float would double the metadata cost for no visible gain. GGUF made the same choice: Q4_0, Q4_1 and Q8_0 keep one fp16 scale per 32 weights, and k-quants keep fp16 numbers per 256-weight super-block. Rounding the scale to fp16 is itself a tiny extra quantization, which is why the equations below use these exact values.</p>`;

const BM_COL = {lo: [127, 178, 255], mid: [120, 112, 138], hi: [242, 180, 76]};
function bitMapHTML() {
  const b = state.bits, n = state.mix.length, cnt = {};
  state.mix.forEach(w => cnt[w] = (cnt[w] || 0) + 1);
  const avg = state.mix.reduce((a, x) => a + x, 0) / n;
  const row = (w, key, label) => cnt[w] ? `<div class="bmrow"><span class="swatch" style="background:rgb(${BM_COL[key]})"></span><b>${w} bit${w === 1 ? "" : "s"}</b><span>${fmt(cnt[w])} blocks · ${(cnt[w] / n * 100).toFixed(0)}% · ${label}</span></div>` : "";
  const legend = row(b + 1, "hi", "busiest quarter") + row(b, "mid", "middle half") + row(b - 1, "lo", b === 1 ? "flattest quarter, no codes" : "flattest quarter");
  return `<div class="bitmap"><canvas id="bitmapCv" aria-label="Bit width per block"></canvas>
    <div class="bmside"><span class="lbl">Bit width per ${state.B}×${state.B} block</span>${legend}
      <p class="note">Average ${avg.toFixed(2)} bits, the same budget as plain ${b}-bit. Click the map to move the patch.</p>
      <p class="note"><b>Only three widths here, by design:</b> b − 1, b and b + 1, assigned by quartile of brightness detail. That keeps the 2-bit tag sufficient. At 1 bit, the flattest blocks drop to 0 bits: they store no codes at all, and every pixel decodes to the block’s midpoint m. Real formats choose differently: the _S/_M/_L GGUF types mix whole tensors, typically jumping two or more bits for the important ones, and EXL2 picks anywhere from 2 to 8 bits for slices of each layer.</p></div></div>`;
}
function drawBitMap() {
  const cv = $("bitmapCv"); if (!cv) return;
  const {W, H, B, bits, ch} = state, bx = Math.ceil(W / B);
  cv.width = W; cv.height = H;
  const img = new ImageData(W, H), d = img.data;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = y * W + x, w = state.mix[Math.floor(y / B) * bx + Math.floor(x / B)];
    const c = BM_COL[w > bits ? "hi" : w < bits ? "lo" : "mid"];
    const l = (.2126 * ch[0][i] + .7152 * ch[1][i] + .0722 * ch[2][i]) * 255 * .5;
    for (let k = 0; k < 3; k++) d[i * 4 + k] = l * .45 + c[k] * .55;
    d[i * 4 + 3] = 255;
  }
  const ctx = cv.getContext("2d"); ctx.putImageData(img, 0, 0);
  const [x0, y0] = patchOrigin();
  ctx.strokeStyle = "#fff"; ctx.lineWidth = Math.max(2, W / 300);
  const pad = Math.max(4, W / 120); ctx.strokeRect(x0 - pad, y0 - pad, B + 2 * pad, B + 2 * pad);
  cv.onclick = e => {
    const r = cv.getBoundingClientRect();
    state.focus = [Math.min(W - 1, Math.floor((e.clientX - r.left) / r.width * W)), Math.min(H - 1, Math.floor((e.clientY - r.top) / r.height * H))];
    drawMarkers(); drawStrip(); inspect();
  };
}
function drawContext(x0, y0) {
  const {W, H, ch, B: P} = state, N = 3 * P, Z = 192 / N;
  const cx = Math.max(0, Math.min(W - N, x0 - P)), cy = Math.max(0, Math.min(H - N, y0 - P));
  $("ctxNote").textContent = `${N}×${N} pixels around the block, magnified ${Z}×. The frame marks the ${P}×${P} block inspected below. Pick a channel by clicking its tile.`;
  const views = [["ctx-c", null], ["ctx-0", 0], ["ctx-1", 1], ["ctx-2", 2]];
  for (const [id, k] of views) {
    const cv = $(id), g = cv.getContext("2d");
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const i = (cy + y) * W + cx + x, c = [0, 0, 0];
      if (k === null) for (let j = 0; j < 3; j++) c[j] = Math.round(ch[j][i] * 255);
      else c[k] = Math.round(ch[k][i] * 255);
      g.fillStyle = `rgb(${c})`; g.fillRect(x * Z, y * Z, Z, Z);
    }
    const fx = (x0 - cx) * Z, fy = (y0 - cy) * Z;
    g.lineWidth = 4; g.strokeStyle = "rgba(0,0,0,.85)"; g.strokeRect(fx, fy, P * Z, P * Z);
    g.lineWidth = 2; g.strokeStyle = "#fff"; g.strokeRect(fx, fy, P * Z, P * Z);
  }
  document.querySelectorAll("#ctxrow .pick").forEach(b => b.setAttribute("aria-pressed", +b.dataset.ch === state.mch));
}
$("ctxrow").addEventListener("click", e => {
  const b = e.target.closest(".pick"); if (!b) return;
  state.mch = +b.dataset.ch;
  $("chanSeg").querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", +x.dataset.v === state.mch));
  microscope();
});
// "In LLMs" text that follows the current bit width
const LLM_TEXT = {
  naive: b => `Per-tensor INT${b} round-to-nearest${b === 8 ? ", the classic INT8 baseline" : ""}. No GGUF type works this way, because a single outlier would stretch the range for the whole tensor.`,
  block: b => ({
    1: `No GGUF type stores 1-bit codes with a scale and minimum. The types near 1 bit work differently: <b>IQ1_S</b> and <b>IQ1_M</b> (about 1.6–1.8 bits per weight) pick each small group of weights from a fixed codebook, and <b>TQ1_0</b> stores ternary −1/0/+1 weights for BitNet-style models.`,
    2: `<b>Q2_K</b>: 2-bit codes with a scale and minimum per 16-weight sub-block; those scales are themselves 4-bit numbers under fp16 values per 256 weights, about 2.6 bits per weight. The <b>IQ2</b> types reach similar sizes with codebooks instead of a uniform grid.`,
    3: `<b>Q3_K</b>: 3-bit codes, symmetric (scale only, no minimum), with 6-bit scales per 16 weights under one fp16 scale per 256, about 3.4 bits per weight.`,
    4: `<b>Q4_1</b>: an fp16 scale and fp16 minimum per 32 weights, exactly this scheme. <b>Q4_0</b> drops the minimum. <b>Q4_K</b> adds a super-block level with 6-bit sub-block scales, 4.5 bits per weight.`,
    5: `<b>Q5_1</b>: an fp16 scale and fp16 minimum per 32 weights with 5-bit codes, exactly this scheme. <b>Q5_0</b> drops the minimum; <b>Q5_K</b> adds the super-block level, 5.5 bits per weight.`,
    6: `<b>Q6_K</b>: 6-bit codes, symmetric, with 8-bit scales per 16 weights under one fp16 scale per 256, about 6.6 bits per weight.`,
    7: `No GGUF type uses 7-bit codes; the ladder goes from Q6_K straight to Q8_0.`,
    8: `<b>Q8_0</b>: 8-bit codes with one fp16 scale per 32 weights and no minimum, 8.5 bits per weight. It is close enough to lossless that people use it as the reference quant.`,
  })[b],
  diff: b => b === 1
    ? `<b>GPTQ</b> doesn’t go down to 1 bit. It is an algorithm, not a GGUF suffix: it decides which codes to store and leaves the format alone. llama.cpp’s nearest tool is the <b>imatrix</b>, which picks scales to protect important weights but does not push error onward.`
    : `<b>GPTQ ${b}-bit</b> (Hugging Face repos tagged <span class="mono">-GPTQ</span>). ${b === 2 ? "2-bit GPTQ exists but usually breaks models; this is where newer methods such as QuIP# and AQLM took over." : b === 3 ? "3-bit works on large models and gets shaky on small ones." : b === 4 ? "4-bit with group size 128 is the most common release." : b === 8 ? "8-bit is near-lossless and rarely needed." : "Most releases are 4-bit or 8-bit."} It is an algorithm, not a GGUF suffix: it decides which codes to store and leaves the format alone. llama.cpp’s nearest tool is the <b>imatrix</b>, which picks scales to protect important weights but does not push error onward.`,
  mixed: b => (b === 1
    ? `Closest are the lowest dynamic quants, such as Unsloth’s <span class="mono">UD-IQ1_S</span>, which keep sensitive tensors at higher precision and squeeze the rest toward 1.5 bits. The 0-bit blocks here go further than any real format.`
    : b === 2
    ? `<b>Q2_K</b> as a file type already keeps some tensors at higher precision, and <b>Q2_K_S</b> mixes less. Dynamic quants such as <span class="mono">UD-Q2_K_XL</span> choose widths per layer.`
    : b <= 5
    ? `<b>Q${b}_K_M</b> keeps some attention-value and feed-forward-down tensors at higher precision and the rest at Q${b}_K; <b>Q${b}_K_S</b> mixes less${b === 3 ? ", <b>Q3_K_L</b> more" : ""}.`
    : b === 6
    ? `<b>Q6_K</b> has no _S/_M/_L variants; at this width there is little left to gain from mixing.`
    : b === 7
    ? `No GGUF mix targets 7 bits.`
    : `<b>Q8_0</b> has no mixed variants; it is already near-lossless.`) + ` Here the choice is per block, finer than any of them.`,
  mixdiff: (b, bpv) => `<b>EXL2</b> (ExLlamaV2) at about <span class="mono">${bpv}bpw</span>: GPTQ-style error compensation plus mixed bit widths chosen by measuring each layer’s error.${+bpv < 2 ? " That is below EXL2’s practical floor of about 2 bits per weight." : ""}`,
};

function patchOrigin() {
  const {W, H, B: P} = state, [fx, fy] = focusPt();
  return [Math.min(Math.floor(fx / P), Math.floor(W / P) - 1) * P, Math.min(Math.floor(fy / P), Math.floor(H / P) - 1) * P];
}
function microscope() {
  if (!state.ch || state.W < state.B || state.H < state.B) return;
  const {W, H, B, bits} = state, k = state.mch, v = state.ch[k];
  const P = B, NN = P * P, [x0, y0] = patchOrigin();
  if (state.sel && state.sel.i >= NN) state.sel = null;
  $("scopeWhere").textContent = `x ${x0}–${x0 + P - 1}, y ${y0}–${y0 + P - 1} (${CH[k]} channel)`;
  drawContext(x0, y0);
  const at = (arr, i) => arr[(y0 + Math.floor(i / P)) * W + x0 + (i % P)];
  const orig = Array.from({length: NN}, (_, i) => Math.round(at(v, i) * 255));
  const nb = Math.ceil(W / B) * Math.ceil(H / B), bx = Math.ceil(W / B);
  const grids = {uni: blockGrid(v, W, H, B, new Int32Array(nb).fill(bits)), mix: blockGrid(v, W, H, B, state.mix)};
  const blockOf = i => Math.floor((y0 + Math.floor(i / P)) / B) * bx + Math.floor((x0 + (i % P)) / B);
  const sel = state.sel;

  let html = `<div class="mrow"><div class="mrow-head"><h3 class="name">Original</h3><span class="meta">${P} × ${P} px × 8 bits = ${NN * 8} bits</span></div><p class="llm"><span class="lbl">In LLMs</span> The unquantized weights: <b>F16</b> or <b>BF16</b> in GGUF, or the original safetensors.</p><div class="trio">`;
  html += matrix("Raw bytes, 0–255", orig, x => ({...chanCell(x, k), text: x, bin: binHTML(x, 8)}), toTSV(orig), null, i => sel && sel.i === i ? "hl" : "");
  html += `<div class="side"><p>These are the bytes as the image stores them: one number per pixel for the ${CH[k]} channel, 0 is off and 255 is full brightness. Each square is colored by its own value, so the grid is literally the patch magnified.</p>
    <p><b>Stored</b> shows the integer codes a quantized file keeps, colored from <span class="swatch" style="background:rgb(${PANEL})"></span>0 to <span class="swatch" style="background:rgb(${AMBER})"></span>the largest code. Below each method, the metadata the decoder also needs is drawn bit by bit; hover or tap a byte to see what it encodes.</p>
    <p><b>Decoded</b> is what you get back on the same 0–255 scale. <b>Error</b> is decoded minus original: <span class="swatch" style="background:rgb(${NEG})"></span>came out darker, <span class="swatch" style="background:rgb(${POS})"></span>came out brighter, colors saturating at ±48.</p>
    <p><b>Click any pixel</b> in a method's grids to see the exact equations that produced it.</p></div>`;
  html += `</div></div>`;

  const defs = [
    {id: "naive", name: "Naive uniform grid", grid: null, llm: LLM_TEXT.naive},
    {id: "block", name: "Block-wise scale", grid: "uni", llm: LLM_TEXT.block},
    {id: "diff", name: "Block + error diffusion", grid: "uni", diffuse: true, llm: LLM_TEXT.diff},
    {id: "mixed", name: "Mixed precision", grid: "mix", llm: LLM_TEXT.mixed},
    {id: "mixdiff", name: "Mixed + diffusion", grid: "mix", diffuse: true, llm: LLM_TEXT.mixdiff},
  ];
  for (const d of defs) {
    const qch = state.out[d.id].q[k];
    const dec01 = Array.from({length: NN}, (_, i) => at(qch, i));
    const dec = dec01.map(x => Math.round(x * 255)), err = dec.map((x, i) => x - orig[i]);
    const g = d.grid ? grids[d.grid] : null;
    const codes = [], Lof = [], bw = [];
    for (let i = 0; i < NN; i++) {
      if (!g) { const L = (1 << bits) - 1; codes.push(Math.round(dec01[i] * L)); Lof.push(L); bw.push(bits); continue; }
      const b = blockOf(i), s = g.st[b];
      codes.push(s > 0 ? Math.round((dec01[i] - g.mn[b]) / s) : 0); Lof.push(g.Ls[b]); bw.push(d.grid === "mix" ? state.mix[b] : bits);
    }
    // header arithmetic
    const groups = new Map(); bw.forEach(w => groups.set(w, (groups.get(w) || 0) + 1));
    const codeBits = bw.reduce((a, b) => a + b, 0);
    const codeExpr = groups.size === 1 ? `${P} × ${P} px × ${bw[0]} bits` : [...groups].sort((a, b) => b[0] - a[0]).map(([w, n]) => `${n} px × ${w} bits`).join(" + ");
    const blocks = new Map(); for (let i = 0; i < NN; i++) { const b = blockOf(i); blocks.set(b, (blocks.get(b) || 0) + 1); }
    let metaBits = 0, metaExpr;
    const per = d.grid === "mix" ? 34 : 32;
    if (!g) { metaExpr = "0 metadata bits"; }
    else {
      for (const [b, n] of blocks) { const size = Math.min(B, W - (b % bx) * B) * Math.min(B, H - Math.floor(b / bx) * B); metaBits += per * n / size; }
      const what = d.grid === "mix" ? "s, m as fp16 + 2-bit tag" : "s and m, 16 bits each";
      metaExpr = `${per} metadata bits (${what})`;
    }
    const total = codeBits + metaBits;
    const mae = err.reduce((a, b) => a + Math.abs(b), 0) / NN, mx = Math.max(...err.map(Math.abs)), rmse = Math.sqrt(err.reduce((a, b) => a + b * b, 0) / NN);
    const isSel = sel && sel.mid === d.id, si = isSel ? sel.i : -1;
    const nbs = new Set();
    if (isSel && d.diffuse) { const r = Math.floor(si / P), c = si % P; [[0, 1], [1, -1], [1, 0], [1, 1]].forEach(([dr, dc]) => { const rr = r + dr, cc = c + dc; if (rr < P && cc >= 0 && cc < P) nbs.add(rr * P + cc); }); }
    const cls = i => i === si ? "hl" : nbs.has(i) ? "nb" : "";

    html += `<div class="mrow"><div class="mrow-head"><h3 class="name">${d.name}</h3></div>
      <p class="llm"><span class="lbl">In LLMs</span> ${d.llm(bits, ((state.cost[d.id].payload + state.cost[d.id].meta) / state.cost.N).toFixed(2))}</p>${d.id === "mixed" ? bitMapHTML() : d.id === "mixdiff" ? `<p class="note">Uses the same bit-width map as mixed precision above.</p>` : ""}
      <div class="arith mono"><span class="c">${codeExpr} = ${codeBits} code bits</span> + <span class="m">${metaExpr}</span> = ${+total.toFixed(2)} bits → ${(total / NN).toFixed(2)} bits/value<br><span class="dim">mean |error| ${mae.toFixed(1)} · max ${mx} · RMSE ${rmse.toFixed(1)}</span></div><div class="trio">`;
    const widths = [...groups.keys()].sort().join("/");
    html += matrix(`Stored codes (${widths} bit${widths === "1" ? "" : "s"})`, codes,
      (q, i) => { const t = Lof[i] ? q / Lof[i] : 0; return {bg: mixc(PANEL, AMBER, t), fg: t > .55 ? "#000" : "#ece8f2", text: bw[i] ? q : "–", bin: bw[i] ? binHTML(q, bw[i]) : null}; },
      toTSV(codes), d.id, cls);
    html += matrix("Decoded, 0–255", dec, x => ({...chanCell(x, k), text: x, bin: binHTML(x, 8)}), toTSV(dec), d.id, i => i === si ? "hl" : "");
    html += matrix("Error, decoded − original", err,
      e => { const t = Math.min(1, Math.abs(e) / 48); return {bg: mixc(PANEL, e < 0 ? NEG : POS, t), fg: t > .6 ? "#000" : "#ece8f2", text: e === 0 ? "0" : sgn(e)}; },
      toTSV(err), d.id, i => i === si ? "hl" : "");
    html += `</div>`;

    // metadata strip
    if (!g) {
      html += `<div class="metawrap"><span class="lbl">Metadata</span><p class="explain">None. The decoder already knows the grid: ${bits} bits give L = ${(1 << bits) - 1}, so code q always means q × 255/${(1 << bits) - 1} = q × ${(255 / ((1 << bits) - 1)).toFixed(2)}.</p></div>`;
    } else {
      const selB = isSel ? blockOf(si) : -1;
      const strips = [...blocks.keys()].map(b => metaBlockHTML(g, b, bx, d.grid === "mix", bits, selB < 0 ? "" : b === selB ? "hl" : "dim")).join("");
      html += `<div class="metawrap"><span class="lbl">Metadata, bit by bit</span>
        <div class="legend-fields"><span><span class="swatch" style="background:#3a3546"></span>sign</span><span><span class="swatch" style="background:#2f5c9e"></span>exponent</span><span><span class="swatch" style="background:#9cc3ff"></span>mantissa</span>${d.grid === "mix" ? `<span><span class="swatch" style="background:var(--accent)"></span>width tag</span>` : ""}</div>
        <div class="strips">${strips}</div>${d.id === "block" ? FP16_NOTE : ""}</div>`;
    }

    // equations for the selected pixel
    if (isSel) {
      const i = si, x = at(v, i), xb = orig[i], q = codes[i], xh = dec01[i], L = Lof[i];
      const px = `x ${x0 + (i % P)}, y ${y0 + Math.floor(i / P)}`;
      let eq = `<span class="dim">Pixel ${px}, row ${Math.floor(i / P) + 1} col ${(i % P) + 1} of the block</span>\n\n<b>Quantize</b>\n`;
      eq += `x  = byte / 255 = ${xb} / 255 = ${n5(x)}\n`;
      let xin = x;
      if (d.diffuse) {
        xin = at(qch.w, i);
        eq += `x′ = x + error pushed in from earlier pixels = ${n5(x)} ${xin - x < 0 ? "−" : "+"} ${n5(Math.abs(xin - x))} = ${n5(xin)}\n`;
      }
      const xv = d.diffuse ? "x′" : "x";
      if (!g) {
        eq += `L  = 2^${bits} − 1 = ${L}\n`;
        eq += `q  = clamp(round(${xv} × L), 0, L) = round(${n5(xin)} × ${L}) = round(${(xin * L).toFixed(3)}) = <span class="c">${q}</span>   stored as <span class="c">${q.toString(2).padStart(bw[i], "0")}</span>\n\n`;
        eq += `<b>Dequantize</b>\nx̂  = q / L = ${q} / ${L} = ${n5(xh)}  →  × 255 = ${(xh * 255).toFixed(2)} ≈ ${dec[i]}\n\n`;
      } else {
        const b = blockOf(i), m = g.mn[b], s = g.st[b];
        eq += `L  = 2^${bw[i]} − 1 = ${L}${d.grid === "mix" ? `   <span class="dim">(width from the block's tag)</span>` : ""}\n`;
        if (L === 0) {
          eq += `<span class="m">m</span>  = ${n5(m)}   <span class="dim">(the block's midpoint, (min + max) / 2, as fp16)</span>\n`;
          eq += `zero-bit block: nothing is stored for this pixel\n\n<b>Dequantize</b>\nx̂  = <span class="m">m</span> = ${n5(xh)}  →  × 255 = ${(xh * 255).toFixed(2)} ≈ ${dec[i]}\n\n`;
        } else {
        eq += `<span class="m">m</span>  = ${n5(m)}   <span class="m">s</span> = ${n5(s)}   <span class="dim">(the fp16 values decoded above)</span>\n`;
        const u = s > 0 ? (xin - m) / s : 0;
        eq += `(${xv} − <span class="m">m</span>) / <span class="m">s</span> = (${n5(xin)} − ${n5(m)}) / ${n5(s)} = ${u.toFixed(3)}\n`;
        eq += `q  = clamp(round(${u.toFixed(3)}), 0, ${L}) = <span class="c">${q}</span>   stored as <span class="c">${q.toString(2).padStart(bw[i], "0")}</span>\n\n`;
        eq += `<b>Dequantize</b>\nx̂  = <span class="m">m</span> + <span class="c">q</span> × <span class="m">s</span> = ${n5(m)} + ${q} × ${n5(s)} = ${n5(xh)}  →  × 255 = ${(xh * 255).toFixed(2)} ≈ ${dec[i]}\n\n`;
        }
      }
      eq += `<b>Error</b>\nx̂ − x = ${n5(xh)} − ${n5(x)} = ${(xh - x).toFixed(5)}  →  ${((xh - x) * 255).toFixed(2)} on the 0–255 scale (grid shows ${sgn(err[i])})`;
      if (d.diffuse) {
        const e = xin - xh;
        eq += `\n\n<b>Passed on</b> <span class="dim">(amber-outlined cells in Stored)</span>\ne = x′ − x̂ = ${n5(e)}\nright +7/16·e = ${n5(e * 7 / 16)}   below-left +3/16·e = ${n5(e * 3 / 16)}\nbelow +5/16·e = ${n5(e * 5 / 16)}   below-right +1/16·e = ${n5(e / 16)}`;
      }
      html += `<div class="eqpanel"><span class="lbl">How this pixel was computed</span><div class="eq">${eq}</div></div>`;
    }
    html += `</div>`;
  }
  $("scope").innerHTML = html;
  drawBitMap();
}
$("scope").addEventListener("click", async e => {
  const cell = e.target.closest(".cell[data-mid]");
  if (cell) {
    const mid = cell.dataset.mid, i = +cell.dataset.i;
    state.sel = state.sel && state.sel.mid === mid && state.sel.i === i ? null : {mid, i};
    microscope(); return;
  }
  const b = e.target.closest(".copy"); if (!b) return;
  const text = b.dataset.tsv;
  try { await navigator.clipboard.writeText(text); b.textContent = "Copied"; setTimeout(() => b.textContent = "Copy", 1400); }
  catch {
    const fig = b.closest(".mat"); let ta = fig.querySelector("textarea");
    if (!ta) { ta = document.createElement("textarea"); ta.className = "fallback"; ta.readOnly = true; fig.appendChild(ta); }
    ta.value = text; ta.select(); b.textContent = "Select and copy below";
  }
});



loadURL(SAMPLE);
