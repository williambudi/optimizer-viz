const MAXSTEPS = { valley: 300, rosenbrock: 1000, saddle: 200, himmelblau: 300, bumpy: 500 };
const $ = (id) => document.getElementById(id);
const fieldC = $("field"), lossC = $("loss"), lctx = lossC.getContext("2d");
const W = 10, H = 3.4, LIFT = 0.05;

const S = {
  surfKey: "valley", start: null, runs: {}, lr: {}, enabled: Object.fromEntries(OPT_KEYS.map(k => [k, true])),
  sel: "adam", playing: false, step: 0, noise: 0, speed: 3, lo: 0, hi: 1, D: 6,
};
const surf = () => SURFACES[S.surfKey];
let T = {};
function readTheme() {
  const cs = getComputedStyle(document.documentElement), g = n => cs.getPropertyValue(n).trim();
  T = { ink: g("--ink"), muted: g("--muted"), line: g("--line"), panel: g("--panel"), bg: g("--bg"), low: g("--t-low"), mid: g("--t-mid"), high: g("--t-high"), tline: g("--t-line"), ui: g("--ui"), math: g("--math") };
  for (const k of OPT_KEYS) T[k] = g("--c-" + k);
}
function fmtNum(v) {
  if (!isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e4)) { const [m, e] = v.toExponential(1).split("e"); return `${m}×10${sup(e)}`; }
  return (a >= 100 ? v.toFixed(0) : a >= 1 ? v.toFixed(2) : v.toFixed(3)).replace("-", "−");
}
function sup(e) { const m = { "-": "⁻", "+": "", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" }; return [...e].map(c => m[c]).join(""); }

// ---------- data ↔ world ----------
const tr = (v) => surf().logField ? Math.log10(Math.max(v - surf().fmin, 0) + 0.05) : v - surf().fmin;
function hOf(x, y) { const t = (tr(surf().f(x, y)) - S.lo) / (S.hi - S.lo); return Math.min(Math.max(t, 0), 1.25) * H; }
function toWorld(x, y, lift = 0) {
  const s = surf();
  return new THREE.Vector3((x - (s.xr[0] + s.xr[1]) / 2) / (s.xr[1] - s.xr[0]) * W, hOf(x, y) + lift, -(y - (s.yr[0] + s.yr[1]) / 2) / (s.yr[1] - s.yr[0]) * S.D);
}
function toData(p) {
  const s = surf();
  return [(s.xr[0] + s.xr[1]) / 2 + p.x / W * (s.xr[1] - s.xr[0]), (s.yr[0] + s.yr[1]) / 2 - p.z / S.D * (s.yr[1] - s.yr[0])];
}

// ---------- three.js scene ----------
let renderer, scene, camera, terrain, skirt, gl = true;
const pathObjs = {}, marks = new THREE.Group(), arrows = { g: null, u: null };
const cam = { theta: -0.55, phi: 0.98, r: 17.5, tTheta: -0.55, tPhi: 0.98, tR: 17.5 };
let dirty = true;
const requestRender = () => { dirty = true; };

const terrainMat = new THREE.ShaderMaterial({
  uniforms: { c0: { value: new THREE.Color() }, c1: { value: new THREE.Color() }, c2: { value: new THREE.Color() }, lc: { value: new THREE.Color() }, la: { value: 0.28 }, bands: { value: 18 }, H: { value: H } },
  vertexShader: `varying float vH; varying vec3 vN; uniform float H;
    void main(){ vH = position.y / H; vN = normal; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`,
  fragmentShader: `uniform vec3 c0,c1,c2,lc; uniform float la, bands; varying float vH; varying vec3 vN;
    vec3 ramp(float t){ return t < .5 ? mix(c0,c1,t*2.) : mix(c1,c2,(t-.5)*2.); }
    void main(){
      float t = clamp(vH,0.,1.);
      vec3 col = ramp(pow(t,.8));
      vec3 L = normalize(vec3(.35,1.,.45));
      float d = max(dot(normalize(vN),L),0.);
      col *= .58 + .5*d;
      float x = vH*bands; float w = fwidth(x);
      float l = 1. - smoothstep(0., w*1.3, abs(fract(x-.5)-.5));
      col = mix(col, lc, l*la);
      gl_FragColor = vec4(col,1.);
    }`,
  extensions: { derivatives: true }, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 2,
});
const skirtMat = new THREE.MeshBasicMaterial({ side: THREE.DoubleSide });

let dotTex;
function makeDotTex() {
  const c = document.createElement("canvas"); c.width = c.height = 64;
  const x = c.getContext("2d"); x.fillStyle = "#fff"; x.beginPath(); x.arc(32, 32, 28, 0, 7); x.fill();
  dotTex = new THREE.CanvasTexture(c);
}

function initGL() {
  try {
    renderer = new THREE.WebGLRenderer({ canvas: fieldC, antialias: true, alpha: true });
  } catch (e) { gl = false; }
  if (!gl || !renderer.getContext()) { gl = false; const d = document.createElement("div"); d.className = "nogl"; d.textContent = "This browser could not start WebGL, so the 3D terrain can't be drawn. The loss chart below still works."; fieldC.parentNode.append(d); return; }
  renderer.setClearColor(0x000000, 0);
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(34, 1.6, 0.1, 200);
  makeDotTex();
  scene.add(marks);
  for (const k of OPT_KEYS) {
    const n = 1001, pos = new Float32Array(n * 3);
    const geo = new THREE.BufferGeometry(); geo.setAttribute("position", new THREE.BufferAttribute(pos, 3)); geo.setDrawRange(0, 0);
    const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ transparent: true, opacity: 0.75 }));
    const pts = new THREE.Points(geo, new THREE.PointsMaterial({ size: 5, sizeAttenuation: false, map: dotTex, alphaTest: 0.5, transparent: true }));
    const head = new THREE.Mesh(new THREE.SphereGeometry(0.13, 20, 14), new THREE.MeshBasicMaterial());
    const ring = new THREE.Mesh(new THREE.SphereGeometry(0.19, 20, 14), new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.35 }));
    const grp = new THREE.Group(); grp.add(line, pts, head, ring); scene.add(grp);
    pathObjs[k] = { grp, geo, line, pts, head, ring };
  }
  arrows.g = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 1, 0x000000, 0.25, 0.14);
  arrows.u = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 1, 0x000000, 0.3, 0.17);
  scene.add(arrows.g, arrows.u);
  bindOrbit();
  requestAnimationFrame(loop);
}

function buildTerrain() {
  if (!gl) { computeRange(); return; }
  const s = surf();
  S.D = W * (s.yr[1] - s.yr[0]) / (s.xr[1] - s.xr[0]);
  computeRange();
  const nx = 180, nz = Math.round(nx * S.D / W);
  const pos = new Float32Array((nx + 1) * (nz + 1) * 3), idx = [];
  for (let j = 0; j <= nz; j++) for (let i = 0; i <= nx; i++) {
    const x = s.xr[0] + i / nx * (s.xr[1] - s.xr[0]), y = s.yr[1] - j / nz * (s.yr[1] - s.yr[0]);
    const p = toWorld(x, y), o = (j * (nx + 1) + i) * 3; pos[o] = p.x; pos[o + 1] = p.y; pos[o + 2] = p.z;
  }
  for (let j = 0; j < nz; j++) for (let i = 0; i < nx; i++) {
    const a = j * (nx + 1) + i, b = a + 1, c = a + nx + 1, d = c + 1; idx.push(a, c, b, b, c, d);
  }
  const geo = new THREE.BufferGeometry(); geo.setAttribute("position", new THREE.BufferAttribute(pos, 3)); geo.setIndex(idx); geo.computeVertexNormals();
  if (terrain) { scene.remove(terrain); terrain.geometry.dispose(); }
  terrain = new THREE.Mesh(geo, terrainMat); scene.add(terrain);
  // plinth walls: turn the surface into a physical relief block
  const base = -0.35, sp = [];
  const edge = (pts) => { for (let i = 0; i < pts.length - 1; i++) { const [a, b] = [pts[i], pts[i + 1]]; sp.push(a.x, a.y, a.z, b.x, b.y, b.z, b.x, base, b.z, a.x, a.y, a.z, b.x, base, b.z, a.x, base, a.z); } };
  const row = (j) => Array.from({ length: nx + 1 }, (_, i) => new THREE.Vector3(pos[(j * (nx + 1) + i) * 3], pos[(j * (nx + 1) + i) * 3 + 1], pos[(j * (nx + 1) + i) * 3 + 2]));
  const col = (i) => Array.from({ length: nz + 1 }, (_, j) => new THREE.Vector3(pos[(j * (nx + 1) + i) * 3], pos[(j * (nx + 1) + i) * 3 + 1], pos[(j * (nx + 1) + i) * 3 + 2]));
  edge(row(0)); edge(row(nz)); edge(col(0)); edge(col(nx));
  const sg = new THREE.BufferGeometry(); sg.setAttribute("position", new THREE.BufferAttribute(new Float32Array(sp), 3));
  if (skirt) { scene.remove(skirt); skirt.geometry.dispose(); }
  skirt = new THREE.Mesh(sg, skirtMat); scene.add(skirt);
  // minima markers
  marks.clear();
  for (const [mx, my] of s.minima) {
    const p = toWorld(mx, my, 0.03);
    const ring = new THREE.Mesh(new THREE.TorusGeometry(0.2, 0.028, 8, 36), new THREE.MeshBasicMaterial({ color: T.tline }));
    ring.rotation.x = Math.PI / 2; ring.position.copy(p); marks.add(ring);
    const stem = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, 0.7, 6), ring.material);
    stem.position.copy(p).add(new THREE.Vector3(0, 0.35, 0)); marks.add(stem);
  }
  const startM = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.16, 0.05, 28), new THREE.MeshBasicMaterial({ color: T.tline }));
  startM.name = "start"; marks.add(startM);
  applyThemeGL(); cam.tR = 17.5 * Math.max(1, S.D / 6.5);
}
function computeRange() {
  const s = surf(); let lo = Infinity, hi = -Infinity;
  for (let j = 0; j <= 80; j++) for (let i = 0; i <= 80; i++) {
    const v = tr(s.f(s.xr[0] + i / 80 * (s.xr[1] - s.xr[0]), s.yr[0] + j / 80 * (s.yr[1] - s.yr[0])));
    if (v < lo) lo = v; if (v > hi) hi = v;
  }
  S.lo = lo; S.hi = hi;
}
function applyThemeGL() {
  if (!gl) return;
  const u = terrainMat.uniforms;
  u.c0.value.set(T.low); u.c1.value.set(T.mid); u.c2.value.set(T.high); u.lc.value.set(T.tline);
  const dark = u.c2.value.getHSL({}).l < 0.4; u.la.value = dark ? 0.18 : 0.22;
  skirtMat.color.set(T.mid).lerp(new THREE.Color(T.bg), 0.35);
  marks.children.forEach(m => m.material.color.set(T.tline));
  for (const k of OPT_KEYS) { const o = pathObjs[k], c = new THREE.Color(T[k]); o.line.material.color.copy(c); o.pts.material.color.copy(c); o.head.material.color.copy(c); o.ring.material.color.copy(c); }
  arrows.g.setColor(new THREE.Color(T.tline));
  requestRender();
}

function syncPaths() {
  if (!gl) return;
  const st = marks.getObjectByName("start"); if (st) st.position.copy(toWorld(S.start[0], S.start[1], 0.03));
  for (const k of OPT_KEYS) {
    const r = S.runs[k], o = pathObjs[k], arr = o.geo.attributes.position.array, n = Math.min(r.path.length, 1001);
    for (let i = 0; i < n; i++) { const p = toWorld(r.path[i][0], r.path[i][1], LIFT); arr[i * 3] = p.x; arr[i * 3 + 1] = p.y; arr[i * 3 + 2] = p.z; }
    o.geo.attributes.position.needsUpdate = true; o.geo.setDrawRange(0, n); o.geo.computeBoundingSphere();
    const hp = toWorld(r.th[0], r.th[1], 0.12); o.head.position.copy(hp); o.ring.position.copy(hp);
    o.grp.visible = S.enabled[k];
    const sel = k === S.sel; o.ring.visible = sel; o.pts.material.size = sel ? 6 : 4.5; o.line.material.opacity = sel ? 0.95 : 0.6;
  }
  // inspector arrows
  const r = S.runs[S.sel], s = surf();
  const show = S.enabled[S.sel] && r.info && !r.dead;
  arrows.g.visible = arrows.u.visible = !!show;
  if (show) {
    const sx = W / (s.xr[1] - s.xr[0]), sz = S.D / (s.yr[1] - s.yr[0]);
    const origin = toWorld(r.th[0], r.th[1], 0.14);
    const dir3 = (dx, dy) => {
      const wx = dx * sx, wz = -dy * sz, n = Math.hypot(wx, wz); if (!n) return null;
      const e = 0.12, ddx = e * wx / n / sx, ddy = -e * wz / n / sz;
      const dh = hOf(r.th[0] + ddx, r.th[1] + ddy) - hOf(r.th[0], r.th[1]);
      return new THREE.Vector3(e * wx / n, dh, e * wz / n).normalize();
    };
    const dg = dir3(-r.info.g[0], -r.info.g[1]), du = dir3(-r.info.u[0], -r.info.u[1]);
    if (dg) { arrows.g.position.copy(origin); arrows.g.setDirection(dg); arrows.g.setLength(1.3, 0.25, 0.14); } else arrows.g.visible = false;
    if (du) { arrows.u.position.copy(origin); arrows.u.setDirection(du); arrows.u.setLength(1.3, 0.3, 0.17); arrows.u.setColor(new THREE.Color(T[S.sel])); } else arrows.u.visible = false;
  }
  requestRender();
}

// ---------- camera & interaction ----------
function placeCamera() {
  const { theta, phi, r } = cam, tgt = new THREE.Vector3(0, H * 0.22, 0);
  camera.position.set(tgt.x + r * Math.sin(phi) * Math.sin(theta), tgt.y + r * Math.cos(phi), tgt.z + r * Math.sin(phi) * Math.cos(theta));
  if (phi > 0.03) camera.up.set(0, 1, 0); else camera.up.set(-Math.sin(theta), 0, -Math.cos(theta));
  camera.lookAt(tgt);
}
function resizeGL() {
  if (!gl) return;
  const b = fieldC.getBoundingClientRect();
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(b.width, b.height, false);
  camera.aspect = b.width / b.height; camera.updateProjectionMatrix(); requestRender();
}
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
function loop() {
  const k = reduceMotion ? 1 : 0.16;
  for (const [a, t] of [["theta", "tTheta"], ["phi", "tPhi"], ["r", "tR"]]) {
    const d = cam[t] - cam[a]; if (Math.abs(d) > 1e-4) { cam[a] += d * k; dirty = true; } else cam[a] = cam[t];
  }
  if (dirty) { placeCamera(); renderer.render(scene, camera); dirty = false; }
  requestAnimationFrame(loop);
}
const ray = new THREE.Raycaster();
function pick(e) {
  const b = fieldC.getBoundingClientRect();
  ray.setFromCamera(new THREE.Vector2((e.clientX - b.left) / b.width * 2 - 1, -((e.clientY - b.top) / b.height) * 2 + 1), camera);
  const hit = terrain && ray.intersectObject(terrain)[0];
  return hit ? toData(hit.point) : null;
}
function setView(top) {
  $("v3d").setAttribute("aria-pressed", !top); $("vtop").setAttribute("aria-pressed", top);
  if (top) { cam.tTheta = 0; cam.tPhi = 0.0001; } else { cam.tTheta = -0.55; cam.tPhi = 0.98; }
  if (cam.tTheta - cam.theta > Math.PI) cam.theta += 2 * Math.PI; else if (cam.theta - cam.tTheta > Math.PI) cam.theta -= 2 * Math.PI;
  requestRender();
}
function bindOrbit() {
  let down = null, moved = false, hoverQ = null;
  fieldC.addEventListener("pointerdown", e => { down = { x: e.clientX, y: e.clientY, th: cam.tTheta, ph: cam.tPhi }; moved = false; fieldC.setPointerCapture(e.pointerId); });
  fieldC.addEventListener("pointermove", e => {
    if (down) {
      const dx = e.clientX - down.x, dy = e.clientY - down.y;
      if (Math.hypot(dx, dy) > 4) { moved = true; fieldC.classList.add("dragging"); }
      if (moved) {
        cam.tTheta = cam.theta = down.th - dx * 0.008;
        cam.tPhi = cam.phi = Math.min(1.45, Math.max(0.0001, down.ph - dy * 0.006));
        $("v3d").setAttribute("aria-pressed", cam.tPhi > 0.05); $("vtop").setAttribute("aria-pressed", cam.tPhi <= 0.05);
        requestRender();
      }
      return;
    }
    hoverQ = e; requestAnimationFrame(() => { if (!hoverQ) return; const d = pick(hoverQ); hoverQ = null; showReadout(d); });
  });
  fieldC.addEventListener("pointerup", e => {
    fieldC.classList.remove("dragging");
    if (down && !moved) { const d = pick(e); if (d) { S.start = d; resetRuns(); setPlaying(true); } }
    down = null;
  });
  fieldC.addEventListener("pointerleave", () => showReadout(null));
  fieldC.addEventListener("wheel", e => { e.preventDefault(); cam.tR = Math.min(40, Math.max(7, cam.tR * Math.exp(e.deltaY * 0.0012))); requestRender(); }, { passive: false });
  $("v3d").onclick = () => setView(false);
  $("vtop").onclick = () => setView(true);
}
function showReadout(d) {
  if (!d) { $("readout").innerHTML = "<i>Hover the terrain to read the loss</i>"; return; }
  $("readout").innerHTML = `<i>x</i> = ${fmtNum(d[0])} &nbsp; <i>y</i> = ${fmtNum(d[1])} &nbsp; <i>f</i> = ${fmtNum(surf().f(d[0], d[1]))}`;
}

// ---------- tabs, roster, runs ----------
function buildTabs() {
  for (const [k, s] of Object.entries(SURFACES)) {
    const b = document.createElement("button");
    b.className = "tab"; b.setAttribute("role", "tab"); b.id = "tab-" + k; b.setAttribute("aria-selected", k === S.surfKey);
    b.innerHTML = `<b>${s.name}</b><span>${s.tag}</span>`; b.onclick = () => setSurface(k);
    $("tabs").append(b);
  }
}
function buildRoster() {
  const r = $("roster"); r.innerHTML = "";
  for (const k of OPT_KEYS) {
    const o = OPTIMIZERS[k], row = document.createElement("div");
    row.className = "row"; row.dataset.k = k; row.style.setProperty("--sw", `var(--c-${k})`);
    row.innerHTML = `<input type="checkbox" id="on-${k}" aria-label="Show ${o.name}" checked>
      <span class="nm">${o.name}</span>
      <input class="lr" id="lr-${k}" type="number" step="any" min="0" value="${S.lr[k]}" aria-label="${o.name} learning rate">
      <span class="ls" id="ls-${k}">—</span>`;
    row.onclick = (e) => { if (e.target.tagName === "INPUT") return; S.sel = k; markSel(); renderInspector(); syncPaths(); drawLoss(); };
    row.querySelector("input[type=checkbox]").onchange = (e) => { S.enabled[k] = e.target.checked; row.classList.toggle("off", !e.target.checked); syncPaths(); drawLoss(); };
    row.querySelector(".lr").onchange = (e) => { const v = parseFloat(e.target.value); if (v > 0) { S.lr[k] = v; resetRuns(); } else e.target.value = S.lr[k]; };
    r.append(row);
  }
  markSel();
}
function markSel() { document.querySelectorAll(".row").forEach(r => r.classList.toggle("sel", r.dataset.k === S.sel)); }

function resetRuns(keepPlay) {
  const s = surf();
  for (const k of OPT_KEYS) S.runs[k] = makeRun(k, s, S.start, S.lr[k], S.noise);
  S.step = 0; if (!keepPlay) setPlaying(false);
  updateStatus(); renderInspector(); syncPaths(); drawLoss();
}
function setSurface(k) {
  S.surfKey = k; const s = surf();
  S.start = s.start.slice(); S.lr = { ...s.lr };
  document.querySelectorAll(".tab").forEach(t => t.setAttribute("aria-selected", t.id === "tab-" + k));
  $("formula").textContent = s.formula;
  $("note").innerHTML = `<strong>What to watch.</strong> ${s.note}`;
  for (const o of OPT_KEYS) $("lr-" + o).value = s.lr[o];
  buildTerrain(); resetRuns();
}
function advance(n) {
  const s = surf(), max = MAXSTEPS[S.surfKey];
  for (let i = 0; i < n && S.step < max; i++) { for (const k of OPT_KEYS) stepRun(S.runs[k], s); S.step++; }
  if (S.step >= max) setPlaying(false);
  updateStatus();
}
function updateStatus() {
  $("stepcount").textContent = `Step ${S.step} of ${MAXSTEPS[S.surfKey]}`;
  const s = surf();
  for (const k of OPT_KEYS) {
    const r = S.runs[k], el = $("ls-" + k);
    el.classList.toggle("dead", r.dead);
    el.textContent = r.dead ? "diverged" : fmtNum(r.loss.at(-1) - s.fmin);
  }
}
function setPlaying(p) { S.playing = p; $("play").textContent = p ? "Pause" : (S.step >= MAXSTEPS[S.surfKey] ? "Replay" : "Play"); if (p) requestAnimationFrame(tick); }
let acc = 0;
function tick() {
  if (!S.playing) return;
  acc += MAXSTEPS[S.surfKey] / (60 * (22 / S.speed));
  const n = Math.floor(acc); acc -= n;
  if (n > 0) { advance(n); renderInspector(); syncPaths(); drawLoss(); }
  if (S.playing) requestAnimationFrame(tick);
}

// ---------- loss chart ----------
function drawLoss() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2), b = lossC.getBoundingClientRect();
  const w = Math.round(b.width * dpr), h = Math.round(b.height * dpr);
  if (lossC.width !== w || lossC.height !== h) { lossC.width = w; lossC.height = h; }
  const ctx = lctx, s = surf(), max = MAXSTEPS[S.surfKey];
  ctx.clearRect(0, 0, w, h);
  const padL = 50 * dpr, padR = 10 * dpr, padT = 8 * dpr, padB = 24 * dpr;
  const lg = v => Math.log10(Math.max(v - s.fmin, 1e-12));
  let lo = Infinity, hi = -Infinity;
  for (const k of OPT_KEYS) for (const v of S.runs[k].loss) { const t = lg(v); if (t < lo) lo = t; if (t > hi) hi = t; }
  lo = Math.max(Math.floor(lo), -12); hi = Math.ceil(hi); if (hi - lo < 2) lo = hi - 2;
  const X = i => padL + i / max * (w - padL - padR), Y = t => padT + (hi - t) / (hi - lo) * (h - padT - padB);
  ctx.font = `${14 * dpr}px ${T.math}`; ctx.fillStyle = T.muted; ctx.strokeStyle = T.line; ctx.lineWidth = dpr;
  const stepD = Math.ceil((hi - lo) / 5);
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (let d = hi; d >= lo; d -= stepD) { const y = Y(d); ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke(); ctx.fillText("10" + sup(String(d)), padL - 8 * dpr, y); }
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (let i = 0; i <= 4; i++) { const st = Math.round(max * i / 4); ctx.fillText(st, X(st), h - padB + 6 * dpr); }
  const order = OPT_KEYS.filter(k => S.enabled[k] && k !== S.sel); if (S.enabled[S.sel]) order.push(S.sel);
  for (const k of order) {
    const L = S.runs[k].loss; if (L.length < 2) continue;
    ctx.beginPath(); L.forEach((v, i) => { const y = Y(Math.max(lg(v), lo)); i ? ctx.lineTo(X(i), y) : ctx.moveTo(X(i), y); });
    ctx.strokeStyle = T[k]; ctx.lineWidth = (k === S.sel ? 2.6 : 1.4) * dpr; ctx.globalAlpha = k === S.sel ? 1 : 0.7; ctx.stroke(); ctx.globalAlpha = 1;
  }
}

// ---------- inspector ----------
function renderInspector() {
  const k = S.sel, o = OPTIMIZERS[k], r = S.runs[k], i = r.info, init = o.init();
  const hpTxt = Object.entries(r.hp).map(([n, v]) => `${({ beta: "β", rho: "ρ", b1: "β₁", b2: "β₂", wd: "λ" })[n]} = ${v}`).join(",  ");
  const v2 = (a) => a ? `<td>${fmtNum(a[0])}</td><td>${fmtNum(a[1])}</td>` : `<td>—</td><td>—</td>`;
  let rows = `<tr><td>Gradient g</td>${v2(i && i.g)}</tr><tr><td>Step taken</td>${v2(i && i.u.map(x => -x))}</tr><tr><td>Effective learning rate</td>${v2(i && i.lrEff)}</tr>`;
  if (init.v || init.m) rows += `<tr><td>${k.startsWith("adam") || k === "lion" ? "First moment m" : "Velocity v"}</td>${v2(i && i.vel)}</tr>`;
  if (init.G || init.s || (init.v && k.startsWith("adam"))) rows += `<tr><td>${k === "adagrad" ? "Sum of g², G" : k === "rmsprop" ? "Mean of g², s" : "Second moment v"}</td>${v2(i && i.sec)}</tr>`;
  $("insp").style.setProperty("--sw", `var(--c-${k})`);
  $("insp").innerHTML = `
    <h3><i></i>${o.name}</h3>
    <p>${o.blurb}</p>
    <div class="eq">${o.eq.join("\n")}</div>
    <div class="hps">η = ${S.lr[k]}${hpTxt ? ",  " + hpTxt : ""}</div>
    <table class="state"><thead><tr><th>At step ${r.path.length - 1}</th><th>x</th><th>y</th></tr></thead><tbody>${rows}</tbody></table>
    <div class="legendkey">
      <span><svg width="28" height="10" aria-hidden="true"><line x1="1" y1="5" x2="27" y2="5" stroke="currentColor" stroke-width="2"/></svg>Steepest descent, the direction of −g</span>
      <span><svg width="28" height="10" aria-hidden="true"><line x1="1" y1="5" x2="27" y2="5" stroke="var(--c-${k})" stroke-width="3"/></svg>Direction ${o.name} actually stepped</span>
    </div>`;
}

// ---------- boot ----------
$("play").onclick = () => { if (S.step >= MAXSTEPS[S.surfKey]) resetRuns(); setPlaying(!S.playing); };
$("step").onclick = () => { setPlaying(false); advance(1); renderInspector(); syncPaths(); drawLoss(); };
$("reset").onclick = () => resetRuns();
$("speed").oninput = e => { S.speed = +e.target.value; $("speedOut").textContent = S.speed + "×"; };
$("noise").oninput = e => { S.noise = +e.target.value / 100; $("noiseOut").textContent = "σ = " + S.noise.toFixed(2); };
$("noise").onchange = () => resetRuns(S.playing);
const onTheme = () => { readTheme(); applyThemeGL(); drawLoss(); };
window.addEventListener("resize", () => { resizeGL(); drawLoss(); });
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", onTheme);
new MutationObserver(onTheme).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

$("speedOut").textContent = "3×"; $("noiseOut").textContent = "σ = 0.00";
readTheme(); initGL(); resizeGL(); buildTabs();
S.lr = { ...surf().lr }; S.start = surf().start.slice();
buildRoster(); setSurface(S.surfKey); showReadout(null);
if (document.fonts) document.fonts.ready.then(() => { drawLoss(); });
