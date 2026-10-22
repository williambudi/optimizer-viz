// ---- Loss surfaces ----
const SURFACES = {
  valley: {
    name: "Narrow valley",
    tag: "Ill-conditioned quadratic",
    formula: "f(x, y) = ½(x² + 20y²)",
    note: "Curvature is 20× steeper in y than in x. Plain SGD has to keep its step small enough for the steep direction, so it zig-zags across the valley and crawls along it. Momentum averages the zig-zags away; the adaptive methods rescale each axis.",
    f: (x, y) => 0.5 * (x * x + 20 * y * y),
    g: (x, y) => [x, 20 * y],
    xr: [-5, 5], yr: [-3, 3], start: [-4.6, 2.5], fmin: 0, minima: [[0, 0]], gscale: 10,
    lr: { sgd: 0.09, momentum: 0.03, nesterov: 0.03, adagrad: 1.2, rmsprop: 0.08, adam: 0.15, adamw: 0.15, lion: 0.04 },
  },
  rosenbrock: {
    name: "Rosenbrock",
    tag: "Curved banana valley",
    formula: "f(x, y) = (1 − x)² + 100(y − x²)²",
    note: "Finding the valley floor is easy; following its curve to the minimum at (1, 1) is hard because the gradient points mostly across the valley. Watch which optimizers keep moving along the floor and which stall.",
    f: (x, y) => (1 - x) ** 2 + 100 * (y - x * x) ** 2,
    g: (x, y) => [-2 * (1 - x) - 400 * x * (y - x * x), 200 * (y - x * x)],
    xr: [-2, 2], yr: [-1, 3], start: [-1.6, 2.6], fmin: 0, minima: [[1, 1]], gscale: 60, logField: true,
    lr: { sgd: 0.0006, momentum: 0.00025, nesterov: 0.00025, adagrad: 0.4, rmsprop: 0.008, adam: 0.04, adamw: 0.04, lion: 0.006 },
  },
  saddle: {
    name: "Saddle point",
    tag: "Escape from a flat ridge",
    formula: "f(x, y) = x² − y² + ½y⁴",
    note: "The run starts almost exactly on the ridge, where the y-gradient is close to zero. SGD and momentum need many steps to notice the downhill direction. Adam, RMSProp and Lion divide by the gradient's own size, so a tiny but consistent slope gets a full-size step.",
    f: (x, y) => x * x - y * y + 0.5 * y ** 4,
    g: (x, y) => [2 * x, -2 * y + 2 * y ** 3],
    xr: [-2.5, 2.5], yr: [-1.8, 1.8], start: [2.2, 0.002], fmin: -0.5, minima: [[0, 1], [0, -1]], gscale: 2,
    lr: { sgd: 0.05, momentum: 0.015, nesterov: 0.015, adagrad: 0.5, rmsprop: 0.03, adam: 0.06, adamw: 0.06, lion: 0.02 },
  },
  himmelblau: {
    name: "Himmelblau",
    tag: "Four minima, one start",
    formula: "f(x, y) = (x² + y − 11)² + (x + y² − 7)²",
    note: "Four equally good minima surround a local maximum near (−0.27, −0.92). The run starts just beside it, so each optimizer's path decides which basin it falls into. Small changes in learning rate or noise send them to different answers.",
    f: (x, y) => (x * x + y - 11) ** 2 + (x + y * y - 7) ** 2,
    g: (x, y) => [4 * x * (x * x + y - 11) + 2 * (x + y * y - 7), 2 * (x * x + y - 11) + 4 * y * (x + y * y - 7)],
    xr: [-5.5, 5.5], yr: [-5, 5], start: [0.2, -1], fmin: 0,
    minima: [[3, 2], [-2.805118, 3.131312], [-3.77931, -3.283186], [3.584428, -1.848126]], gscale: 40, logField: true,
    lr: { sgd: 0.008, momentum: 0.002, nesterov: 0.002, adagrad: 0.8, rmsprop: 0.05, adam: 0.12, adamw: 0.12, lion: 0.03 },
  },
  bumpy: {
    name: "Bumpy bowl",
    tag: "Local minima everywhere",
    formula: "f(x, y) = 0.1(x² + y²) + ½(1 − cos 2x · cos 2y)",
    note: "A wide bowl covered in ripples. Methods without memory settle into the first dip they meet. Momentum carries speed over the small bumps, and gradient noise can shake a stuck optimizer loose. Try raising the noise slider.",
    f: (x, y) => 0.1 * (x * x + y * y) + 0.5 * (1 - Math.cos(2 * x) * Math.cos(2 * y)),
    g: (x, y) => [0.2 * x + Math.sin(2 * x) * Math.cos(2 * y), 0.2 * y + Math.cos(2 * x) * Math.sin(2 * y)],
    xr: [-6.5, 6.5], yr: [-5, 5], start: [5.6, 4.4], fmin: 0, minima: [[0, 0]], gscale: 1,
    lr: { sgd: 0.15, momentum: 0.04, nesterov: 0.04, adagrad: 0.8, rmsprop: 0.06, adam: 0.2, adamw: 0.2, lion: 0.06 },
  },
};

// ---- Optimizers ----
// Each step(state, theta, g, lr, hp) mutates theta and returns info for the inspector.
const OPTIMIZERS = {
  sgd: {
    name: "SGD", hp: {},
    eq: ["θ ← θ − η·g"],
    blurb: "Steps straight down the local gradient. Step size scales with gradient size, so steep directions overshoot and flat ones crawl.",
    init: () => ({}),
    step(s, th, g, lr) { const u = [lr * g[0], lr * g[1]]; return { u, lrEff: [lr, lr] }; },
  },
  momentum: {
    name: "Momentum", hp: { beta: 0.9 },
    eq: ["v ← β·v + g", "θ ← θ − η·v"],
    blurb: "Keeps a running velocity. Consistent gradients add up to speed; gradients that flip sign cancel out, which damps zig-zagging.",
    init: () => ({ v: [0, 0] }),
    step(s, th, g, lr, hp) {
      for (let i = 0; i < 2; i++) s.v[i] = hp.beta * s.v[i] + g[i];
      return { u: [lr * s.v[0], lr * s.v[1]], lrEff: [lr, lr], vel: s.v.slice() };
    },
  },
  nesterov: {
    name: "Nesterov", hp: { beta: 0.9 },
    eq: ["v ← β·v + g", "θ ← θ − η·(g + β·v)"],
    blurb: "Momentum that looks ahead: the step uses where the velocity is about to carry it, so it brakes earlier before overshooting.",
    init: () => ({ v: [0, 0] }),
    step(s, th, g, lr, hp) {
      for (let i = 0; i < 2; i++) s.v[i] = hp.beta * s.v[i] + g[i];
      return { u: [lr * (g[0] + hp.beta * s.v[0]), lr * (g[1] + hp.beta * s.v[1])], lrEff: [lr, lr], vel: s.v.slice() };
    },
  },
  adagrad: {
    name: "AdaGrad", hp: {},
    eq: ["G ← G + g²", "θ ← θ − η·g / (√G + ε)"],
    blurb: "Gives each parameter its own learning rate, shrinking it by the total squared gradient seen so far. Because G only grows, steps eventually become tiny.",
    init: () => ({ G: [0, 0] }),
    step(s, th, g, lr) {
      const u = [0, 0], le = [0, 0];
      for (let i = 0; i < 2; i++) { s.G[i] += g[i] * g[i]; le[i] = lr / (Math.sqrt(s.G[i]) + 1e-8); u[i] = le[i] * g[i]; }
      return { u, lrEff: le, sec: s.G.slice() };
    },
  },
  rmsprop: {
    name: "RMSProp", hp: { rho: 0.9 },
    eq: ["s ← ρ·s + (1−ρ)·g²", "θ ← θ − η·g / (√s + ε)"],
    blurb: "AdaGrad with a leaky memory: the squared-gradient average forgets old values, so the per-parameter learning rate can recover.",
    init: () => ({ s: [0, 0] }),
    step(st, th, g, lr, hp) {
      const u = [0, 0], le = [0, 0];
      for (let i = 0; i < 2; i++) { st.s[i] = hp.rho * st.s[i] + (1 - hp.rho) * g[i] * g[i]; le[i] = lr / (Math.sqrt(st.s[i]) + 1e-8); u[i] = le[i] * g[i]; }
      return { u, lrEff: le, sec: st.s.slice() };
    },
  },
  adam: {
    name: "Adam", hp: { b1: 0.9, b2: 0.999 },
    eq: ["m ← β₁·m + (1−β₁)·g", "v ← β₂·v + (1−β₂)·g²", "m̂ = m/(1−β₁ᵗ),  v̂ = v/(1−β₂ᵗ)", "θ ← θ − η·m̂ / (√v̂ + ε)"],
    blurb: "Momentum on top, RMSProp-style scaling underneath, with bias correction for the first steps. Each step is roughly η in size whatever the raw gradient scale.",
    init: () => ({ m: [0, 0], v: [0, 0], t: 0 }),
    step(s, th, g, lr, hp) {
      s.t++; const u = [0, 0], le = [0, 0];
      for (let i = 0; i < 2; i++) {
        s.m[i] = hp.b1 * s.m[i] + (1 - hp.b1) * g[i];
        s.v[i] = hp.b2 * s.v[i] + (1 - hp.b2) * g[i] * g[i];
        const mh = s.m[i] / (1 - hp.b1 ** s.t), vh = s.v[i] / (1 - hp.b2 ** s.t);
        le[i] = lr / (Math.sqrt(vh) + 1e-8); u[i] = le[i] * mh;
      }
      return { u, lrEff: le, vel: s.m.slice(), sec: s.v.slice() };
    },
  },
  adamw: {
    name: "AdamW", hp: { b1: 0.9, b2: 0.999, wd: 0.05 },
    eq: ["(m, v, m̂, v̂ as in Adam)", "θ ← θ − η·( m̂ / (√v̂ + ε) + λ·θ )"],
    blurb: "Adam with weight decay applied directly to the weights rather than mixed into the gradient. Here λ pulls every path gently toward the origin.",
    init: () => ({ m: [0, 0], v: [0, 0], t: 0 }),
    step(s, th, g, lr, hp) {
      s.t++; const u = [0, 0], le = [0, 0];
      for (let i = 0; i < 2; i++) {
        s.m[i] = hp.b1 * s.m[i] + (1 - hp.b1) * g[i];
        s.v[i] = hp.b2 * s.v[i] + (1 - hp.b2) * g[i] * g[i];
        const mh = s.m[i] / (1 - hp.b1 ** s.t), vh = s.v[i] / (1 - hp.b2 ** s.t);
        le[i] = lr / (Math.sqrt(vh) + 1e-8); u[i] = le[i] * mh + lr * hp.wd * th[i];
      }
      return { u, lrEff: le, vel: s.m.slice(), sec: s.v.slice() };
    },
  },
  lion: {
    name: "Lion", hp: { b1: 0.9, b2: 0.99 },
    eq: ["c = β₁·m + (1−β₁)·g", "θ ← θ − η·sign(c)", "m ← β₂·m + (1−β₂)·g"],
    blurb: "Keeps only a momentum buffer and steps by its sign, so every coordinate moves exactly η each step. Cheap on memory; needs a smaller learning rate than Adam.",
    init: () => ({ m: [0, 0] }),
    step(s, th, g, lr, hp) {
      const u = [0, 0];
      for (let i = 0; i < 2; i++) {
        const c = hp.b1 * s.m[i] + (1 - hp.b1) * g[i];
        u[i] = lr * Math.sign(c);
        s.m[i] = hp.b2 * s.m[i] + (1 - hp.b2) * g[i];
      }
      return { u, lrEff: [lr, lr], vel: s.m.slice() };
    },
  },
};
const OPT_KEYS = Object.keys(OPTIMIZERS);

function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function makeGauss(seed) { const r = mulberry32(seed); return () => { let u = 0, v = 0; while (u === 0) u = r(); v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }; }

function makeRun(key, surf, start, lr, noise) {
  const o = OPTIMIZERS[key];
  return { key, th: start.slice(), st: o.init(), hp: { ...o.hp }, lr, path: [start.slice()], loss: [surf.f(start[0], start[1])], dead: false, info: null, gnoise: makeGauss(1234), noise };
}
function stepRun(run, surf) {
  if (run.dead) return;
  const o = OPTIMIZERS[run.key];
  const g = surf.g(run.th[0], run.th[1]);
  if (run.noise > 0) { g[0] += run.noise * surf.gscale * run.gnoise(); g[1] += run.noise * surf.gscale * run.gnoise(); }
  const info = o.step(run.st, run.th, g, run.lr, run.hp);
  run.th[0] -= info.u[0]; run.th[1] -= info.u[1];
  info.g = g; run.info = info;
  const L = surf.f(run.th[0], run.th[1]);
  if (!isFinite(L) || Math.abs(run.th[0]) > 1e3 || Math.abs(run.th[1]) > 1e3) { run.dead = true; return; }
  run.path.push(run.th.slice()); run.loss.push(L);
}
if (typeof module !== "undefined") module.exports = { SURFACES, OPTIMIZERS, OPT_KEYS, makeRun, stepRun };
