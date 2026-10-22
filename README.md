# Optimizer Viz

An interactive 3D playground that shows how common machine-learning optimizers move across a loss surface. Eight optimizers start from the same point on the same terrain, and you can watch where each update rule takes them.

**Optimizers:** SGD, Momentum, Nesterov, AdaGrad, RMSProp, Adam, AdamW, Lion

**Surfaces:**

| Surface | What it shows |
| --- | --- |
| Narrow valley, ½(x² + 20y²) | Ill-conditioning: SGD zig-zags, momentum and adaptive methods don't |
| Rosenbrock | Following a curved valley floor to the minimum |
| Saddle point, x² − y² + ½y⁴ | Escaping a near-flat ridge; adaptive methods leave far sooner |
| Himmelblau | Four minima; small differences decide which basin each optimizer reaches |
| Bumpy bowl | Local minima; momentum and gradient noise help escape them |

## Features

- 3D relief view with orbit and zoom, plus a top-down contour view
- One dot per step, so step size is visible directly
- Per-optimizer learning rates, gradient-noise slider to simulate minibatch SGD
- Inspector showing each optimizer's update equations and live internal state (gradient, velocity or moments, effective learning rate per axis), with arrows comparing steepest descent to the actual step
- Log-scale loss curves
- Click the terrain to restart from any point

## Run it

Open `index.html` in a browser. It is a single self-contained file; three.js loads from cdnjs.

## Project layout

```
index.html          built page (open this)
src/optimizers.js   loss surfaces and optimizer update rules, no DOM; runs in Node too
src/app.js          three.js scene, controls, loss chart, inspector
src/shell.html      markup and styles
build.py            stitches src/ into index.html
```

After editing anything in `src/`, run `python3 build.py`.

To check the update rules without a browser:

```bash
node -e 'const {SURFACES,OPT_KEYS,makeRun,stepRun}=require("./src/optimizers.js");
const s=SURFACES.rosenbrock;
for (const k of OPT_KEYS){const r=makeRun(k,s,s.start,s.lr[k],0);for(let i=0;i<1000;i++)stepRun(r,s);console.log(k,r.loss.at(-1))}'
```
