const $ = (id) => document.getElementById(id);
const state = { run: null, stageA: null, stageB: null, framesA: [], framesB: [], idx: 0 };

async function api(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(path);
  return r.json();
}

async function loadRuns() {
  const runs = await api("/api/runs");
  const sel = $("runSel");
  sel.innerHTML = runs.map(r => `<option value="${r.name}">${r.label}</option>`).join("");
  state.run = runs[0]?.name;
  sel.onchange = () => { state.run = sel.value; loadStages(); };
  if (state.run) loadStages();
}

async function loadStages() {
  const stages = await api(`/api/stages/${encodeURIComponent(state.run)}`);
  const opts = stages.map(s => `<option value="${s.frames_path}">${s.label || s.id}</option>`).join("");
  $("stageA").innerHTML = opts;
  $("stageB").innerHTML = opts;
  if (stages.length) {
    $("stageA").value = stages[0].frames_path;
    $("stageB").value = stages[stages.length - 1].frames_path;
  }
  $("stageA").onchange = refreshPane.bind(null, "A");
  $("stageB").onchange = refreshPane.bind(null, "B");
  await Promise.all([refreshPane("A"), refreshPane("B")]);
  renderFrame();
}

async function refreshPane(which) {
  const sel = $("stage" + which);
  const stagePath = sel.value;
  const label = sel.options[sel.selectedIndex]?.textContent || stagePath;
  const frames = await api(`/api/frames/${encodeURIComponent(state.run)}/${stagePath}`);
  // Each entry is now {name, real} — backward-compat with plain strings if server is old.
  state["frames" + which] = frames.map(f => typeof f === "string" ? { name: f, real: true } : f);
  state["stage" + which] = stagePath;
  $("title" + which).textContent = label;
  updateValidRange();
}

// Clamp the scrub range to the intersection of "real" (non-symlink) frames
// across both panes. Lets a deflicker variant that only processed 116 frames
// show just those 116 positions in the viewer even though each variant dir
// contains 7,821 symlinked frames for positional alignment.
function updateValidRange() {
  const A = state.framesA, B = state.framesB;
  if (!A?.length || !B?.length) return;
  const n = Math.min(A.length, B.length);
  let minValid = -1, maxValid = -1;
  for (let i = 0; i < n; i++) {
    if (A[i]?.real && B[i]?.real) {
      if (minValid === -1) minValid = i;
      maxValid = i;
    }
  }
  if (minValid === -1) { minValid = 0; maxValid = n - 1; }
  state.rangeMin = minValid;
  state.rangeMax = maxValid;
  $("scrub").min = minValid;
  $("scrub").max = maxValid;
  if (state.idx < minValid || state.idx > maxValid) state.idx = minValid;
  const count = maxValid - minValid + 1;
  $("frameTotal").textContent = count === n ? count : `${count} (frames ${minValid}–${maxValid})`;
  renderFrame();
}

function renderFrame() {
  const i = state.idx;
  const fa = state.framesA[i];
  const fb = state.framesB[i];
  if (fa) $("imgA").src = `/runs/${state.run}/${state.stageA}/${fa.name}`;
  if (fb) $("imgB").src = `/runs/${state.run}/${state.stageB}/${fb.name}`;
  $("frameIdx").textContent = i;
  $("scrub").value = i;
}

$("scrub").oninput = (e) => { state.idx = parseInt(e.target.value); renderFrame(); };

let playTimer = null;
function isPlaying() { return playTimer !== null; }
function stop() {
  if (playTimer) { clearInterval(playTimer); playTimer = null; }
  $("playBtn").textContent = "▶ Play";
  $("playBtn").classList.remove("active");
}
function play() {
  if (isPlaying()) return;
  const fps = parseInt($("fpsSel").value) || 24;
  const interval = 1000 / fps;
  $("playBtn").textContent = "⏸ Pause";
  $("playBtn").classList.add("active");
  playTimer = setInterval(() => {
    const lo = state.rangeMin ?? 0;
    const hi = state.rangeMax ?? parseInt($("scrub").max);
    if (state.idx >= hi) {
      if ($("loopChk").checked) { state.idx = lo; }
      else { stop(); return; }
    } else {
      state.idx = Math.max(lo, state.idx + 1);
    }
    renderFrame();
  }, interval);
}
function togglePlay() { isPlaying() ? stop() : play(); }

$("playBtn").onclick = togglePlay;
$("fpsSel").onchange = () => { if (isPlaying()) { stop(); play(); } };

document.addEventListener("keydown", (e) => {
  const lo = state.rangeMin ?? 0;
  const hi = state.rangeMax ?? parseInt($("scrub").max);
  const step = e.shiftKey ? 10 : 1;
  if (e.key === "ArrowLeft")  { stop(); state.idx = Math.max(lo, state.idx - step); renderFrame(); }
  if (e.key === "ArrowRight") { stop(); state.idx = Math.min(hi, state.idx + step); renderFrame(); }
  if (e.code === "Space" && e.target.tagName !== "SELECT" && e.target.tagName !== "INPUT") {
    e.preventDefault(); togglePlay();
  }
});

loadRuns().catch((e) => { document.body.innerHTML = "<p style='color:#f66;padding:20px'>Error: " + e.message + "</p>"; });
