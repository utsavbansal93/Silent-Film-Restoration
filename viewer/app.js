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
  state["frames" + which] = frames;
  state["stage" + which] = stagePath;
  $("title" + which).textContent = label;
  const total = Math.min(state.framesA.length, state.framesB.length);
  $("scrub").max = Math.max(0, total - 1);
  $("frameTotal").textContent = total;
}

function renderFrame() {
  const i = state.idx;
  const fa = state.framesA[i];
  const fb = state.framesB[i];
  if (fa) $("imgA").src = `/runs/${state.run}/${state.stageA}/${fa}`;
  if (fb) $("imgB").src = `/runs/${state.run}/${state.stageB}/${fb}`;
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
    const total = parseInt($("scrub").max);
    if (state.idx >= total) {
      if ($("loopChk").checked) { state.idx = 0; }
      else { stop(); return; }
    } else {
      state.idx += 1;
    }
    renderFrame();
  }, interval);
}
function togglePlay() { isPlaying() ? stop() : play(); }

$("playBtn").onclick = togglePlay;
$("fpsSel").onchange = () => { if (isPlaying()) { stop(); play(); } };

document.addEventListener("keydown", (e) => {
  const total = parseInt($("scrub").max);
  const step = e.shiftKey ? 10 : 1;
  if (e.key === "ArrowLeft") { stop(); state.idx = Math.max(0, state.idx - step); renderFrame(); }
  if (e.key === "ArrowRight") { stop(); state.idx = Math.min(total, state.idx + step); renderFrame(); }
  if (e.code === "Space" && e.target.tagName !== "SELECT" && e.target.tagName !== "INPUT") {
    e.preventDefault(); togglePlay();
  }
});

loadRuns().catch((e) => { document.body.innerHTML = "<p style='color:#f66;padding:20px'>Error: " + e.message + "</p>"; });
