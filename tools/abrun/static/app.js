/* abrun UI: vanilla JS, no build step, no external assets.
   fetch wrappers, EventSource with Last-Event-ID replay + polling fallback, live plan preview (POST /api/plan on
   every change), runs table, log console (capped at 2000 lines), compare table, copy-markdown / download-json,
   three inline SVG plots. */
(() => {
'use strict';

const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const fmtE = (x, nd = 8) => (x === null || x === undefined) ? '-' : Number(x).toFixed(nd);
const fmtSci = (x) => (x === null || x === undefined) ? '-' : Number(x).toExponential(1);
const TERMINAL = new Set(['done', 'failed', 'cancelled']);
const CONSOLE_CAP = 2000;
const SERIES = ['var(--series-1)', 'var(--series-2)', 'var(--series-3)', 'var(--series-4)', 'var(--series-5)', 'var(--series-6)', 'var(--series-7)', 'var(--series-8)'];
const HIST_BINS = ['1-32', '33-128', '129-512', '513-1K', '1K-2K', '2K-4K', '4K-8K', '8K+'];

const S = {
  health: null, machine: null, binaries: [], presets: [], modes: [], runs: [],
  selected: null, logSeq: 0, logLines: [], es: null, pollTimer: null, esErrors: 0, logState: null, logLive: false,
  cmpSel: new Set(), cmpRef: null, cmpResult: null, records: {}, globalEs: null, planTimer: null, refreshTimer: null,
  showDev: false, stateTick: null, runDirs: {},
};

async function api(path, opts = {}) {
  const r = await fetch(path, Object.assign({headers: {'content-type': 'application/json'}}, opts));
  const ct = r.headers.get('content-type') || '';
  const body = ct.includes('json') ? await r.json() : await r.text();
  if (!r.ok) {
    const e = new Error((body && body.error) || r.statusText);
    e.status = r.status; e.violations = (body && body.violations) || []; e.body = body;
    throw e;
  }
  return body;
}
const post = (path, body) => api(path, {method: 'POST', body: JSON.stringify(body || {})});
const del = (path) => api(path, {method: 'DELETE'});

function msg(text, cls = '') {
  const m = $('#launch-msg'); m.textContent = text; m.className = 'msg ' + cls;
}
async function copyText(text) {
  try { await navigator.clipboard.writeText(text); return true; } catch (e) {
    const ta = document.createElement('textarea'); ta.value = text; document.body.appendChild(ta); ta.select();
    try { document.execCommand('copy'); } catch (e2) { /* ignore */ } ta.remove(); return true;
  }
}
function download(name, text, type = 'application/json') {
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([text], {type})); a.download = name;
  document.body.appendChild(a); a.click(); setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 500);
}
function elapsed(since) {
  if (!since) return '';
  const t = Date.parse(since.replace(' ', 'T'));
  if (isNaN(t)) return '';
  const s = Math.max(0, Math.round((Date.now() - t) / 1000));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), ss = s % 60;
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(ss).padStart(2, '0')}`;
}

// ------------------------------------------------------------------ status bar
function chip(id, text, cls, title) {
  const c = $(id); c.textContent = text; c.className = 'chip ' + (cls || ''); if (title !== undefined) c.title = title;
}
async function pollMachine() {
  let m;
  try { m = await api('/api/machine'); } catch (e) { chip('#chip-worker', 'server unreachable', 'bad'); return; }
  S.machine = m;
  const l = m.load1;
  chip('#chip-load', `load1 ${l === null || l === undefined ? '-' : l.toFixed(2)}`, l < 1.5 ? 'ok' : (l < 5 ? 'warn' : 'bad'));
  const tl = $('#chip-tlock');
  if (m.timing_lock && m.timing_lock.present) {
    tl.classList.remove('hidden'); tl.className = 'chip bad';
    tl.textContent = `TIMING LOCK PRESENT: every run will wait${m.timing_lock.text ? ' (' + m.timing_lock.text + ')' : ''}`;
  } else tl.classList.add('hidden');
  const ml = m.machlock || {};
  chip('#chip-machlock', ml.available === false ? 'machlock: unavailable (guard mode)' : (ml.bench_held ? 'machlock: bench held by another session' : 'machlock: free'),
       ml.available === false ? 'muted' : (ml.bench_held ? 'warn' : 'ok'), (ml.log_tail || []).join('\n') || ml.path || '');
  const fs = m.foreign_solvers || [];
  chip('#chip-foreign', `foreign ${fs.length}`, fs.length ? 'warn' : 'ok', fs.map((f) => `${f.pid} ${f.cmd}`).join('\n') || 'no foreign solvers');
  chip('#chip-power', `power ${m.power}`, m.power === 'AC' ? 'ok' : 'warn');
  chip('#chip-disk', `disk ${m.disk_free_gb === null ? '-' : m.disk_free_gb + ' GB free'}`, m.disk_warn ? 'warn' : '');
  const w = m.worker || {};
  let wt = w.state === 'busy' ? `running ${w.current}` : 'idle';
  if (w.queued) wt += ` / ${w.queued} queued`;
  if (w.held) wt += ` / held: ${w.paused_reason || 'a previous solver is still alive'}`;
  else if (w.paused) wt += w.paused_reason ? ` / paused: ${w.paused_reason}` : ' / paused';
  chip('#chip-worker', `worker: ${wt}`, (w.paused || w.held) ? 'warn' : (w.state === 'busy' ? 'ok' : ''), w.paused_reason || '');
  $('#btn-pause').textContent = w.paused ? 'Resume' : 'Pause';
  chip('#chip-timed', `timed runs possible: ${m.timed_possible ? 'yes' : 'no'}`, m.timed_possible ? 'ok' : 'warn',
       (m.timed_blockers || []).join('\n') || 'load1 < 1.5, AC, no timing lock, no bench held, no foreign solvers');
  $('#chip-fake').classList.toggle('hidden', !m.fake_pw);
  $('#version').textContent = S.health ? 'v' + S.health.version : '';
  updateTimedControl();
  updateWillWait();
}
function updateWillWait() {
  const m = S.machine; const b = $('#btn-run');
  if (!m) return;
  const lock = (document.querySelector('input[name=lock]:checked') || {}).value || 'auto';
  const timed = $('#f-timed') && $('#f-timed').checked;
  const why = [];
  if (m.timing_lock && m.timing_lock.present) why.push('timing lock present');
  if (m.machlock && m.machlock.bench_held) {
    // share never waits on the bench (foreign solvers are recorded instead); auto resolves to build, build/bench queue behind it
    if (lock === 'auto' && !timed) why.push('bench held -> auto takes machlock build and waits');
    else if (lock !== 'share' || timed) why.push(`bench held -> machlock ${timed ? 'bench' : lock} waits`);
  }
  b.textContent = why.length ? `Run (will wait: ${why.join('; ')})` : (lock === 'share' && m.machlock && m.machlock.bench_held ? 'Run (share: no wait; foreign solvers recorded)' : 'Run');
}

// ------------------------------------------------------------------ launch form
function currentBinary() { return S.binaries.find((b) => b.id === $('#f-binary').value) || null; }
function currentMode() { const r = document.querySelector('input[name=mode]:checked'); return r ? r.value : 'ab_gpu_all'; }
function currentPreset() { return S.presets.find((p) => p.id === $('#f-preset').value) || null; }

function buildPresetSelect() {
  const sel = $('#f-preset');
  sel.innerHTML = S.presets.map((p) => `<option value="${esc(p.id)}" ${p.available === false ? 'class="muted"' : ''}>${esc(p.label)}${p.available === false ? ' [unavailable]' : ''}</option>`).join('');
  sel.value = 'si2';
}
function renderPresetCard() {
  const p = currentPreset(); const card = $('#preset-card');
  $('#row-input-path').classList.toggle('hidden', !p || p.id !== 'custom');
  if (!p) { card.innerHTML = ''; return; }
  const lines = [];
  if (p.available === false) lines.push(`<div class="line cross">unavailable: ${esc(p.reason)}</div>`);
  lines.push(`<div class="line"><span class="k">source</span> ${esc(p.source || '(custom path)')}</div>`);
  if (p.input_sha256) lines.push(`<div class="line"><span class="k">input sha256</span> ${esc(p.input_sha256.slice(0, 12))}</div>`);
  lines.push(`<div class="line"><span class="k">nat</span> ${p.nat ?? '-'} <span class="k">nbnd</span> ${p.nbnd ?? 'default'} <span class="k">k-points</span> ${esc(p.kpoints || '-')}</div>`);
  if (p.pseudos && p.pseudos.length) lines.push(`<div class="line"><span class="k">pseudos</span> ${p.pseudos.map((u) => `${esc(u.file)} <span class="${u.pinned ? 'tick' : 'muted'}">${u.pinned ? '&#10003; sha pinned' : '(unpinned)'}</span>`).join(', ')}</div>`);
  if (p.expected_E_Ry !== null && p.expected_E_Ry !== undefined) lines.push(`<div class="line"><span class="k">expected E</span> ${fmtE(p.expected_E_Ry)} Ry <span class="muted">(${esc(p.expected_source)})</span></div>`);
  else lines.push(`<div class="line"><span class="k">expected E</span> none: the first reference run becomes the pinned reference</div>`);
  const walls = Object.entries(p.wall_obs || {});
  if (walls.length) lines.push(`<div class="line"><span class="k">campaign walls (observation)</span> ${walls.map(([k, v]) => `${esc(k)} ${v} s`).join(', ')} <span class="muted">${esc(p.wall_source || '')}</span></div>`);
  card.innerHTML = lines.join('');
}
async function toggleInput() {
  const pre = $('#preset-input-pre');
  if (!pre.classList.contains('hidden')) { pre.classList.add('hidden'); $('#btn-show-input').textContent = 'Show rewritten input'; return; }
  const p = currentPreset(); if (!p) return;
  const q = p.id === 'custom' ? `?input_path=${encodeURIComponent($('#f-input-path').value)}` : '';
  try { pre.textContent = await api(`/api/presets/${p.id}/input${q}`); } catch (e) { pre.textContent = `error: ${e.message}`; }
  pre.classList.remove('hidden'); $('#btn-show-input').textContent = 'Hide rewritten input';
}

function buildBinarySelect(keep) {
  const sel = $('#f-binary');
  const groups = [
    ['Reference (apple-bottom-free)', (b) => b.group === 'reference'],
    ['Frozen: citable', (b) => b.group === 'frozen' && b.citable],
    ['Frozen: experiments / parity-only / pre-1.4.0 / experimental', (b) => b.group === 'frozen' && !b.citable],
    ['Static relink', (b) => b.group === 'static'],
    ['Dev builds (never citable; rebuilt by other sessions; a LIVE apple-bottom/build dylib is flagged on the card)', (b) => b.group === 'dev'],
    ['Other', (b) => !['reference', 'frozen', 'static', 'dev'].includes(b.group)],
  ];
  sel.innerHTML = groups.map(([name, f]) => {
    const bs = S.binaries.filter(f);
    if (!bs.length) return '';
    return `<optgroup label="${esc(name)}">${bs.map((b) => `<option value="${esc(b.id)}" ${b.launchable ? '' : 'disabled'}>${esc(b.label)}${b.launchable ? '' : ' [not launchable]'}</option>`).join('')}</optgroup>`;
  }).join('');
  const want = keep || (S.health && S.health.fake_pw ? 'fake' : (S.binariesMeta && S.binariesMeta.default_binary_id) || '33fbaa8c8944');
  if (S.binaries.some((b) => b.id === want)) sel.value = want;
  else if (S.binaries.length) sel.value = S.binaries.find((b) => b.launchable)?.id || S.binaries[0].id;
}
async function loadBinaries() {
  const d = await api(`/api/binaries${S.showDev ? '?all=1' : ''}`);
  S.binaries = d.binaries; S.binariesMeta = d;
}
function renderBinaryCard() {
  const b = currentBinary(); const card = $('#binary-card');
  if (!b) { card.innerHTML = ''; return; }
  const L = b.links || {};
  const links = ['applebottom', 'openblas', 'accelerate', 'fftw3_omp'].map((k) => `${k}: ${L[k] ? '&#10003;' : '&#10007;'}`).join(' &nbsp;');
  const prov = (b.provenance || '').split('\n');
  const provHead = prov.slice(0, 6).join('\n'), provMore = prov.length > 6;
  const warn = [];
  if (b.omp1_only) warn.push('OMP_NUM_THREADS=1 only');
  if (b.live_library) warn.push('LIVE library (loads apple-bottom/build): never citable');
  if (b.sha_verified === false) warn.push('SHA256SUMS mismatch');
  card.innerHTML = `
    <div class="line"><span class="badge ${esc(b.class)}">${esc(b.class)}</span> ${b.citable ? '<span class="tick">citable</span>' : '<span class="muted">not citable</span>'} <span class="muted">${esc(b.class_reason)}</span></div>
    <div class="line"><span class="k">path</span> ${esc(b.path)}</div>
    <div class="line"><span class="k">QE</span> ${esc(b.qe_version || '?')} ${esc(b.qe_build || '')}</div>
    <div class="line"><span class="k">links</span> ${links} &nbsp;<span class="k">libmpi</span> ${esc(L.libmpi ? L.libmpi.split('/').pop() : 'none (direct launch)')}</div>
    <div class="line"><span class="k">nm _ab_</span> U ${b.ab_symbols?.U ?? 0} T ${b.ab_symbols?.T ?? 0} <span class="k">linkage</span> ${esc(b.ab_linkage)}</div>
    ${b.dylib_path ? `<div class="line"><span class="k">dylib</span> ${esc(b.dylib_path)} <span class="k">build</span> ${esc(b.ab_build_info || 'no ab_build_info (pre-1.4.0)')}</div>` : ''}
    <div class="line"><span class="k">pw.x sha</span> ${esc((b.sha256 || '?').slice(0, 12))} ${b.sha_verified ? '<span class="tick">&#10003; SHA256SUMS</span>' : (b.sha_verified === false ? '<span class="cross">&#10007; SHA256SUMS</span>' : '')}
      ${b.dylib_sha256 ? `<span class="k">dylib sha</span> ${esc(b.dylib_sha256.slice(0, 12))}` : ''} <span class="k">schema</span> ${esc(b.profile_schema)}</div>
    ${b.features && b.features.length ? `<div class="line">${b.features.map((f) => `<span class="feat">${esc(f)}</span>`).join('')}</div>` : ''}
    ${b.blas ? `<div class="line"><span class="k">BLAS</span> ${esc(b.blas)}</div>` : ''}
    ${warn.length ? `<div class="line cross">${esc(warn.join('; '))}</div>` : ''}
    ${b.provenance ? `<div class="line"><span class="k">PROVENANCE</span> <button type="button" class="btn small" id="btn-prov">expand</button></div><pre class="mono" id="prov-pre">${esc(provHead)}${provMore ? '\n...' : ''}</pre>` : ''}`;
  const pb = $('#btn-prov');
  if (pb) pb.onclick = () => { const pre = $('#prov-pre'); const full = pre.dataset.full === '1'; pre.textContent = full ? provHead + (provMore ? '\n...' : '') : b.provenance; pre.dataset.full = full ? '0' : '1'; pb.textContent = full ? 'expand' : 'collapse'; };
}

function buildModeRadios() {
  const fs = $('#mode-radios');
  fs.querySelectorAll('label').forEach((l) => l.remove());
  for (const m of S.modes) {
    const l = document.createElement('label');
    l.innerHTML = `<input type="radio" name="mode" value="${esc(m.id)}"> <b>${esc(m.id)}</b>: ${esc(m.label)}<small>${esc(m.short)}</small>`;
    fs.appendChild(l);
  }
  const d = fs.querySelector('input[value=ab_gpu_all]'); if (d) d.checked = true;
}
function modeViolation(m, b) {
  if (!b) return 'no binary';
  if (m.id === 'reference') return b.ab_linkage === 'none' ? '' : `mode reference requires an apple-bottom-free binary; ${b.id} ${b.ab_linkage === 'dynamic' ? 'links @rpath/libapplebottom.dylib' : 'has static _ab_ symbols'}`;
  return ['dynamic', 'static'].includes(b.ab_linkage) ? '' : `mode ${m.id} requires an apple-bottom-linked binary; ${b.id} has ab_linkage ${b.ab_linkage}`;
}
function updateModeRadios() {
  const b = currentBinary();
  let firstOk = null, curOk = false;
  for (const m of S.modes) {
    const input = document.querySelector(`input[name=mode][value=${m.id}]`); if (!input) continue;
    const v = modeViolation(m, b);
    input.disabled = !!v; input.parentElement.classList.toggle('disabled', !!v); input.parentElement.title = v;
    if (!v && !firstOk) firstOk = input;
    if (!v && input.checked) curOk = true;
  }
  if (!curOk && firstOk) firstOk.checked = true;
}
function updateAdvanced() {
  const b = currentBinary(); const mode = currentMode();
  const ab = b && b.ab_linkage !== 'none';
  const omp = $('#f-omp'), allow = $('#f-allow-omp');
  allow.disabled = !ab || !!(b && b.omp1_only);
  allow.parentElement.title = !ab ? 'reference binaries take OMP as given' : (b.omp1_only ? `${b.id} is OMP=1 only (PROVENANCE / -O3 pw.x)` : 'apple-bottom binaries run at OMP=1 unless allowed');
  if (ab && !allow.checked) { omp.value = 1; omp.disabled = true; omp.title = 'locked to 1 for apple-bottom binaries (tick allow OMP>1)'; }
  else { omp.disabled = false; omp.title = ''; }
  const mg = $('#f-min-gpu-dim'); mg.disabled = mode !== 'ab_gpu'; mg.title = mode === 'ab_gpu' ? '' : 'only for mode ab_gpu (ab_gpu_all fixes 0)';
  if (mode !== 'ab_gpu') mg.value = '';
  const tr = $('#f-trace'); const hasTrace = !!(b && (b.features || []).includes('AB_TRACE')) && mode !== 'reference';
  tr.disabled = !hasTrace; if (!hasTrace) tr.checked = false; tr.parentElement.title = hasTrace ? '' : 'the dylib does not list AB_TRACE';
  const pj = $('#f-profile-json'); const ranks = Number($('#f-ranks').value || 1);
  pj.parentElement.title = ranks > 1 ? 'AB_PROFILE_JSON has no pid suffix, last rank wins; AB_REPORT per rank is summed instead (auto-off at ranks>1)' : '';
  $('#f-report').disabled = !ab; $('#f-profile-json').disabled = !ab;
  updateTimedControl();
}
function updateTimedControl() {
  const b = currentBinary(); const m = S.machine; const t = $('#f-timed'); const bl = [];
  if (b && !b.citable) bl.push(`binary ${b.id} is class ${b.class} (not citable)`);
  if (b && b.live_library) bl.push('binary loads the live apple-bottom build');
  if (m && !m.timed_possible) bl.push(...(m.timed_blockers || []));
  t.disabled = bl.length > 0; if (bl.length) t.checked = false;
  $('#timed-blockers').textContent = bl.length ? 'timed disabled: ' + bl.join('; ') : '';
}
function requestBody() {
  const b = currentBinary(); const preset = $('#f-preset').value;
  const mg = $('#f-min-gpu-dim').value;
  return {
    preset, input_path: preset === 'custom' ? ($('#f-input-path').value || null) : null,
    binary_id: b ? b.id : $('#f-binary').value, mode: currentMode(),
    ranks: Number($('#f-ranks').value || 1), nk: Number($('#f-nk').value || 1), omp: Number($('#f-omp').value || 1),
    min_gpu_dim: mg === '' ? null : Number(mg), profile_json: $('#f-profile-json').checked, report: $('#f-report').checked,
    trace: $('#f-trace').checked, verify_dylib: $('#f-verify-dylib').checked,
    lock: (document.querySelector('input[name=lock]:checked') || {}).value || 'auto', timed: $('#f-timed').checked,
    allow_omp_gt1: $('#f-allow-omp').checked, allow_dev: !!(b && b.group === 'dev'), purge_tmp: $('#f-purge-tmp').checked,
    label: $('#f-label').value || '',
  };
}
function schedulePlan() { clearTimeout(S.planTimer); S.planTimer = setTimeout(refreshPlan, 150); }
async function refreshPlan() {
  updateModeRadios(); updateAdvanced();
  const body = requestBody(); S.lastBody = body;
  let res, violations = [];
  try { res = await post('/api/plan', body); } catch (e) { res = e.body || {}; violations = e.violations.length ? e.violations : [e.message]; }
  const p = res && res.plan; const pre = $('#preview'); const vb = $('#preview-violations');
  vb.classList.toggle('hidden', !violations.length); vb.textContent = violations.join('\n');
  $('#btn-run').disabled = violations.length > 0;
  if (!p) { pre.textContent = '(no plan)'; S.shell = ''; return; }
  const lines = [`cwd: ${p.cwd}`, `lock: requested ${p.lock_requested} -> ${p.lock_would_resolve_to}; launcher ${p.launcher}; timing grade possible: ${p.timing_grade_possible}; OMP ${p.omp_effective}; AB_MIN_GPU_DIM ${p.min_gpu_dim_effective}`];
  for (const name of ['wrapper_argv', 'shim_argv', 'solver_argv']) { lines.push(`${name}:`); for (const t of p[name] || []) lines.push(`    ${t}`); if (!(p[name] || []).length) lines.push('    (none)'); }
  lines.push('env set:'); for (const [k, v] of Object.entries(p.env_set || {})) lines.push(`    ${k}=${v}`);
  lines.push(`env removed: ${(p.env_removed || []).join(', ') || '(none inherited)'}`);
  for (const n of p.notes || []) lines.push(`note: ${n}`);
  lines.push(`shell: ${p.shell}`);
  pre.textContent = lines.join('\n'); S.shell = p.shell;
}
function bindForm() {
  $('#f-preset').addEventListener('change', () => { renderPresetCard(); $('#preset-input-pre').classList.add('hidden'); $('#btn-show-input').textContent = 'Show rewritten input'; schedulePlan(); });
  $('#f-input-path').addEventListener('input', schedulePlan);
  $('#btn-show-input').addEventListener('click', toggleInput);
  $('#f-binary').addEventListener('change', () => { renderBinaryCard(); schedulePlan(); });
  $('#f-show-dev').addEventListener('change', async () => { S.showDev = $('#f-show-dev').checked; const keep = $('#f-binary').value; await loadBinaries(); buildBinarySelect(keep); renderBinaryCard(); schedulePlan(); });
  $('#launch-form').addEventListener('change', (e) => { if (e.target.id !== 'f-preset' && e.target.id !== 'f-binary' && e.target.id !== 'f-show-dev') schedulePlan(); if (e.target.name === 'lock' || e.target.id === 'f-timed') updateWillWait(); });
  $('#launch-form').addEventListener('input', (e) => { if (['f-ranks', 'f-nk', 'f-omp', 'f-min-gpu-dim', 'f-label'].includes(e.target.id)) schedulePlan(); });
  $('#btn-copy-shell').addEventListener('click', async () => { if (S.shell) { await copyText(S.shell); msg('shell line copied', 'ok'); } });
  $('#btn-run').addEventListener('click', async () => {
    try { const r = await post('/api/runs', requestBody()); msg(`queued ${r.id} (position ${r.position})`, 'ok'); await refreshRuns(); selectRun(r.id); }
    catch (e) { msg(`${e.message}${e.violations.length ? '\n' + e.violations.join('\n') : ''}`, 'err'); }
  });
  $('#btn-abset').addEventListener('click', async () => {
    const b = requestBody(); const legs = $$('input[name=leg]:checked').map((i) => i.value);
    if (!legs.length) { msg('tick at least one leg', 'err'); return; }
    const body = {preset: b.preset, input_path: b.input_path, binary_id: b.binary_id, ranks: b.ranks, nk: b.nk, omp: b.omp, lock: b.lock, timed: b.timed,
                  legs, label: b.label, profile_json: b.profile_json, report: b.report, verify_dylib: b.verify_dylib, purge_tmp: b.purge_tmp, allow_omp_gt1: b.allow_omp_gt1, allow_dev: b.allow_dev};
    try { const r = await post('/api/ab_set', body); msg(`queued A/B set: ${r.ids.join(', ')}${r.ref ? ' (reference ' + r.ref + ' pinned)' : ''}`, 'ok'); await refreshRuns(); if (r.ids.length) selectRun(r.ids[0]); }
    catch (e) { msg(`${e.message}${e.violations.length ? '\n' + e.violations.join('\n') : ''}`, 'err'); }
  });
  $('#btn-clear').addEventListener('click', () => {
    $('#f-preset').value = 'si2'; renderPresetCard(); buildBinarySelect(); renderBinaryCard();
    const d = document.querySelector('input[name=mode][value=ab_gpu_all]'); if (d) d.checked = true;
    for (const [id, v] of [['f-ranks', 1], ['f-nk', 1], ['f-omp', 1], ['f-min-gpu-dim', ''], ['f-label', ''], ['f-input-path', '']]) $('#' + id).value = v;
    for (const [id, v] of [['f-allow-omp', false], ['f-profile-json', true], ['f-report', true], ['f-trace', false], ['f-verify-dylib', true], ['f-timed', false], ['f-purge-tmp', true]]) $('#' + id).checked = v;
    document.querySelector('input[name=lock][value=auto]').checked = true;
    msg(''); schedulePlan();
  });
  $('#btn-pause').addEventListener('click', async () => { const paused = S.machine && S.machine.worker && S.machine.worker.paused; await post(paused ? '/api/queue/resume' : '/api/queue/pause'); pollMachine(); });
  $('#btn-refresh').addEventListener('click', refreshRuns);
  for (const id of ['flt-preset', 'flt-mode', 'flt-binary', 'flt-state']) $('#' + id).addEventListener('change', renderRuns);
  $('#f-stream').addEventListener('change', renderConsole);
  $('#btn-log-cancel').addEventListener('click', () => { if (S.selected) cancelRun(S.selected); });
  $('#btn-compare').addEventListener('click', runCompare);
  $('#btn-copy-md').addEventListener('click', async () => { if (S.cmpResult) { await copyText(S.cmpResult.markdown); $('#cmp-msg').textContent = 'markdown copied'; } });
  $('#btn-dl-json').addEventListener('click', () => { if (S.cmpResult) download('compare.json', JSON.stringify(S.cmpResult, null, 2)); });
}

// ------------------------------------------------------------------ runs table
function scheduleRefresh() { clearTimeout(S.refreshTimer); S.refreshTimer = setTimeout(refreshRuns, 300); }
async function refreshRuns() {
  try { const d = await api('/api/runs?limit=500'); S.runs = d.runs; } catch (e) { return; }
  for (const r of S.runs) if (r.run_dir) S.runDirs[r.id] = r.run_dir;
  fillFilters(); renderRuns(); renderCmpList();
  if (S.selected) { const r = S.runs.find((x) => x.id === S.selected); if (r) updateSelectedHeader(r); }
}
function fillFilters() {
  const opts = (id, key, label) => {
    const sel = $('#' + id); const cur = sel.value;
    const vals = Array.from(new Set(S.runs.map((r) => r[key]).filter(Boolean))).sort();
    sel.innerHTML = `<option value="">${label}: all</option>` + vals.map((v) => `<option value="${esc(v)}">${esc(v)}</option>`).join('');
    if (vals.includes(cur)) sel.value = cur;
  };
  opts('flt-preset', 'preset', 'preset'); opts('flt-mode', 'mode', 'mode'); opts('flt-binary', 'binary_id', 'binary');
}
function gpuCell(r) {
  if (r.mode === 'reference') return `<span class="gpu na" title="apple-bottom-free binary">n/a</span>`;
  if (r.state !== 'done' && r.state !== 'failed') return `<span class="gpu na">-</span>`;
  if (r.gemm_calls === null || r.gemm_calls === undefined) return `<span class="gpu amber" title="no counters: evidence ${esc(r.evidence_source)}">unverifiable</span>`;
  const badge = r.gpu_badge || (r.gpu_verified ? 'green' : (r.gpu_verified === false ? 'red' : 'amber'));
  const title = `gpu_verified: ${r.gpu_verified} (evidence ${r.evidence_source}; fallbacks ${r.fallbacks})`;
  return `<span class="gpu ${esc(badge)}" title="${esc(title)}">${r.gpu_calls}/${r.gemm_calls}${r.fallbacks ? ' fb' + r.fallbacks : ''}</span>`;
}
function renderRuns() {
  const fp = $('#flt-preset').value, fm = $('#flt-mode').value, fb = $('#flt-binary').value, fs = $('#flt-state').value;
  const rows = S.runs.filter((r) => (!fp || r.preset === fp) && (!fm || r.mode === fm) && (!fb || r.binary_id === fb) && (!fs || r.state === fs));
  $('#runs-count').textContent = `${rows.length} of ${S.runs.length}`;
  const tb = $('#runs-body');
  tb.innerHTML = rows.map((r) => {
    const dE = (r.dE_vs_expected_Ry === null || r.dE_vs_expected_Ry === undefined) ? '-' : fmtSci(r.dE_vs_expected_Ry);
    const wall = r.wall_qe_s === null || r.wall_qe_s === undefined ? '-' : `${Number(r.wall_qe_s).toFixed(2)}<span class="grade ${esc(r.timing_grade || 'untimed')}" title="${esc(r.timing_grade_reason || '')}">${esc(r.timing_grade || 'untimed')}</span>`;
    const term = TERMINAL.has(r.state);
    const others = S.modes.filter((m) => m.id !== r.mode && (r.mode === 'reference') === (m.id === 'reference'));
    const rerun = `<select class="rerun" data-id="${esc(r.id)}" title="rerun on another mode, same binary"><option value="">Rerun on &#9662;</option>${others.map((m) => `<option value="${m.id}">${m.id}</option>`).join('')}</select>`;
    const actions = `${term ? '' : `<button class="btn small danger act" data-act="cancel" data-id="${esc(r.id)}">Cancel</button>`}${rerun}` +
      `<button class="btn small act" data-act="pin" data-id="${esc(r.id)}" title="pin as the reference for this input">${r.pinned_reference ? 'Unpin' : 'Pin'}</button>` +
      `${term ? `<button class="btn small act" data-act="delete" data-id="${esc(r.id)}">Delete</button>` : ''}` +
      `<button class="btn small act" data-act="copydir" data-id="${esc(r.id)}" title="${esc(r.run_dir || '')}">Copy dir</button>`;
    return `<tr class="clickable ${r.id === S.selected ? 'selected' : ''}" data-id="${esc(r.id)}">
      <td><span class="state ${esc(r.state)}" title="${esc(r.state_why || '')}">${esc(r.state)}</span>${r.position !== null && r.position !== undefined ? ` <span class="muted">#${r.position}</span>` : ''}</td>
      <td>${esc(r.preset)}</td><td title="${esc(r.binary_id)}">${esc(r.binary_short)}</td><td><span class="mode ${esc(r.mode)}">${esc(r.mode)}</span></td>
      <td class="num">${fmtE(r.total_energy_Ry)}</td><td class="num" title="vs the preset's expected energy">${dE}</td><td class="num">${r.scf_iterations ?? '-'}</td>
      <td class="num">${wall}</td><td>${gpuCell(r)}</td><td>${esc(r.lock_effective || '-')}</td>
      <td title="${esc(r.id)}">${r.pinned_reference ? '<span class="pin" title="pinned reference">&#9733;</span> ' : ''}${esc(r.label || '')}<div class="muted small">${esc(r.id)}</div></td>
      <td>${actions}</td></tr>`;
  }).join('');
  tb.querySelectorAll('tr').forEach((tr) => tr.addEventListener('click', (e) => { if (e.target.closest('button,select')) return; selectRun(tr.dataset.id); }));
  tb.querySelectorAll('button.act').forEach((b) => b.addEventListener('click', (e) => { e.stopPropagation(); runAction(b.dataset.act, b.dataset.id); }));
  tb.querySelectorAll('select.rerun').forEach((s) => s.addEventListener('change', async (e) => {
    e.stopPropagation(); const mode = s.value; if (!mode) return; s.value = '';
    try { const r = await post(`/api/runs/${s.dataset.id}/rerun`, {mode}); msg(`queued rerun ${r.id} (${mode})`, 'ok'); await refreshRuns(); selectRun(r.id); }
    catch (err) { msg(`${err.message}${err.violations.length ? '\n' + err.violations.join('\n') : ''}`, 'err'); }
  }));
  // campaign reference pseudo-rows for the si64 presets visible here
  const visible = new Set(rows.map((r) => r.preset)); const lp = $('#f-preset').value; if (lp) visible.add(lp);
  $('#campaign-rows').innerHTML = S.presets.filter((p) => visible.has(p.id) && p.expected_E_Ry !== null && p.expected_E_Ry !== undefined).map((p) =>
    `<tr class="campaign" title="${esc(p.expected_source || '')}"><td>campaign</td><td>${esc(p.id)}</td><td>frozen 33fbaa 1.4.0</td><td>ab_cpu / ab_gpu</td>
     <td class="num">${fmtE(p.expected_E_Ry)}</td><td class="num">ref</td><td>-</td>
     <td class="num">${Object.entries(p.wall_obs || {}).map(([k, v]) => `${esc(k)} ${v}`).join('; ')} <span class="grade untimed">observation</span></td>
     <td colspan="4" class="small">${esc(p.wall_source || '')}</td></tr>`).join('');
}
async function runAction(act, id) {
  try {
    if (act === 'cancel') await cancelRun(id);
    else if (act === 'pin') { const r = S.runs.find((x) => x.id === id); if (r && r.pinned_reference) await del(`/api/runs/${id}/pin`); else await post(`/api/runs/${id}/pin`); }
    else if (act === 'delete') { if (!confirm(`delete run ${id} and its directory?`)) return; await del(`/api/runs/${id}`); if (S.selected === id) { S.selected = null; } S.cmpSel.delete(id); }
    else if (act === 'copydir') { await copyText(S.runDirs[id] || id); msg(`copied ${S.runDirs[id] || id}`, 'ok'); return; }
    await refreshRuns();
  } catch (e) { msg(`${act} ${id}: ${e.message}`, 'err'); }
}
async function cancelRun(id) { const r = await post(`/api/runs/${id}/cancel`); msg(`${id}: ${r.state}`, 'ok'); scheduleRefresh(); }

// ------------------------------------------------------------------ log console
function closeStream() {
  if (S.es) { S.es.close(); S.es = null; }
  if (S.pollTimer) { clearInterval(S.pollTimer); S.pollTimer = null; }
  if (S.stateTick) { clearInterval(S.stateTick); S.stateTick = null; }
}
function updateSelectedHeader(r) {
  $('#log-title').textContent = `Log: ${r.id}`;
  $('#btn-log-cancel').classList.toggle('hidden', TERMINAL.has(r.state));
  for (const [id, f] of [['dl-out', 'out.txt'], ['dl-err', 'err.txt'], ['dl-profile', 'ab_profile.json'], ['dl-summary', 'summary.json']]) {
    const a = $('#' + id); a.href = `/api/runs/${r.id}/files/${f}`; a.classList.toggle('disabled', !TERMINAL.has(r.state) && f !== 'out.txt' && f !== 'err.txt');
  }
}
function setLogState(state, why, since) {
  S.logState = {state, why, since};
  const el = $('#log-state');
  const render = () => {
    const e = S.logState; if (!e) { el.textContent = ''; return; }
    const waiting = e.state === 'waiting_timing_lock' || e.state === 'waiting_guard' || e.state === 'queued' || e.state === 'cancelling';
    el.innerHTML = `<span class="state ${esc(e.state)}">${esc(e.state)}</span> ${esc(e.why || '')}${waiting && e.since ? ` <span class="muted">since ${esc(e.since)} (${elapsed(e.since)})</span>` : ''}`;
  };
  render();
  if (S.stateTick) clearInterval(S.stateTick);
  if (!TERMINAL.has(state)) S.stateTick = setInterval(render, 1000);
}
function setProgress(p) {
  if (!p) return;
  const pct = p.max ? Math.min(100, 100 * (p.iteration || p.value || 0) / p.max) : 0;
  $('#progress-fill').style.width = pct + '%';
  $('#progress-text').textContent = `iteration #${p.iteration ?? p.value ?? 0} / ${p.max ?? '?'}${p.scf_accuracy_Ry !== null && p.scf_accuracy_Ry !== undefined ? `, estimated scf accuracy ${Number(p.scf_accuracy_Ry).toExponential(2)} Ry` : ''}`;
}
function lineClass(l) {
  const [, stream, text] = l;
  if (stream === 'guard') return 'guard';
  if (text.startsWith('+') && text.includes('suppressed')) return 'marker';
  if (stream === 'err' && (text.startsWith('AB_') || text.startsWith('=== apple-bottom') || text.startsWith('ab_prof') || text.includes('libapplebottom'))) return 'ab';
  return stream;
}
function lineVisible(l, f) {
  const c = lineClass(l);
  if (f === 'all') return true;
  if (f === 'out') return l[1] === 'out';
  if (f === 'err') return l[1] === 'err';
  if (f === 'ab') return c === 'ab' || c === 'marker';
  if (f === 'guard') return c === 'guard';
  return true;
}
function appendLine(l) {
  if (l[0] <= S.logSeq) return;
  S.logSeq = l[0]; S.logLines.push(l);
  if (S.logLines.length > CONSOLE_CAP) S.logLines.splice(0, S.logLines.length - CONSOLE_CAP);
  if (!lineVisible(l, $('#f-stream').value)) return;
  const c = $('#log-console'); const span = document.createElement('span'); span.className = 'ln ' + lineClass(l); span.textContent = l[2]; c.appendChild(span);
  while (c.childElementCount > CONSOLE_CAP) c.removeChild(c.firstChild);
  if ($('#f-autoscroll').checked) c.scrollTop = c.scrollHeight;
}
function renderConsole() {
  const c = $('#log-console'); const f = $('#f-stream').value; c.innerHTML = '';
  const frag = document.createDocumentFragment();
  for (const l of S.logLines) if (lineVisible(l, f)) { const s = document.createElement('span'); s.className = 'ln ' + lineClass(l); s.textContent = l[2]; frag.appendChild(s); }
  c.appendChild(frag); if ($('#f-autoscroll').checked) c.scrollTop = c.scrollHeight;
}
async function selectRun(id) {
  closeStream();
  S.selected = id; S.logSeq = 0; S.logLines = []; $('#log-console').innerHTML = ''; $('#result-strip').classList.add('hidden'); $('#log-suppressed').textContent = '';
  renderRuns();
  const r = S.runs.find((x) => x.id === id); if (r) updateSelectedHeader(r);
  let d;
  try { d = await api(`/api/runs/${id}/log?since=0&max=${CONSOLE_CAP}`); } catch (e) { $('#log-state').textContent = `log: ${e.message}`; return; }
  for (const l of d.lines) appendLine(l);
  setLogState(d.state, d.state_why, d.state_since); setProgress(d.progress);
  if (d.suppressed_profile_lines) $('#log-suppressed').textContent = `+${d.suppressed_profile_lines} AB_PROFILE per-call lines suppressed from the stream (see err.txt)`;
  if (TERMINAL.has(d.state)) { showResult(id); return; }
  openRunStream(id, d.event_seq || 0);
}
function openRunStream(id, since) {
  S.esErrors = 0;
  const es = new EventSource(`/api/runs/${id}/events?since=${since}`); S.es = es;
  es.onmessage = (ev) => {
    S.esErrors = 0;
    let m; try { m = JSON.parse(ev.data); } catch (e) { return; }
    handleRunEvent(id, m.type, m.data);
  };
  es.onerror = () => {
    S.esErrors += 1;
    if (S.esErrors >= 2 && S.es === es) { es.close(); S.es = null; startPolling(id); }
  };
}
function handleRunEvent(id, type, data) {
  if (data && data.run_id && data.run_id !== id) return;
  if (type === 'muppet.log') { appendLine([data.seq, data.stream, data.text]); if (data.text && data.text.startsWith('+') && data.text.includes('suppressed')) $('#log-suppressed').textContent = data.text; }
  else if (type === 'progress') setProgress({iteration: data.value, max: data.max, scf_accuracy_Ry: data.scf_accuracy_Ry});
  else if (type === 'abrun.state') { setLogState(data.state, data.why, data.since); scheduleRefresh(); }
  else if (type === 'muppet.done') { closeStream(); scheduleRefresh(); showResult(id); }
  else if (type === 'executed' || type === 'execution_error' || type === 'execution_interrupted') scheduleRefresh();
}
function startPolling(id) {
  $('#chip-conn').textContent = 'events: polling fallback'; $('#chip-conn').className = 'chip warn';
  S.pollTimer = setInterval(async () => {
    let d; try { d = await api(`/api/runs/${id}/log?since=${S.logSeq}&max=${CONSOLE_CAP}`); } catch (e) { return; }
    for (const l of d.lines) appendLine(l);
    setLogState(d.state, d.state_why, d.state_since); setProgress(d.progress);
    if (TERMINAL.has(d.state)) { closeStream(); scheduleRefresh(); showResult(id); }
  }, 2000);
}
async function showResult(id) {
  let rec; try { rec = await api(`/api/runs/${id}`); } catch (e) { return; }
  S.records[id] = rec;
  const r = rec.result || {}, ab = rec.ab || {}, e = rec.etiquette || {};
  const tick = (v) => v === true ? '<span class="tick">&#10003;</span>' : (v === false ? '<span class="cross">&#10007;</span>' : '<span class="muted">?</span>');
  const lines = [
    `<b>${esc(rec.state)}</b> ${esc(rec.state_why || '')}`,
    `E_tot ${fmtE(r.total_energy_Ry)} Ry${r.expected_E_Ry !== null && r.expected_E_Ry !== undefined ? ` (dE vs expected ${fmtSci(r.dE_vs_expected_Ry)} Ry)` : ''}, ${r.scf_iterations ?? '-'} SCF iterations${r.converged === false ? ', NOT converged' : ''}`,
    `QE wall ${r.wall_qe_s ?? '-'} s (observation, grade ${esc(e.timing_grade)}: ${esc(e.timing_grade_reason || '')}); runner wall ${rec.wall_runner_s ?? '-'} s; lock ${esc(e.lock_effective)}`,
  ];
  if ((rec.request || {}).mode !== 'reference') {
    lines.push(`lib_loaded ${tick(ab.lib_ok)} ${esc(ab.lib_loaded || 'not seen')}; build ${esc(ab.build || '-')} ${ab.build_matches_dylib === true ? '(matches dylib)' : (ab.build_matches_dylib === false ? '(DOES NOT match dylib)' : '')}`);
    lines.push(`gpu_verified ${esc(String(ab.gpu_verified))}: ${esc(ab.gpu_verified_reason || '')} [${esc(ab.evidence_source)}${ab.schema ? ' ' + ab.schema : ''}]`);
  } else lines.push(`apple-bottom-free: libapplebottom dyld lines ${tick(ab.lib_ok)}`);
  if (ab.warnings && ab.warnings.length) lines.push(`warnings: ${esc(ab.warnings.join('; '))}`);
  const rs = $('#result-strip'); rs.innerHTML = lines.join('<br>'); rs.classList.remove('hidden');
}

// ------------------------------------------------------------------ global events
function openGlobalEvents() {
  const es = new EventSource('/api/events'); S.globalEs = es;
  es.onopen = () => { $('#chip-conn').textContent = 'events: live'; $('#chip-conn').className = 'chip ok'; };
  es.onerror = () => { $('#chip-conn').textContent = 'events: reconnecting'; $('#chip-conn').className = 'chip warn'; };
  es.onmessage = (ev) => {
    let m; try { m = JSON.parse(ev.data); } catch (e) { return; }
    if (['abrun.state', 'executed', 'execution_error', 'execution_interrupted', 'muppet.done', 'status', 'execution_start'].includes(m.type)) scheduleRefresh();
  };
}

// ------------------------------------------------------------------ compare
function renderCmpList() {
  const term = S.runs.filter((r) => TERMINAL.has(r.state));
  const list = $('#cmp-list');
  if (!S.cmpRef || !term.some((r) => r.id === S.cmpRef)) {
    const pinned = term.find((r) => r.pinned_reference);
    const ref = term.find((r) => r.mode === 'reference');
    S.cmpRef = pinned ? pinned.id : (ref ? ref.id : (term[0] ? term[0].id : null));
  }
  list.innerHTML = term.map((r) => `<div class="item"><input type="checkbox" class="cmp-sel" value="${esc(r.id)}" ${S.cmpSel.has(r.id) ? 'checked' : ''}>
    <input type="radio" name="cmpref" value="${esc(r.id)}" ${S.cmpRef === r.id ? 'checked' : ''} title="reference">
    <span class="lbl" title="${esc(r.id)}"><span class="mode ${esc(r.mode)}">${esc(r.mode)}</span> ${esc(r.preset)} ${esc(r.binary_short)} ${fmtE(r.total_energy_Ry)} ${r.pinned_reference ? '&#9733;' : ''} <span class="muted">${esc(r.id)}</span></span></div>`).join('')
    || '<div class="muted">no terminal runs yet</div>';
  list.querySelectorAll('.cmp-sel').forEach((i) => i.addEventListener('change', () => { if (i.checked) S.cmpSel.add(i.value); else S.cmpSel.delete(i.value); }));
  list.querySelectorAll('input[name=cmpref]').forEach((i) => i.addEventListener('change', () => { S.cmpRef = i.value; S.cmpSel.add(i.value); const cb = list.querySelector(`.cmp-sel[value="${i.value}"]`); if (cb) cb.checked = true; }));
}
async function runCompare() {
  const ids = Array.from(S.cmpSel); if (S.cmpRef && !ids.includes(S.cmpRef)) ids.unshift(S.cmpRef);
  if (!ids.length) { $('#cmp-msg').textContent = 'tick at least one run'; return; }
  let res; try { res = await api(`/api/compare?ids=${encodeURIComponent(ids.join(','))}${S.cmpRef ? '&ref=' + encodeURIComponent(S.cmpRef) : ''}`); }
  catch (e) { $('#cmp-msg').textContent = e.message; return; }
  S.cmpResult = res; $('#btn-copy-md').disabled = false; $('#btn-dl-json').disabled = false; $('#cmp-msg').textContent = `ref ${res.ref}`;
  renderCompareTable(res);
  const recs = await Promise.all(ids.map(async (id) => S.records[id] && TERMINAL.has(S.records[id].state) ? S.records[id] : (S.records[id] = await api(`/api/runs/${id}`))));
  renderPlots(res, recs);
}
function renderCompareTable(res) {
  const same = new Set((res.sections || {}).same_binary || []), cross = new Set((res.sections || {}).cross_build || []);
  const withFS = res.rows.some((r) => r.max_dF_Ry_bohr !== null || r.max_dstress_kbar !== null);
  const head = `<tr><th>run</th><th>binary</th><th>mode</th><th>E (Ry)</th><th>dE vs ref</th><th>tol</th><th>iters</th><th>QE wall</th><th>ratio</th><th>GEMM</th><th>GPU</th><th>GPU %</th><th>fb</th><th>lib build</th>${withFS ? '<th>max |dF|</th><th>max |ds|</th>' : ''}</tr>`;
  const row = (r) => {
    const tolCls = r.within_tolerance === null ? '' : (r.within_tolerance ? 'pass' : 'fail');
    const dE = r.is_ref ? 'ref' : (!r.same_input ? 'different input' : fmtSci(r.dE_vs_ref_Ry));
    const ratio = r.is_ref ? '-' : (r.wall_ratio_vs_ref === null ? `<span class="muted" title="${esc(r.wall_note || '')}">${esc(r.wall_note ? 'not comparable' : '-')}</span>` : (r.wall_comparable ? Number(r.wall_ratio_vs_ref).toFixed(3) : `<span class="struck" title="${esc(r.wall_note)}">${Number(r.wall_ratio_vs_ref).toFixed(3)}</span>`));
    const fbTitle = r.fallback_breakdown ? Object.entries(r.fallback_breakdown).map(([k, v]) => `${k}=${v}`).join(' ') : '';
    const gpu = r.mode === 'reference' ? 'n/a' : (r.gemm_calls === null ? 'unverifiable' : `${r.gpu_calls}/${r.gemm_calls}`);
    const gcls = r.mode === 'reference' ? 'na' : (r.gpu_verified === true ? (r.gpu_calls ? 'green' : 'grey') : (r.gpu_verified === false ? 'red' : 'amber'));
    return `<tr><td title="${esc(r.id)}">${esc(r.id.slice(0, 15))}${r.is_ref ? ' (ref)' : ''}</td><td>${esc(r.binary_short)}</td><td><span class="mode ${esc(r.mode)}">${esc(r.mode)}</span></td>
      <td class="num">${fmtE(r.total_energy_Ry)}</td><td class="num ${tolCls}">${dE}</td><td class="num ${tolCls}" title="${esc(r.tolerance_note)}">${r.is_ref ? '-' : fmtSci(r.tolerance_Ry)}</td>
      <td class="num">${r.scf_iterations ?? '-'}${r.iters_delta ? ` (${r.iters_delta > 0 ? '+' : ''}${r.iters_delta})` : ''}</td>
      <td class="num">${r.wall_qe_s === null ? '-' : Number(r.wall_qe_s).toFixed(2)}<span class="grade ${esc(r.timing_grade)}">${esc(r.timing_grade)}</span></td>
      <td class="num">${ratio}</td><td class="num">${r.mode === 'reference' ? 'n/a' : (r.gemm_calls ?? 'no counters')}</td><td class="gpu ${gcls}">${gpu}</td>
      <td class="num">${r.gpu_pct === null || r.gpu_pct === undefined ? '-' : Number(r.gpu_pct).toFixed(1)}</td><td class="num" title="${esc(fbTitle)}">${r.mode === 'reference' ? 'n/a' : (r.fallbacks ?? '-')}</td>
      <td class="small">${esc(r.lib_build || '-')}</td>${withFS ? `<td class="num ${r.dF_within === false ? 'fail' : ''}">${r.is_ref ? 'ref' : fmtSci(r.max_dF_Ry_bohr)}</td><td class="num ${r.dstress_within === false ? 'fail' : ''}">${r.is_ref ? 'ref' : (r.max_dstress_kbar === null ? '-' : Number(r.max_dstress_kbar).toFixed(3))}</td>` : ''}</tr>`;
  };
  const ncol = withFS ? 16 : 14;
  const refRow = res.rows.filter((r) => r.is_ref), sameRows = res.rows.filter((r) => same.has(r.id)), crossRows = res.rows.filter((r) => cross.has(r.id));
  $('#cmp-table').innerHTML = `<table>${head}${refRow.map(row).join('')}
    <tr class="section-row"><td colspan="${ncol}">BLAS A/B: same pw.x and same libapplebottom sha256, only AB_MODE differs (1e-8 Ry)${sameRows.length ? '' : ': none'}</td></tr>${sameRows.map(row).join('')}
    <tr class="section-row"><td colspan="${ncol}">Not the one-binary A/B: different pw.x (cross-build, 1e-5 Ry) or the same pw.x with another libapplebottom (library A/B, 1e-8 Ry); the tol tooltip names both builds${crossRows.length ? '' : ': none'}</td></tr>${crossRows.map(row).join('')}</table>`;
}

// ------------------------------------------------------------------ inline SVG plots
function svgOpen(w, h) { return `<svg viewBox="0 0 ${w} ${h}" xmlns="http://www.w3.org/2000/svg" role="img">`; }
function legend(items, x, y) {
  return `<g class="legend">${items.map((it, i) => `<rect x="${x}" y="${y + i * 12 - 7}" width="10" height="8" rx="2" fill="${it.color}"/><text x="${x + 14}" y="${y + i * 12}">${esc(it.label)}</text>`).join('')}</g>`;
}
function renderPlots(res, recs) {
  const W = 400, H = 180, L = 44, R = 110, T = 12, B = 28;
  const series = recs.map((rec, i) => ({rec, color: SERIES[i % SERIES.length], label: `${(rec.request || {}).mode} ${rec.id.slice(0, 15)}`}));
  // 1. SCF convergence: log10|E_i - E_final|
  let pts = [], ymin = 0, ymax = -12, xmax = 1;
  for (const s of series) {
    const es = ((s.rec.result || {}).scf_energies) || []; const ef = (s.rec.result || {}).total_energy_Ry ?? es[es.length - 1];
    const p = es.map((e, i) => [i + 1, Math.abs(e - ef)]).filter(([, d]) => d > 0).map(([i, d]) => [i, Math.log10(d)]);
    pts.push(p); for (const [x, y] of p) { ymin = Math.min(ymin, y); ymax = Math.max(ymax, y); xmax = Math.max(xmax, x); }
  }
  if (ymax <= ymin) ymax = ymin + 1;
  const sx = (x) => L + (x - 1) / Math.max(1, xmax - 1) * (W - L - R), sy = (y) => T + (ymax - y) / (ymax - ymin) * (H - T - B);
  let g = svgOpen(W, H);
  for (let k = Math.ceil(ymin); k <= Math.floor(ymax); k++) g += `<line class="grid" x1="${L}" x2="${W - R}" y1="${sy(k)}" y2="${sy(k)}"/><text x="${L - 4}" y="${sy(k) + 3}" text-anchor="end">${k}</text>`;
  g += `<line class="axis" x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}"/><line class="axis" x1="${L}" x2="${L}" y1="${T}" y2="${H - B}"/>`;
  for (let x = 1; x <= xmax; x += Math.max(1, Math.ceil(xmax / 10))) g += `<text x="${sx(x)}" y="${H - B + 12}" text-anchor="middle">${x}</text>`;
  g += `<text x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle">SCF iteration</text>`;
  series.forEach((s, i) => {
    const p = pts[i]; if (!p.length) return;
    g += `<polyline fill="none" stroke="${s.color}" stroke-width="2" stroke-linejoin="round" points="${p.map(([x, y]) => `${sx(x).toFixed(1)},${sy(y).toFixed(1)}`).join(' ')}"/>`;
    for (const [x, y] of p) g += `<circle class="pt" cx="${sx(x).toFixed(1)}" cy="${sy(y).toFixed(1)}" r="4" fill="${s.color}" stroke="var(--surface)" stroke-width="2"><title>${esc(s.label)}: iteration ${x}, log10|dE| = ${y.toFixed(2)}</title></circle>`;
  });
  g += legend(series, W - R + 6, T + 8) + '</svg>';
  $('#plot-scf').innerHTML = g;
  // 2. QE wall bars, hatched when untimed
  const walls = series.map((s) => ({s, w: (s.rec.result || {}).wall_qe_s, grade: (s.rec.etiquette || {}).timing_grade || 'untimed'}));
  const wmax = Math.max(1e-9, ...walls.map((w) => w.w || 0));
  const bh = Math.min(22, (H - T - B) / Math.max(1, walls.length) - 4);
  g = svgOpen(W, H) + `<defs><pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="6" stroke="var(--text-2)" stroke-width="2"/></pattern></defs>`;
  const barMax = W - L - R - 96;                       // room for the value label before the legend column
  walls.forEach((w, i) => {
    const y = T + i * (bh + 4); const bw = w.w ? (w.w / wmax) * barMax : 0;
    g += `<rect x="${L}" y="${y}" width="${bw.toFixed(1)}" height="${bh}" rx="3" fill="${w.grade === 'bench' ? w.s.color : 'url(#hatch)'}" stroke="${w.s.color}" stroke-width="1.5"><title>${esc(w.s.label)}: ${w.w ?? '-'} s, grade ${esc(w.grade)} (${esc((w.s.rec.etiquette || {}).timing_grade_reason || '')})</title></rect>`;
    g += `<text x="${L + bw + 4}" y="${y + bh / 2 + 3}">${w.w === null || w.w === undefined ? '-' : Number(w.w).toFixed(2)} s ${w.grade === 'bench' ? '' : '(untimed)'}</text>`;
  });
  g += `<line class="axis" x1="${L}" x2="${L}" y1="${T}" y2="${H - B}"/>` + legend(series, W - R + 6, T + 8) + '</svg>';
  $('#plot-wall').innerHTML = g;
  // 3. GEMM dim-bin histogram grouped per run
  const hists = series.map((s) => { const out = new Array(8).fill(0); for (const r of Object.values(((s.rec.ab || {}).routines) || {})) (r.dim_histogram || []).forEach((v, b) => { out[b] += v; }); return out; });
  const hmax = Math.max(1, ...hists.flat());
  const gw = (W - L - R) / 8, n = Math.max(1, series.length), bw2 = Math.max(2, (gw - 6) / n);
  g = svgOpen(W, H) + `<line class="axis" x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}"/>`;
  for (const f of [0.5, 1]) { const y = T + (1 - f) * (H - T - B); g += `<line class="grid" x1="${L}" x2="${W - R}" y1="${y}" y2="${y}"/><text x="${L - 4}" y="${y + 3}" text-anchor="end">${Math.round(hmax * f)}</text>`; }
  HIST_BINS.forEach((lbl, b) => {
    g += `<text x="${L + b * gw + gw / 2}" y="${H - B + 12}" text-anchor="middle">${lbl}</text>`;
    hists.forEach((h, i) => {
      const v = h[b]; const hh = v / hmax * (H - T - B); const x = L + b * gw + 3 + i * bw2;
      if (v > 0) g += `<rect x="${x.toFixed(1)}" y="${(H - B - hh).toFixed(1)}" width="${Math.max(1, bw2 - 1).toFixed(1)}" height="${hh.toFixed(1)}" rx="2" fill="${series[i].color}"><title>${esc(series[i].label)}: ${lbl}: ${v} calls</title></rect>`;
    });
  });
  g += `<text x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle">max(M,N,K) bin (apple-bottom profiler bins)</text>` + legend(series, W - R + 6, T + 8) + '</svg>';
  $('#plot-hist').innerHTML = g;
}

// ------------------------------------------------------------------ init
async function init() {
  try {
    const [health, modes, presets] = await Promise.all([api('/api/health'), api('/api/modes'), api('/api/presets')]);
    S.health = health; S.modes = modes.modes; S.presets = presets.presets;
    await loadBinaries();
  } catch (e) { msg(`cannot load the API: ${e.message}`, 'err'); return; }
  $('#version').textContent = 'v' + S.health.version;
  buildPresetSelect(); renderPresetCard(); buildBinarySelect(); renderBinaryCard(); buildModeRadios(); bindForm();
  await pollMachine(); setInterval(pollMachine, 5000);
  await refreshRuns();
  openGlobalEvents();
  schedulePlan();
}
document.addEventListener('DOMContentLoaded', init);
})();
