/* Chunk Monitor frontend: vanilla JS, SSE, hand-drawn SVG charts. No build step, no storage APIs. */
(() => {
  'use strict';
  const RAW = window.__API_BASE__ || '';
  const API = RAW.startsWith('__') ? '' : RAW.replace(/\/$/, '');
  const $ = (s, el = document) => el.querySelector(s);
  const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const NS = 'http://www.w3.org/2000/svg';

  const STRATEGY_INFO = {
    fixed: 'Fixed-size', recursive: 'Recursive', document: 'Document', semantic: 'Semantic',
    propositional: 'Propositional', agentic: 'Agentic',
  };
  const LLM_STRATS = new Set(['agentic', 'propositional']);
  const ICON = {
    ok: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="m8 12 3 3 5-6"/></svg>',
    degraded: '<svg viewBox="0 0 24 24"><path d="M12 3 2 20h20L12 3z"/><path d="M12 10v4M12 17h.01"/></svg>',
    down: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="m9 9 6 6M15 9l-6 6"/></svg>',
    off: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M8 12h8"/></svg>',
    done: '<svg viewBox="0 0 24 24"><path d="m5 12 5 5 9-10"/></svg>',
    error: '<svg viewBox="0 0 24 24"><path d="m6 6 12 12M18 6 6 18"/></svg>',
    queued: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="8"/></svg>',
    cancelled: '<svg viewBox="0 0 24 24"><path d="M6 12h12"/></svg>',
  };
  const STATUS_LABEL = { ok: 'Healthy', degraded: 'Degraded', down: 'Down', off: 'Off', unknown: 'Connecting' };

  const S = {
    config: null, settings: { embed: 'hashing', llm: 'none' }, health: null, metrics: null,
    source: 'redis-guide', selected: new Set(['recursive', 'document', 'semantic', 'agentic']),
    run: null, tab: null, pinned: false, logCalls: true, theme: 'dark',
  };

  // ---------------------------------------------------------------- formatting
  const fmtMs = (ms) => ms == null ? '–' : ms < 1 ? `${ms.toFixed(2)} ms` : ms < 1000 ? `${ms.toFixed(ms < 10 ? 1 : 0)} ms` : `${(ms / 1000).toFixed(ms < 10000 ? 2 : 1)} s`;
  const fmtPct = (x, d = 1) => x == null ? '–' : `${(x * 100).toFixed(d)}%`;
  const fmtBytes = (b) => b == null ? '–' : b < 1024 ? `${b} B` : b < 1048576 ? `${(b / 1024).toFixed(1)} KB` : b < 1073741824 ? `${(b / 1048576).toFixed(1)} MB` : `${(b / 1073741824).toFixed(2)} GB`;
  const fmtDur = (s) => { s = Math.max(0, Math.round(s)); if (s < 60) return `${s}s`; if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`; return `${Math.floor(s / 3600)}h ${Math.floor(s % 3600 / 60)}m`; };
  const clock = (ts) => new Date(ts * 1000).toLocaleTimeString([], { hour12: false });
  const num = (n) => n == null ? '–' : Number(n).toLocaleString();

  // ---------------------------------------------------------------- api
  async function api(path, opts = {}) {
    const res = await fetch(`${API}${path}`, { headers: { 'content-type': 'application/json' }, ...opts });
    let body = null;
    try { body = await res.json(); } catch { /* empty */ }
    if (!res.ok) throw new Error(body?.detail ? (typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)) : `HTTP ${res.status}`);
    return body;
  }
  let toastTimer;
  function toast(msg, isErr = false) {
    const t = $('#toast');
    t.textContent = msg; t.className = `toast${isErr ? ' err' : ''}`; t.hidden = false;
    clearTimeout(toastTimer); toastTimer = setTimeout(() => { t.hidden = true; }, 3800);
  }

  // ---------------------------------------------------------------- charts
  function svgEl(tag, attrs = {}, parent) {
    const el = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
    if (parent) parent.appendChild(el);
    return el;
  }
  function niceMax(v) {
    if (!v || v <= 0) return 1;
    const p = 10 ** Math.floor(Math.log10(v)); const f = v / p;
    return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * p;
  }
  function pathFor(points) {
    let d = '', pen = false;
    for (const p of points) {
      if (p == null) { pen = false; continue; }
      d += `${pen ? 'L' : 'M'}${p[0].toFixed(1)},${p[1].toFixed(1)}`; pen = true;
    }
    return d;
  }
  /** Time-series latency chart. series: [{values:[number|null], color, width}] */
  function lineChart(host, series, { fmt = fmtMs, emptyText = 'No calls yet' } = {}) {
    const w = host.clientWidth || 300, h = host.clientHeight || 90, padL = 44, padB = 14, padT = 4;
    const svg = svgEl('svg', { viewBox: `0 0 ${w} ${h}`, preserveAspectRatio: 'none', role: 'img' });
    const all = series.flatMap((s) => s.values.filter((v) => v != null));
    const max = niceMax(Math.max(...all, 0) * 1.1);
    const ih = h - padB - padT, iw = w - padL - 4;
    [0, 0.5, 1].forEach((f) => {
      const y = padT + ih * (1 - f);
      svgEl('line', { x1: padL, x2: w, y1: y, y2: y, class: 'axis', 'stroke-dasharray': f ? '2 3' : '' }, svg);
      const t = svgEl('text', { x: padL - 6, y: y + 3, 'text-anchor': 'end' }, svg); t.textContent = all.length ? (f ? fmt(max * f) : '0') : '';
    });
    const n = series[0]?.values.length || 1;
    const lbl = svgEl('text', { x: padL, y: h - 2 }, svg); lbl.textContent = '−10m';
    const lbr = svgEl('text', { x: w - 4, y: h - 2, 'text-anchor': 'end' }, svg); lbr.textContent = 'now';
    if (!all.length) {
      const t = svgEl('text', { x: padL + iw / 2, y: padT + ih / 2 + 4, 'text-anchor': 'middle', class: 'nodata' }, svg); t.textContent = emptyText;
    }
    for (const s of series) {
      const pts = s.values.map((v, i) => v == null ? null : [padL + iw * (i / Math.max(1, n - 1)), padT + ih * (1 - v / max)]);
      svgEl('path', { d: pathFor(pts), fill: 'none', stroke: s.color, 'stroke-width': s.width || 1.5, 'vector-effect': 'non-scaling-stroke', 'stroke-linejoin': 'round' }, svg);
      // isolated points (single-bucket bursts) would be invisible as a path
      pts.forEach((p, i) => { if (p && !pts[i - 1] && !pts[i + 1]) svgEl('circle', { cx: p[0], cy: p[1], r: 2, fill: s.color }, svg); });
    }
    host.replaceChildren(svg);
  }
  function sparkline(values, color) {
    const w = 92, h = 22, vals = values.filter((v) => v != null);
    const svg = svgEl('svg', { viewBox: `0 0 ${w} ${h}`, class: 'spark', preserveAspectRatio: 'none', 'aria-hidden': 'true' });
    if (vals.length < 2) return svg;
    const max = Math.max(...vals) * 1.15 || 1, min = 0;
    const pts = values.map((v, i) => v == null ? null : [w * i / (values.length - 1), h - 2 - (h - 4) * ((v - min) / (max - min))]);
    svgEl('path', { d: pathFor(pts), fill: 'none', stroke: color, 'stroke-width': 1.25, 'vector-effect': 'non-scaling-stroke' }, svg);
    const last = pts.filter(Boolean).pop();
    if (last) svgEl('circle', { cx: last[0], cy: last[1], r: 1.8, fill: color }, svg);
    return svg;
  }
  /** Semantic distance curve with percentile threshold, cut spikes highlighted (like the reel). */
  function distanceChart(host, ev, cuts) {
    const w = host.clientWidth || 280, h = 150, padL = 30, padB = 16, padT = 8;
    const d = ev.distances, n = d.length;
    const svg = svgEl('svg', { viewBox: `0 0 ${w} ${h}`, role: 'img', 'aria-label': 'Distance between neighbouring sentence windows' });
    const max = Math.max(...d, ev.threshold, 0.01) * 1.08;
    const iw = w - padL - 6, ih = h - padT - padB;
    const X = (i) => padL + (n === 1 ? iw / 2 : iw * i / (n - 1)), Y = (v) => padT + ih * (1 - v / max);
    [0, max / 2, max].forEach((v) => {
      svgEl('line', { x1: padL, x2: w - 6, y1: Y(v), y2: Y(v), class: 'axis', 'stroke-dasharray': v ? '2 3' : '' }, svg);
      const t = svgEl('text', { x: padL - 5, y: Y(v) + 3, 'text-anchor': 'end' }, svg); t.textContent = v.toFixed(2);
    });
    const area = `M${X(0)},${Y(0)} ` + d.map((v, i) => `L${X(i)},${Y(v)}`).join(' ') + ` L${X(n - 1)},${Y(0)} Z`;
    svgEl('path', { d: area, fill: 'var(--accent-soft)', stroke: 'none' }, svg);
    svgEl('path', { d: d.map((v, i) => `${i ? 'L' : 'M'}${X(i)},${Y(v)}`).join(' '), fill: 'none', stroke: 'var(--accent)', 'stroke-width': 1.6 }, svg);
    svgEl('line', { x1: padL, x2: w - 6, y1: Y(ev.threshold), y2: Y(ev.threshold), stroke: 'var(--c2b)', 'stroke-width': 1.2, 'stroke-dasharray': '5 4' }, svg);
    const tl = svgEl('text', { x: w - 8, y: Y(ev.threshold) - 4, 'text-anchor': 'end' }, svg); tl.textContent = `threshold ${ev.threshold.toFixed(2)}`;
    d.forEach((v, i) => {
      const isCut = cuts ? cuts.has(ev.gaps[i]) : v >= ev.threshold;
      const c = svgEl('circle', { cx: X(i), cy: Y(v), r: isCut ? 3.5 : 2, fill: isCut ? 'var(--cut)' : 'var(--surface-2)', stroke: isCut ? 'var(--cut)' : 'var(--accent)', 'stroke-width': 1.2 }, svg);
      const tt = svgEl('title', {}, c); tt.textContent = `gap ${i + 1}: distance ${v.toFixed(3)}${isCut ? ' (cut)' : ''}`;
    });
    const xl = svgEl('text', { x: padL, y: h - 3 }, svg); xl.textContent = 'sentence gaps →';
    host.replaceChildren(svg);
  }

  // ---------------------------------------------------------------- setup
  async function init() {
    bindUI();
    try {
      S.config = await api('/api/config');
    } catch (e) {
      toast(`Could not reach the server: ${e.message}`, true);
      setTimeout(init, 3000);
      return;
    }
    S.settings = S.config.settings;
    buildSources(); buildChips(); buildSelects();
    if (S.config.limits?.public) { ['#sel-embed', '#sel-llm'].forEach((id) => { $(id).disabled = true; $(id).title = 'Backends are fixed on this public demo'; }); $('#btn-apply').hidden = true; }
    loadSample(S.source);
    connect();
  }

  function bindUI() {
    $('#btn-run').addEventListener('click', startRun);
    $('#btn-cancel').addEventListener('click', cancelRun);
    $('#btn-apply').addEventListener('click', applySettings);
    $('#btn-probe').addEventListener('click', async () => { try { await api('/api/probe', { method: 'POST' }); toast('Probing backends…'); } catch (e) { toast(e.message, true); } });
    $('#btn-theme').addEventListener('click', () => {
      S.theme = S.theme === 'dark' ? 'light' : 'dark';
      document.documentElement.className = S.theme;
      if (S.metrics) renderMetrics(S.metrics);
      if (S.run) renderLive();
    });
    ['#sel-embed', '#sel-llm'].forEach((id) => $(id).addEventListener('change', syncApply));
    $('#log-calls').addEventListener('change', (e) => { S.logCalls = e.target.checked; });
    $('#doc').addEventListener('input', () => { if (S.source !== 'custom') { S.source = 'custom'; buildSources(); } });
    window.addEventListener('resize', debounce(() => { if (S.metrics) renderMetrics(S.metrics); renderSidePanel(); }, 150));
    setInterval(tickCountdowns, 1000);
  }
  const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };

  function buildSources() {
    const seg = $('#source-seg');
    const items = [...Object.entries(S.config.samples).map(([k, v]) => [k, v.label.replace(/ \(.*\)/, '')]), ['custom', 'Custom']];
    seg.innerHTML = items.map(([k, l]) => `<button role="tab" data-src="${k}" aria-selected="${S.source === k}" data-testid="tab-source-${k}">${esc(l)}</button>`).join('');
    seg.querySelectorAll('button').forEach((b) => b.addEventListener('click', () => {
      S.source = b.dataset.src;
      if (S.source !== 'custom') loadSample(S.source); else $('#doc').focus();
      buildSources();
    }));
  }
  function loadSample(key) { const s = S.config.samples[key]; if (s) $('#doc').value = s.text; }

  function buildChips() {
    const host = $('#strategy-chips');
    host.innerHTML = S.config.strategies.map((s, i) => `<button class="chip" data-s="${s}" aria-pressed="${S.selected.has(s)}" data-testid="chip-strategy-${s}">
      <span class="n">${i + 1}</span>${STRATEGY_INFO[s] || s}${LLM_STRATS.has(s) ? '<span class="llm-tag">LLM</span>' : ''}</button>`).join('');
    host.querySelectorAll('.chip').forEach((c) => c.addEventListener('click', () => {
      const s = c.dataset.s;
      S.selected.has(s) ? S.selected.delete(s) : S.selected.add(s);
      c.setAttribute('aria-pressed', S.selected.has(s));
      updateHint();
    }));
    updateHint();
  }
  function updateHint() {
    const llm = S.settings.llm;
    const usesLLM = [...S.selected].some((s) => LLM_STRATS.has(s));
    let hint = '';
    if (usesLLM && llm === 'none') hint = 'Agentic and propositional run on offline heuristics. Pick an LLM above to use a real model.';
    else if (usesLLM) hint = `Agentic and propositional call ${llm} once per sentence or passage, so they can take a while on CPU.`;
    if ($('#in-stability').checked && usesLLM && llm !== 'none') hint += ' Stability check reruns each strategy twice.';
    const c = S.config;
    if (c?.limits && S.settings.embed.startsWith('deepinfra')) {
      hint = `Public demo on hosted models (DeepInfra): ${S.settings.embed.split(':')[1]} for embeddings and ${S.settings.llm.split(':')[1] || 'no LLM'} for agentic/propositional. Documents up to ${c.limits.max_chars.toLocaleString()} characters.`;
    } else if (c?.limits && !c.available?.ollama_running && S.settings.embed === 'hashing') {
      hint = `Public demo: runs on the offline hashing embedder and rule-based heuristics (no local models on this server). Documents up to ${c.limits.max_chars.toLocaleString()} characters. Run it locally with Ollama for real models.`;
    }
    if (c?.limits?.public) hint += ' Runs are visible to everyone viewing this page, so don\'t paste private text.';
    $('#run-hint').textContent = hint;
  }

  function buildSelects() {
    const av = S.config.available;
    const fill = (sel, opts, cur) => {
      const list = opts.some((o) => o.spec === cur) ? opts : [...opts, { spec: cur, label: cur }];
      sel.innerHTML = list.map((o) => `<option value="${esc(o.spec)}" ${o.spec === cur ? 'selected' : ''}>${esc(o.label)}</option>`).join('');
    };
    fill($('#sel-embed'), av.embed, S.settings.embed);
    fill($('#sel-llm'), av.llm, S.settings.llm);
    syncApply();
  }
  function syncApply() {
    const changed = $('#sel-embed').value !== S.settings.embed || $('#sel-llm').value !== S.settings.llm;
    $('#btn-apply').disabled = !changed;
    $('#btn-apply').classList.toggle('btn-primary', changed);
  }
  async function applySettings() {
    const btn = $('#btn-apply'); btn.disabled = true; btn.textContent = 'Applying…';
    try {
      S.settings = await api('/api/settings', { method: 'POST', body: JSON.stringify({ embed: $('#sel-embed').value, llm: $('#sel-llm').value }) });
      toast(`Using ${S.settings.embed} + ${S.settings.llm}`);
      addLog({ ts: Date.now() / 1000, text: `settings: embed=${S.settings.embed} llm=${S.settings.llm}`, cls: 'good' });
    } catch (e) { toast(e.message, true); }
    btn.textContent = 'Apply'; syncApply(); updateHint();
  }

  // ---------------------------------------------------------------- SSE
  let es;
  function setConn(state, label) { const c = $('#conn'); c.dataset.state = state; $('.conn-label', c).textContent = label; }
  function connect() {
    if (es) es.close();
    es = new EventSource(`${API}/api/stream`);
    setConn('connecting', 'Connecting');
    es.onopen = () => setConn('live', 'Live');
    es.onerror = () => { setConn('lost', 'Reconnecting'); };
    const on = (type, fn) => es.addEventListener(type, (m) => { try { fn(JSON.parse(m.data)); } catch (err) { console.error(type, err); } });
    on('hello', (d) => {
      setConn('live', 'Live');
      if (d.settings) { S.settings = d.settings; buildSelects(); }
      if (d.health && d.health.components) renderHealth(d.health);
      if (d.metrics) renderMetrics(d.metrics);
      const cur = d.worker?.current;
      if (cur && (!S.run || S.run.id !== cur.id)) resumeRun(cur.id);
    });
    on('health', renderHealth);
    on('metrics', renderMetrics);
    on('settings', (d) => { S.settings = d; buildSelects(); updateHint(); });
    on('run.queued', (d) => { addLog({ ts: d.created, text: `${d.id} queued: ${d.strategies.join(', ')}` }); if (!S.run || S.run.id === d.id) markRun(d); });
    on('run.started', (d) => {
      addLog({ ts: Date.now() / 1000, text: `${d.id} started with ${d.embed} / ${d.llm}` });
      if (!S.run || S.run.id !== d.id) newRunState(d, d.text);
      markRun(d);
    });
    on('run.finished', (d) => {
      const cls = d.status === 'done' ? 'good' : d.status === 'error' ? 'err' : 'warn';
      addLog({ ts: d.finished || Date.now() / 1000, text: `${d.id} ${d.status}${d.error ? `: ${d.error}` : ''}`, cls });
      if (S.run?.id === d.id) { markRun(d); renderCompare(); }
    });
    on('strategy.started', (d) => {
      const r = runFor(d.run); if (!r) return;
      r.progress = { index: d.index, total: d.total, strategy: d.strategy };
      Object.assign(r.results[d.strategy] ||= blankResult(), { status: 'running', t0: performance.now() });
      if (!S.pinned) S.tab = d.strategy;
      renderTabs(); renderLive(); renderMeta(); renderCompare();
    });
    on('strategy.finished', (d) => {
      const r = runFor(d.run); if (!r) return;
      const res = r.results[d.strategy] ||= blankResult();
      Object.assign(res, { status: d.status, chunks: d.chunks, stats: d.stats, seconds: d.seconds, error: d.error, stability: d.stability, trial: null });
      addLog({ ts: Date.now() / 1000, text: `${d.strategy}: ${d.status === 'done' ? `${d.stats.chunks} chunks in ${fmtMs(d.seconds * 1000)}` : d.status}${d.error ? ` (${d.error})` : ''}${d.stability ? `, stability ${fmtPct(d.stability.score, 0)}` : ''}`, cls: d.status === 'error' ? 'err' : '' });
      renderTabs(); if (S.tab === d.strategy) renderLive(); renderCompare();
    });
    on('stability.trial', (d) => { const r = runFor(d.run); if (!r) return; const res = r.results[d.strategy]; if (res) res.trial = d.size; if (S.tab === d.strategy) renderStats(); });
    on('agentic.start', (d) => { const res = liveRes(d, 'agentic'); if (!res) return; Object.assign(res, { spans: d.spans, decisions: [], mode: d.mode }); if (S.tab === 'agentic') renderLive(); });
    on('agentic.decision', (d) => { const res = liveRes(d, 'agentic'); if (!res) return; (res.decisions ||= []).push(d); if (S.tab === 'agentic') { if (res.spans) applyDecision(res, d); else { appendFeed(d); renderStats(); } } });
    on('semantic.distances', (d) => { const res = liveRes(d, 'semantic'); if (!res) return; res.distances = d; if (S.tab === 'semantic') renderLive(); });
    on('propositional.passage', (d) => { const res = liveRes(d, 'propositional'); if (!res) return; (res.passages ||= []).push(d); if (S.tab === 'propositional') renderLive(); });
    on('llm.retry', (d) => addLog({ ts: d.ts, text: `retry ${d.backend}: ${d.reason}`, cls: 'warn' }));
    on('llm.json_failure', (d) => addLog({ ts: d.ts, text: `json failure ${d.backend}: ${d.reason}`, cls: 'err' }));
    on('fallback', (d) => addLog({ ts: d.ts, text: `fallback ${d.strategy}: ${d.reason}`, cls: 'warn' }));
    on('call', (d) => {
      const r = d.run && runFor(d.run);
      if (r && !d.trial) { const res = r.results[r.progress?.strategy]; if (res && !res.trial) { res.calls = (res.calls || 0) + 1; if (S.tab === r.progress.strategy) renderStats(); } }
      if (!d.ok) addLog({ ts: d.ts, text: `${d.type} ${d.backend} failed: ${d.error}`, cls: 'err' });
      else if (S.logCalls) addLog({ ts: d.ts, text: `${d.type === 'llm.call' ? 'llm' : 'embed'} ${d.backend}${d.n ? ` ×${d.n}` : ''} ${fmtMs(d.ms)}${d.tokens_per_s ? ` · ${d.tokens_per_s} tok/s` : ''}` });
    });
  }

  // ---------------------------------------------------------------- runs
  const blankResult = () => ({ status: 'queued' });
  const runFor = (id) => (S.run && S.run.id === id ? S.run : null);
  function liveRes(ev, name) { const r = runFor(ev.run); return r ? (r.results[name] ||= blankResult()) : null; }
  function newRunState(summary, text, own = false) {
    S.run = { id: summary.id, text, strategies: summary.strategies, status: summary.status, summary, results: {}, progress: null, own };
    summary.strategies.forEach((s) => { S.run.results[s] = blankResult(); });
    S.tab = summary.strategies[0]; S.pinned = false;
    renderTabs(); renderLive(); renderCompare(); renderMeta();
  }
  function markRun(summary) {
    if (!S.run || S.run.id !== summary.id) return;
    const ORDER = { queued: 0, running: 1, done: 2, error: 2, cancelled: 2 };
    if ((ORDER[summary.status] ?? 0) < (ORDER[S.run.status] ?? 0)) return; // stale (out-of-order) update
    Object.assign(S.run, { status: summary.status, summary, progress: summary.progress || S.run.progress });
    $('#btn-run').disabled = false;
    renderMeta();
    if (!['queued', 'running'].includes(summary.status)) renderTabs();
  }
  async function resumeRun(id) {
    try {
      const d = await api(`/api/runs/${id}`);
      newRunState(d, d.text);
      Object.entries(d.live || {}).forEach(([k, v]) => Object.assign(S.run.results[k] ||= blankResult(), v));
      Object.entries(d.results).forEach(([k, v]) => Object.assign(S.run.results[k] ||= blankResult(), v));
      if (d.progress) S.tab = d.progress.strategy;
      markRun(d); renderTabs(); renderLive(); renderCompare();
    } catch { /* run expired */ }
  }
  async function startRun() {
    const strategies = S.config.strategies.filter((s) => S.selected.has(s));
    if (!strategies.length) { toast('Pick at least one strategy.', true); return; }
    const text = $('#doc').value;
    if (!text.trim()) { toast('The document is empty.', true); return; }
    const body = { text, strategies, size: +$('#in-size').value || 80, overlap: +$('#in-overlap').value || 0, stability: $('#in-stability').checked };
    $('#btn-run').disabled = true;
    try {
      const summary = await api('/api/runs', { method: 'POST', body: JSON.stringify(body) });
      // the SSE 'run.started' event can beat this response; never reset a run already in progress
      if (!S.run || S.run.id !== summary.id) newRunState(summary, text, true);
      else S.run.own = true;
      S.run.token = summary.cancel_token;
      markRun(summary);
      renderMeta(); // markRun skips stale statuses, but ownership changed
    } catch (e) { toast(e.message, true); $('#btn-run').disabled = false; }
  }
  async function cancelRun() {
    if (!S.run) return;
    try { await api(`/api/runs/${S.run.id}/cancel`, { method: 'POST', body: JSON.stringify({ token: S.run.token || '' }) }); toast('Cancelling after the current model call…'); } catch (e) { toast(e.message, true); }
  }

  function renderMeta() {
    const r = S.run, el = $('#run-meta');
    if (!r) { el.textContent = 'No run yet'; return; }
    const sm = r.summary || {};
    const prog = r.progress ? ` · ${r.progress.index + 1}/${r.progress.total} ${r.progress.strategy}` : '';
    el.textContent = `${r.own ? '' : 'Watching (started elsewhere) · '}${r.id} · ${r.status}${r.status === 'running' ? prog : ''} · ${sm.embed || ''} / ${sm.llm || ''}`;
    el.classList.toggle('watching', !r.own);
    $('#btn-cancel').hidden = !(r.own && ['queued', 'running'].includes(r.status));
  }
  function renderTabs() {
    const r = S.run, host = $('#tabs');
    if (!r) { host.innerHTML = ''; return; }
    host.innerHTML = r.strategies.map((s) => {
      const res = r.results[s] || {};
      const st = res.status || 'queued';
      const icon = st === 'running' ? '<span class="spin"></span>' : ICON[st] || ICON.queued;
      const count = res.stats ? `<span class="count">${res.stats.chunks}</span>` : '';
      return `<button class="tab" role="tab" data-s="${s}" aria-selected="${S.tab === s}" data-testid="tab-strategy-${s}">
        <span class="st st-${st}">${icon}</span>${STRATEGY_INFO[s] || s}${count}</button>`;
    }).join('');
    host.querySelectorAll('.tab').forEach((t) => t.addEventListener('click', () => { S.tab = t.dataset.s; S.pinned = true; renderTabs(); renderLive(); }));
  }

  // ---------------------------------------------------------------- live view
  function renderStats() {
    const r = S.run, host = $('#stat-strip');
    const res = r && r.results[S.tab];
    const st = res?.stats;
    const cell = (k, v, id) => `<div class="stat" data-testid="stat-${id}"><div class="k">${k}</div><div class="v">${v}</div></div>`;
    const running = res?.status === 'running';
    const elapsed = running && res.t0 ? `${((performance.now() - res.t0) / 1000).toFixed(1)}<small> s</small>` : res?.seconds != null ? fmtMs(res.seconds * 1000).replace(/ (m?s)$/, '<small> $1</small>') : '–';
    let stab = '–';
    if (res?.stability) { const sc = res.stability.score; stab = `<span class="${sc >= 0.8 ? 'good' : sc >= 0.5 ? 'meh' : 'bad'}">${fmtPct(sc, 0)}</span>`; }
    else if (res?.trial) stab = `<small>trial ${res.trial}</small>`;
    const fifth = res?.stability || res?.trial ? cell('Stability', stab, 'stability')
      : cell('Model calls', res?.calls ?? (res && res.status !== 'queued' ? 0 : '–'), 'calls');
    host.innerHTML = [
      cell('Chunks', st ? st.chunks : running && res.decisions ? res.decisions.filter((d) => d.boundary).length + 1 : '–', 'chunks'),
      cell('Avg tokens', st ? st.avg_tokens : '–', 'avg-tokens'),
      cell('Min – max', st ? `${st.min_tokens}<small>–</small>${st.max_tokens}` : '–', 'minmax'),
      cell('Mid-sentence cuts', st ? `<span class="${st.mid_sentence_cuts ? 'bad' : 'good'}">${st.mid_sentence_cuts}</span>` : '–', 'cuts'),
      fifth,
      cell(running ? 'Elapsed' : 'Time', elapsed, 'time'),
    ].join('');
  }

  function isSentenceEnd(text, i) {
    const before = text.slice(Math.max(0, i - 3), i).trimEnd();
    return !before || /[.!?:;|`)\]]$/.test(before) || /\n\s*$/.test(text.slice(Math.max(0, i - 2), i)) || /^\s*\n/.test(text.slice(i, i + 2));
  }
  /** Colour each chunk's span and drop a cut marker at every boundary. */
  function chunksHTML(text, chunks) {
    const real = chunks.filter((c) => c.start >= 0).sort((a, b) => a.start - b.start || a.end - b.end);
    const uniq = []; const seen = new Set();
    for (const c of real) { const k = `${c.start}:${c.end}`; if (!seen.has(k)) { seen.add(k); uniq.push(c); } }
    let html = '', pos = 0;
    uniq.forEach((c, i) => {
      const a = Math.max(c.start, pos), b = Math.max(c.end, a);
      if (a > pos) html += esc(text.slice(pos, a));
      if (i > 0) html += `<span class="cut${isSentenceEnd(text, a) ? '' : ' mid'}" title="boundary ${i}${c.meta?.cut_reason ? `: ${esc(c.meta.cut_reason)}` : c.meta?.opened_by ? `: ${esc(c.meta.opened_by)}` : ''}"></span>`;
      const tip = `chunk ${i + 1} · ${c.tokens} tokens${c.meta?.title ? ` · ${c.meta.title}` : ''}${c.meta?.headings?.length ? ` · ${[].concat(c.meta.headings).join(' > ')}` : ''}`;
      html += `<span class="seg-c k${i % 6}" title="${esc(tip)}" data-testid="chunk-${i}">${esc(text.slice(a, b))}</span>`;
      pos = b;
    });
    return html + esc(text.slice(pos));
  }
  function sentencesHTML(text, spans) {
    let html = '', pos = 0;
    spans.forEach(([a, b], i) => {
      if (a > pos) html += esc(text.slice(pos, a));
      html += `<span class="sent pending" data-i="${i}">${esc(text.slice(a, b))}</span>`;
      pos = b;
    });
    return html + esc(text.slice(pos));
  }

  function renderLive() {
    renderStats();
    const r = S.run, doc = $('#docmap');
    const res = r && r.results[S.tab];
    $('.live-body').classList.toggle('wide', !['semantic', 'agentic', 'propositional'].includes(S.tab));
    if (!r || !res) { doc.innerHTML = '<p class="empty">Run a strategy to watch the boundaries appear.</p>'; $('#side-panel').innerHTML = ''; return; }
    if (res.status === 'error') {
      doc.innerHTML = `<p class="empty bad">${esc(res.error || 'failed')}</p>`;
    } else if (res.chunks) {
      doc.innerHTML = chunksHTML(r.text, res.chunks);
    } else if (S.tab === 'agentic' && res.spans) {
      doc.innerHTML = sentencesHTML(r.text, res.spans);
      res.k = 0;
      const first = doc.querySelector('.sent[data-i="0"]');
      if (first) { first.classList.remove('pending'); first.classList.add('seg-c', 'k0'); }
      (res.decisions ||= []).forEach((d) => applyDecision(res, d, true));
      markInspecting(res);
    } else if (S.tab === 'propositional' && res.passages?.length) {
      const fake = res.passages.map((p) => ({ start: p.start, end: p.end, tokens: p.propositions.length, meta: { title: `${p.propositions.length} facts` } }));
      doc.innerHTML = chunksHTML(r.text, fake);
    } else {
      doc.innerHTML = esc(r.text) + (res.status === 'running' ? '' : '');
    }
    renderSidePanel();
  }

  function markInspecting(res) {
    const doc = $('#docmap');
    doc.querySelectorAll('.sent.inspecting').forEach((e) => e.classList.remove('inspecting'));
    if (res.status !== 'running') return;
    const next = res.decisions.length ? res.decisions[res.decisions.length - 1].index + 1 : 1;
    const el = doc.querySelector(`.sent[data-i="${next}"]`);
    if (el) { el.classList.add('inspecting'); if (!isVisible(el, doc)) el.scrollIntoView({ block: 'nearest', behavior: 'smooth' }); }
  }
  const isVisible = (el, box) => { const a = el.getBoundingClientRect(), b = box.getBoundingClientRect(); return a.top >= b.top && a.bottom <= b.bottom; };

  function applyDecision(res, d, silent = false) {
    const doc = $('#docmap');
    const el = doc.querySelector(`.sent[data-i="${d.index}"]`);
    if (el && !res.chunks) {
      if (d.boundary) { res.k = (res.k || 0) + 1; el.insertAdjacentHTML('beforebegin', `<span class="cut" title="${esc(d.reason)}"></span>`); }
      el.classList.remove('pending');
      el.classList.add('seg-c', `k${(res.k || 0) % 6}`);
      el.title = `${d.boundary ? 'new chunk' : 'continues'} · ${d.reason}${d.title ? ` · ${d.title}` : ''}`;
    }
    if (!silent) { markInspecting(res); renderStats(); appendFeed(d); flashSteps(d); }
  }

  function renderSidePanel() {
    const host = $('#side-panel'), r = S.run;
    const res = r && r.results[S.tab];
    if (!res) { host.innerHTML = ''; return; }
    if (S.tab === 'semantic') {
      host.innerHTML = `<div class="panel"><h3>Distance between windows <span>${res.distances ? `${res.distances.distances.length} gaps` : ''}</span></h3>
        <div id="dist-chart" data-testid="chart-semantic-distance">${res.distances ? '' : '<p class="empty">Waiting for embeddings…</p>'}</div></div>
        <div class="panel"><h3>Cuts</h3><ol class="feed" id="sem-cuts" data-testid="list-semantic-cuts"></ol></div>`;
      if (res.distances) {
        const cuts = res.chunks ? new Set(res.chunks.filter((c) => c.meta?.cut_reason === 'distance spike').map((c) => c.end)) : null;
        requestAnimationFrame(() => distanceChart($('#dist-chart'), res.distances, cuts && cuts.size ? nearestGaps(res.distances.gaps, cuts) : null));
      }
      const list = (res.chunks || []).filter((c) => c.meta?.cut_reason).slice(0, 40);
      $('#sem-cuts').innerHTML = list.length ? list.map((c, i) => `<li class="${c.meta.cut_reason === 'distance spike' ? 'b' : ''}"><span class="i">${i + 1}</span><span class="t"><strong>${esc(c.meta.cut_reason)}</strong>${c.meta.cut_distance != null ? ` · d=${c.meta.cut_distance}` : ''}</span><span class="ms">${c.tokens} tok</span></li>`).join('') : '<li><span></span><span class="t">Cuts appear when the run finishes.</span></li>';
    } else if (S.tab === 'agentic') {
      host.innerHTML = `<div class="panel"><h3>Agent loop <span>${esc(res.mode || '')}</span></h3>
        <div class="steps" id="steps"><div class="step" data-k="inspect"><b>01</b>Inspect</div><div class="step" data-k="ctx"><b>02</b>Contextualize</div><div class="step" data-k="bound"><b>03</b>Boundary</div></div>
        <ol class="feed" id="feed" data-testid="list-agentic-decisions"></ol></div>`;
      (res.decisions || []).slice(-150).forEach((d) => appendFeed(d, true));
      if (!res.decisions?.length) $('#feed').innerHTML = '<li><span></span><span class="t">Decisions stream here, one per sentence.</span></li>';
    } else if (S.tab === 'propositional') {
      const facts = res.chunks ? res.chunks.map((c) => ({ text: c.text, k: 0 })) : (res.passages || []).flatMap((p, i) => p.propositions.map((t) => ({ text: t, k: i })));
      host.innerHTML = `<div class="panel"><h3>Propositions <span>${facts.length}</span></h3><ol class="feed facts" data-testid="list-propositions">${facts.length ? facts.map((f, i) => `<li><span class="dot" style="background:var(--c${(res.chunks ? passageIndex(res, i) : f.k) % 6}b)"></span><span class="t" title="${esc(f.text)}">${esc(f.text)}</span></li>`).join('') : '<li><span></span><span class="t">Facts appear passage by passage.</span></li>'}</ol></div>`;
    } else host.innerHTML = '';
  }
  function passageIndex(res, i) {
    const c = res.chunks[i]; const spans = [...new Set(res.chunks.map((x) => `${x.start}:${x.end}`))];
    return spans.indexOf(`${c.start}:${c.end}`);
  }
  function nearestGaps(gaps, ends) {
    const out = new Set();
    ends.forEach((e) => { let best = null, bd = Infinity; gaps.forEach((g) => { const dd = Math.abs(g - e); if (dd < bd) { bd = dd; best = g; } }); if (best != null && bd < 40) out.add(best); });
    return out;
  }
  function appendFeed(d, bulk = false) {
    const feed = $('#feed'); if (!feed) return;
    if (feed.firstElementChild && !feed.firstElementChild.dataset.i) feed.innerHTML = '';
    const li = document.createElement('li');
    li.dataset.i = d.index; if (d.boundary) li.className = 'b';
    li.innerHTML = `<span class="i">#${d.index}</span><span class="t">${d.boundary ? `<strong>new chunk</strong> · ${esc(d.title || '')}` : 'continues'} <span class="muted">${esc(d.reason === 'llm' ? 'model' : d.reason)}</span></span><span class="ms">${fmtMs(d.ms)}</span>`;
    feed.appendChild(li);
    while (feed.children.length > 150) feed.firstElementChild.remove();
    if (!bulk) feed.scrollTop = feed.scrollHeight;
  }
  let stepTimer;
  function flashSteps(d) {
    const steps = $('#steps'); if (!steps) return;
    const seq = ['inspect', 'ctx', 'bound']; let i = 0;
    clearInterval(stepTimer);
    const tick = () => { steps.querySelectorAll('.step').forEach((s) => s.classList.toggle('on', s.dataset.k === seq[i])); i++; if (i > seq.length) { clearInterval(stepTimer); steps.querySelectorAll('.step').forEach((s) => s.classList.remove('on')); } };
    tick(); stepTimer = setInterval(tick, Math.max(60, Math.min(260, d.ms / 3)));
  }

  function renderCompare() {
    const r = S.run, body = $('#compare tbody');
    if (!r) return;
    body.innerHTML = r.strategies.map((s) => {
      const res = r.results[s] || {}; const st = res.stats;
      const stab = res.stability ? `<span class="${res.stability.score >= 0.8 ? 'good' : res.stability.score >= 0.5 ? 'meh' : 'bad'}">${fmtPct(res.stability.score, 0)}</span>` : s === 'propositional' && r.summary?.stability ? 'n/a' : '–';
      return `<tr data-s="${s}" data-testid="row-compare-${s}"><td>${STRATEGY_INFO[s] || s}</td><td><span class="st-${res.status || 'queued'}">${res.status || 'queued'}</span></td>
        <td class="num">${st ? st.chunks : '–'}</td><td class="num">${st ? st.avg_tokens : '–'}</td><td class="num">${st ? `${st.min_tokens}–${st.max_tokens}` : '–'}</td>
        <td class="num ${st ? (st.mid_sentence_cuts ? 'bad' : 'good') : ''}">${st ? st.mid_sentence_cuts : '–'}</td><td class="num">${stab}</td><td class="num">${res.seconds != null ? fmtMs(res.seconds * 1000) : '–'}</td></tr>`;
    }).join('');
    body.querySelectorAll('tr').forEach((tr) => tr.addEventListener('click', () => { S.tab = tr.dataset.s; S.pinned = true; renderTabs(); renderLive(); $('#card-live').scrollIntoView({ behavior: 'smooth', block: 'start' }); }));
  }

  // ---------------------------------------------------------------- health
  let lastOverall = null;
  function renderHealth(h) {
    if (!h || !h.components) return;
    S.health = h;
    const ov = $('#overall'); ov.dataset.status = h.overall;
    $('.overall-label', ov).textContent = STATUS_LABEL[h.overall] || h.overall;
    if (lastOverall && lastOverall !== h.overall) addLog({ ts: h.ts, text: `health ${lastOverall} → ${h.overall}`, cls: h.overall === 'ok' ? 'good' : h.overall === 'down' ? 'err' : 'warn' });
    lastOverall = h.overall;
    $('#health-updated').textContent = clock(h.ts);
    const c = h.components, hist = h.history || {};
    const color = { ok: 'var(--ok)', degraded: 'var(--warn)', down: 'var(--err)', off: 'var(--off)' };
    const o = c.ollama, e = c.embedder, l = c.llm, ca = c.cache, w = c.worker, p = c.process;
    const m = S.metrics;
    const rows = [
      { id: 'ollama', name: 'Ollama', s: o.status, detail: o.status === 'down' || o.status === 'off' ? `<span class="err">${esc(o.error || 'not running')}</span>${o.required ? '' : ' · not required'}` : `v${esc(o.version)} · ping ${fmtMs(o.ping_ms)} · ${o.installed?.length ?? 0} installed · ${o.loaded?.length ?? 0} loaded`, spark: (hist.ollama || []).map((x) => x.ms) },
      { id: 'embedder', name: 'Embedder', s: e.status, detail: `${esc(e.name || e.spec)}${e.dim ? ` · ${e.dim}-d` : ''} · probe ${fmtMs(e.probe_ms)}${e.paused ? ' (paused during run)' : ''}${m?.ops.embed.p95 != null ? ` · p95 ${fmtMs(m.ops.embed.p95)}` : ''}${e.reason ? ` · ${esc(e.reason)}` : ''}${e.error ? `<br><span class="err">${esc(e.error)}</span>` : ''}`, spark: (hist.embedder || []).map((x) => x.ms) },
      { id: 'llm', name: 'LLM', s: l.status, detail: l.status === 'off' ? 'offline heuristics (no model)' : `${esc(l.name || l.spec)} · probe ${fmtMs(l.probe_ms)}${l.paused ? ' (paused during run)' : ''}${m?.ops.llm.p95 != null ? ` · p95 ${fmtMs(m.ops.llm.p95)}` : ''}${m?.ops.llm.tokens_per_s ? ` · ${m.ops.llm.tokens_per_s} tok/s` : ''}${l.reason ? ` · ${esc(l.reason)}` : ''}${l.error ? `<br><span class="err">${esc(l.error)}</span>` : ''}`, spark: (hist.llm || []).map((x) => x.ms) },
      { id: 'cache', name: 'Embedding cache', s: ca.status, detail: `${num(ca.entries)} vectors · ${fmtBytes(ca.bytes)}${m?.cache.hit_rate != null ? ` · hit rate ${fmtPct(m.cache.hit_rate, 0)}` : ''}` },
      { id: 'worker', name: 'Worker', s: w.status, detail: `${w.state}${w.current ? ` ${esc(w.current.id)}${w.current.progress ? ` (${esc(w.current.progress.strategy)} ${w.current.progress.index + 1}/${w.current.progress.total})` : ''}` : ''} · queue ${w.queued} · ${w.completed} done · ${w.failed} failed${w.llm_budget ? ` · model calls ${num(w.llm_budget.used)}/${num(w.llm_budget.limit)} this hour` : ''}` },
      { id: 'process', name: 'Process', s: p.status, detail: `RSS ${p.rss_mb ?? '–'} MB · CPU ${p.cpu_pct}% · load ${p.load?.[0] ?? '–'}/${p.cpus} · ${p.threads} threads · up ${fmtDur(p.uptime_s)}` },
    ];
    const host = $('#components');
    host.innerHTML = rows.map((r) => `<li class="comp" data-testid="component-${r.id}" data-status="${r.s}">
      <span class="ic s-${r.s}">${ICON[r.s] || ICON.off}</span>
      <span class="name"><b>${r.name}</b><span class="lbl s-${r.s}">${STATUS_LABEL[r.s] || r.s}</span></span>
      <span class="spark-slot"></span>
      <span class="detail">${r.detail}</span></li>`).join('');
    rows.forEach((r, i) => { if (r.spark) host.children[i].querySelector('.spark-slot').replaceWith(sparkline(r.spark.slice(-60), color[r.s] || 'var(--accent)')); });

    const tl = $('#timeline'); const hs = (hist.overall || []).slice(-150);
    tl.innerHTML = Array.from({ length: 150 - hs.length }, () => '<i></i>').join('') + hs.map((x) => `<i class="${x.s}" title="${clock(x.t)} ${x.s}"></i>`).join('');
    renderKpis(); renderModels(o, c.hosted || []);
  }

  function renderKpis() {
    const h = S.health, m = S.metrics; if (!h) return;
    const st = h.stability, ms = m?.stability || {};
    const jit = m ? (m.ops.llm.jitter_cv ?? m.ops.embed.jitter_cv) : null;
    const kpi = (k, v, id, cls = '') => `<div class="kpi" data-testid="kpi-${id}"><div class="k">${k}</div><div class="v ${cls}">${v}</div></div>`;
    $('#kpis').innerHTML = [
      kpi('Healthy (5m)', fmtPct(st.healthy_share_5m, 0), 'healthy', st.healthy_share_5m >= 0.99 ? 'good' : st.healthy_share_5m >= 0.9 ? 'meh' : 'bad'),
      kpi('Status changes', st.status_changes_10m, 'flaps', st.status_changes_10m > 2 ? 'meh' : ''),
      kpi('Error rate', fmtPct(ms.error_rate_5m ?? 0), 'errors', (ms.error_rate_5m || 0) > 0.05 ? 'bad' : ''),
      kpi('Retries', ms.retries_5m ?? 0, 'retries', ms.retries_5m ? 'meh' : ''),
      kpi('Fallbacks', ms.fallbacks_5m ?? 0, 'fallbacks', ms.fallbacks_5m ? 'meh' : ''),
      kpi('Latency jitter', jit == null ? '–' : `${(jit * 100).toFixed(0)}%`, 'jitter', jit > 0.5 ? 'meh' : ''),
    ].join('');
  }

  function renderMetrics(m) {
    S.metrics = m;
    const css = getComputedStyle(document.documentElement);
    const accent = css.getPropertyValue('--accent').trim(), amber = css.getPropertyValue('--c2b').trim();
    lineChart($('#chart-embed'), [
      { values: m.series.map((b) => b.embed_p95), color: amber, width: 1.2 },
      { values: m.series.map((b) => b.embed_p50), color: accent, width: 1.6 }], { emptyText: 'No embedding calls in the last 10 min' });
    lineChart($('#chart-llm'), [
      { values: m.series.map((b) => b.llm_p95), color: amber, width: 1.2 },
      { values: m.series.map((b) => b.llm_p50), color: accent, width: 1.6 }], { emptyText: 'No LLM calls in the last 10 min' });
    renderKpis();
    if (S.run && S.run.results[S.tab]?.status === 'running') renderStats();
  }

  function hostedHTML(hosted) {
    return hosted.map((x) => `<li class="model" data-testid="model-hosted-${esc(x.kind)}">
        <b>${esc(x.model)}</b><span class="exp s-${esc(x.status || 'off')}">${esc(STATUS_LABEL[x.status] || x.status || '')}</span>
        <span class="meta">${[`${x.provider} · ${x.kind}`, x.dim ? `${x.dim}-d` : '', x.probe_ms != null ? `probe ${fmtMs(x.probe_ms)}` : '',
          x.p95 != null ? `p95 ${fmtMs(x.p95)}` : '', x.tokens_per_s ? `${x.tokens_per_s} tok/s` : '', `${x.calls_5m} calls (5m)`].filter(Boolean).join(' · ')}</span></li>`).join('');
  }
  function renderModels(o, hosted = []) {
    const host = $('#models');
    const ollamaUp = o && o.status !== 'down' && o.status !== 'off';
    if (!ollamaUp) {
      $('#models-sub').textContent = hosted.length ? `${hosted.length} hosted` : 'Ollama';
      host.innerHTML = hosted.length ? hostedHTML(hosted) : `<li class="empty">Ollama is not reachable at ${esc(o?.host || 'localhost:11434')}.</li>`;
      return;
    }
    $('#models-sub').textContent = `${hosted.length ? `${hosted.length} hosted · ` : ''}${o.loaded.length} loaded · ${o.installed.length} installed`;
    if (!o.loaded.length) { host.innerHTML = hostedHTML(hosted) + (hosted.length ? '' : '<li class="empty">No models in memory. They load on first use.</li>'); return; }
    host.innerHTML = hostedHTML(hosted) + o.loaded.map((x) => {
      const gpu = x.size ? x.size_vram / x.size : 0;
      return `<li class="model" data-testid="model-${esc(x.name)}" data-expires="${esc(x.expires_at || '')}">
        <b>${esc(x.name)}</b><span class="exp" title="Unloads when idle">–</span>
        <span class="meta">${[x.params, x.quant, fmtBytes(x.size), gpu > 0.01 ? `${Math.round(gpu * 100)}% GPU` : 'CPU', x.context_length ? `ctx ${num(x.context_length)}` : ''].filter(Boolean).join(' · ')}</span>
        <span class="bar"><i style="width:0%"></i></span></li>`;
    }).join('');
    tickCountdowns();
  }
  function tickCountdowns() {
    document.querySelectorAll('.model[data-expires]').forEach((el) => {
      const exp = Date.parse(el.dataset.expires); if (!exp) return;
      const left = (exp - Date.now()) / 1000;
      $('.exp', el).textContent = left > 0 ? `unloads in ${fmtDur(left)}` : 'unloading';
      $('.bar i', el).style.width = `${Math.max(0, Math.min(100, left / 300 * 100))}%`;
    });
    if (S.run) { const res = S.run.results[S.tab]; if (res?.status === 'running') renderStats(); }
  }

  // ---------------------------------------------------------------- log
  function addLog({ ts, text, cls = '' }) {
    const log = $('#log');
    const atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 8;
    const li = document.createElement('li'); li.className = cls;
    li.innerHTML = `<span class="ts">${clock(ts)}</span><span>${esc(text)}</span>`;
    log.appendChild(li);
    while (log.children.length > 300) log.firstElementChild.remove();
    if (atBottom) log.scrollTop = log.scrollHeight;
  }

  window.__chunkMonitor = S; // read-only handle for debugging in devtools
  init();
})();
