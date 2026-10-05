/*
 * Cohort report charts.
 *
 * Reads the aggregate numbers for one cohort from <script type="application/json" id="cohort-data">
 * (written by _includes/nd/cohort_report.html from docs/_data/cohorts/<id>.json) and draws the report
 * into <div id="cohort-report">. The data holds no names or email addresses.
 *
 * Every chart has a Chart / Table switch, and every value in a tooltip is also in the table.
 * Colours come from the theme's CSS variables (--nd-viz-*), so the charts follow the light and dark
 * themes. Text is inserted as text nodes, never as HTML.
 *
 * No dependencies and no network requests. ES5 syntax, so the file needs no build step.
 */
(function () {
  'use strict';

  var root = document.getElementById('cohort-report');
  var dataNode = document.getElementById('cohort-data');
  if (!root || !dataNode) { return; }
  var D;
  try { D = JSON.parse(dataNode.textContent); } catch (e) { return; }

  var SVGNS = 'http://www.w3.org/2000/svg';
  var C = {
    blue: 'var(--nd-viz-blue)', orange: 'var(--nd-viz-orange)', neutral: 'var(--nd-viz-neutral)',
    r1: 'var(--nd-viz-ramp-1)', r2: 'var(--nd-viz-ramp-2)', r3: 'var(--nd-viz-ramp-3)', r4: 'var(--nd-viz-ramp-4)'
  };

  /* ---------- helpers ---------- */
  function make(ns, tag, attrs, kids) {
    var e = ns ? document.createElementNS(ns, tag) : document.createElement(tag);
    var k;
    for (k in (attrs || {})) {
      if (Object.prototype.hasOwnProperty.call(attrs, k) && attrs[k] !== false && attrs[k] != null) {
        if (k === 'class') { e.setAttribute('class', attrs[k]); }
        else if (k === 'text') { e.textContent = attrs[k]; }
        else { e.setAttribute(k, attrs[k] === true ? '' : attrs[k]); }
      }
    }
    (function add(list) {
      (list || []).forEach(function (c) {
        if (c == null) { return; }
        if (Array.isArray(c)) { add(c); return; }
        e.appendChild(c.nodeType ? c : document.createTextNode(String(c)));
      });
    })(kids);
    return e;
  }
  function h(tag, attrs) { return make(null, tag, attrs, Array.prototype.slice.call(arguments, 2)); }
  function s(tag, attrs) { return make(SVGNS, tag, attrs, Array.prototype.slice.call(arguments, 2)); }
  function num(n) { return n == null ? '–' : n.toLocaleString('en-US'); }
  function pct(a, b, d) { return b ? (100 * a / b).toFixed(d == null ? 0 : d) + '%' : '–'; }
  function dayLabel(iso, long) {
    var d = new Date(iso + 'T12:00:00');
    return d.toLocaleDateString('en-US', long ? { weekday: 'short', month: 'short', day: 'numeric' } : { month: 'short', day: 'numeric' });
  }

  /* ---------- tooltip (enhances; values are also in the tables) ---------- */
  var tip = h('div', { 'class': 'cr-tip', role: 'tooltip', hidden: true });
  document.body.appendChild(tip);
  function showTip(spec, x, y) {
    while (tip.firstChild) { tip.removeChild(tip.firstChild); }
    tip.appendChild(h('div', { 'class': 't', text: spec.title }));
    spec.rows.forEach(function (r) {
      var row = h('div', { 'class': 'row' });
      if (r.color) { row.appendChild(h('s', { style: 'background:' + r.color })); }
      row.appendChild(h('b', { text: r.value }));
      row.appendChild(h('span', { text: r.label }));
      tip.appendChild(row);
    });
    tip.hidden = false;
    var w = tip.offsetWidth, hh = tip.offsetHeight, left = x + 14, top = y + 14;
    if (left + w > window.innerWidth - 8) { left = Math.max(8, x - w - 14); }
    if (top + hh > window.innerHeight - 8) { top = Math.max(8, y - hh - 14); }
    tip.style.left = left + 'px';
    tip.style.top = top + 'px';
  }
  function hideTip() { tip.hidden = true; }
  function bindTip(node, getSpec) {
    node.setAttribute('tabindex', '0');
    node.addEventListener('pointermove', function (e) { showTip(getSpec(), e.clientX, e.clientY); });
    node.addEventListener('pointerleave', hideTip);
    node.addEventListener('focus', function () {
      var r = node.getBoundingClientRect();
      showTip(getSpec(), r.left + Math.min(r.width, 120), r.top + r.height / 2);
    });
    node.addEventListener('blur', hideTip);
  }

  /* Charts are drawn at the card's real pixel width, so text stays readable; redrawn when the width changes. */
  var redraws = [], resizeTimer = null;
  window.addEventListener('resize', function () {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(function () {
      redraws.forEach(function (r) {
        var w = r.body.clientWidth;
        if (r.chart() && Math.abs(w - r.w) > 24) { r.w = w; r.redraw(); }
      });
    }, 150);
  });

  /* ---------- card with a Chart / Table twin ---------- */
  function dataTable(head, rows) {
    return h('div', { 'class': 'cr-tw' }, h('table', { 'class': 'cr-dt' },
      h('thead', {}, h('tr', {}, head.map(function (t, i) { return h('th', { 'class': i ? 'num' : '', scope: 'col', text: t }); }))),
      h('tbody', {}, rows.map(function (r) {
        return h('tr', {}, r.map(function (c, i) { return h('td', { 'class': i ? 'num' : '', text: c }); }));
      }))));
  }
  function card(parent, title, sub, draw, table, foot, wide) {
    var body = h('div', { 'class': 'cr-body' });
    var bChart = h('button', { type: 'button', 'aria-pressed': 'true', text: 'Chart' });
    var bTable = h('button', { type: 'button', 'aria-pressed': 'false', text: 'Table' });
    function set(t) {
      bChart.setAttribute('aria-pressed', String(!t));
      bTable.setAttribute('aria-pressed', String(t));
      hideTip();
      while (body.firstChild) { body.removeChild(body.firstChild); }
      body.appendChild(t ? dataTable(table.head, table.rows) : draw(Math.max(260, Math.floor(body.clientWidth || 560))));
    }
    bChart.addEventListener('click', function () { set(false); });
    bTable.addEventListener('click', function () { set(true); });
    var head = h('header', {}, h('div', {}, h('h3', { text: title }), sub ? h('p', { text: sub }) : null),
      h('div', { 'class': 'cr-tools', role: 'group', 'aria-label': title + ' view' }, bChart, bTable));
    var fig = h('figure', { 'class': 'cr-card' + (wide ? ' cr-wide' : '') }, head, body, foot ? h('p', { 'class': 'cr-foot', text: foot }) : null);
    parent.appendChild(fig);
    set(false);
    redraws.push({ body: body, chart: function () { return bChart.getAttribute('aria-pressed') === 'true'; }, redraw: function () { set(false); }, w: body.clientWidth });
    return fig;
  }
  function legend(items) {
    return h('ul', { 'class': 'cr-legend' }, items.map(function (i) {
      return h('li', {}, h('i', { style: 'background:' + i.color }), i.label);
    }));
  }
  function section(title, lede) {
    var sec = h('section', { 'class': 'cr-section' }, h('h2', { text: title }), lede ? h('p', { 'class': 'cr-lede', text: lede }) : null);
    var grid = h('div', { 'class': 'cr-grid' });
    sec.appendChild(grid);
    root.appendChild(sec);
    return grid;
  }

  /* ---------- svg columns (single or stacked) ---------- */
  function niceStep(v) {
    var p = Math.pow(10, Math.floor(Math.LOG10E * Math.log(v))), m, steps = [1, 2, 5, 10];
    for (m = 0; m < steps.length; m++) { if (steps[m] * p >= v) { return steps[m] * p; } }
    return 10 * p;
  }
  function roundedTop(x, y, w, hh, r) {
    r = Math.max(0, Math.min(r, hh, w / 2));
    return 'M' + x + ',' + (y + hh) + 'V' + (y + r) + 'Q' + x + ',' + y + ' ' + (x + r) + ',' + y + 'H' + (x + w - r) +
      'Q' + (x + w) + ',' + y + ' ' + (x + w) + ',' + (y + r) + 'V' + (y + hh) + 'Z';
  }
  /* cols: [{label, long, values: {key: n|null}}], series: [{k, label, color}] */
  function columns(cols, series, ariaLabel, opts) {
    opts = opts || {};
    var W = opts.width || 560, H = 260, m = { l: 40, r: 10, t: 24, b: 34 };
    var totals = cols.map(function (c) {
      return series.reduce(function (a, se) { return a + (c.values[se.k] || 0); }, 0);
    });
    var top = opts.max || Math.max.apply(null, totals.concat([1]));
    var step = opts.step || niceStep(top / 4), ymax = opts.max || step * Math.ceil(top / step), nt = Math.round(ymax / step);
    var pw = W - m.l - m.r, ph = H - m.t - m.b, band = pw / cols.length, cw = Math.min(24, band * 0.6);
    function y(v) { return m.t + ph - ph * v / ymax; }
    var svg = s('svg', { 'class': 'cr-svg', viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': ariaLabel });
    var grid = s('g', { 'class': 'grid' }), axis = s('g', { 'class': 'axis' }), i;
    for (i = 0; i <= nt; i++) {
      var v = step * i;
      (i ? grid : axis).appendChild(s('line', { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v) }));
      svg.appendChild(s('text', { x: m.l - 8, y: y(v) + 4, 'text-anchor': 'end', text: opts.fmt ? opts.fmt(v) : num(Math.round(v)) }));
    }
    svg.insertBefore(axis, svg.firstChild);
    svg.insertBefore(grid, svg.firstChild);
    cols.forEach(function (c, idx) {
      var cx = m.l + band * idx + band / 2, x0 = cx - cw / 2, acc = 0;
      var present = series.filter(function (se) { return c.values[se.k] > 0; });
      present.forEach(function (se, j) {
        var val = c.values[se.k], yTop = y(acc + val), hh = Math.max(1, y(acc) - yTop - (j ? 2 : 0));
        svg.appendChild(s('path', {
          'class': 'mark', fill: se.color,
          d: j === present.length - 1 ? roundedTop(x0, yTop, cw, hh, 4) : 'M' + x0 + ',' + yTop + 'h' + cw + 'v' + hh + 'h' + (-cw) + 'Z'
        }));
        acc += val;
      });
      if (totals[idx] > 0 && !opts.noTotals && cols.length <= 16) {
        svg.appendChild(s('text', { 'class': 'val', x: cx, y: y(totals[idx]) - 7, 'text-anchor': 'middle', text: opts.fmt ? opts.fmt(totals[idx]) : num(totals[idx]) }));
      }
      if (totals[idx] === 0 && c.noData) {
        svg.appendChild(s('text', { x: cx, y: y(0) - 6, 'text-anchor': 'middle', text: '–' }));
      }
      if (idx % (cols.length > 10 && W < 520 ? 2 : 1) === 0) { svg.appendChild(s('text', { x: cx, y: H - 12, 'text-anchor': 'middle', text: c.label })); }
      var hit = s('rect', { 'class': 'hit', x: m.l + band * idx, y: m.t, width: band, height: ph + 6, 'aria-label': c.long + ': ' + (opts.fmt ? opts.fmt(totals[idx]) : totals[idx]) });
      bindTip(hit, function () {
        return { title: c.long, rows: series.map(function (se) {
          return { color: se.color, value: c.values[se.k] == null ? '–' : (opts.fmt ? opts.fmt(c.values[se.k]) : num(c.values[se.k])), label: se.label };
        }).concat(c.extra || []) };
      });
      svg.appendChild(hit);
    });
    return svg;
  }

  /* ---------- svg line (one series) ---------- */
  function line(points, ariaLabel, valueLabel, width) {
    var W = width || 560, H = 250, m = { l: 44, r: 44, t: 24, b: 34 };
    var pw = W - m.l - m.r, ph = H - m.t - m.b;
    function x(i) { return m.l + (points.length === 1 ? pw / 2 : pw * i / (points.length - 1)); }
    function y(v) { return m.t + ph - ph * v; }
    var svg = s('svg', { 'class': 'cr-svg', viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': ariaLabel });
    var grid = s('g', { 'class': 'grid' }), axis = s('g', { 'class': 'axis' }), i;
    for (i = 0; i <= 4; i++) {
      (i ? grid : axis).appendChild(s('line', { x1: m.l, x2: W - m.r, y1: y(i / 4), y2: y(i / 4) }));
      svg.appendChild(s('text', { x: m.l - 8, y: y(i / 4) + 4, 'text-anchor': 'end', text: (i * 25) + '%' }));
    }
    svg.insertBefore(axis, svg.firstChild);
    svg.insertBefore(grid, svg.firstChild);
    var segs = [], cur = [];
    points.forEach(function (p, idx) {
      if (p.v == null) { if (cur.length) { segs.push(cur); cur = []; } return; }
      cur.push(x(idx) + ',' + y(p.v));
    });
    if (cur.length) { segs.push(cur); }
    segs.forEach(function (sg) {
      if (sg.length > 1) { svg.appendChild(s('polyline', { 'class': 'ln', points: sg.join(' '), fill: 'none', stroke: C.blue })); }
    });
    var lastIdx = -1;
    points.forEach(function (p, idx) { if (p.v != null) { lastIdx = idx; } });
    points.forEach(function (p, idx) {
      if (idx % (points.length > 10 && W < 520 ? 2 : 1) === 0) { svg.appendChild(s('text', { x: x(idx), y: H - 12, 'text-anchor': 'middle', text: p.label })); }
      if (p.v == null) {
        svg.appendChild(s('text', { x: x(idx), y: y(0) - 6, 'text-anchor': 'middle', text: '–' }));
      } else {
        svg.appendChild(s('circle', { 'class': 'dot', cx: x(idx), cy: y(p.v), r: 4, fill: C.blue }));
        if (idx === lastIdx || idx === 0) {
          svg.appendChild(s('text', { 'class': 'val', x: x(idx) + 8, y: y(p.v) - 10, 'text-anchor': 'start', text: Math.round(p.v * 100) + '%' }));
        }
      }
      var hit = s('rect', { 'class': 'hit', x: x(idx) - pw / points.length / 2, y: m.t, width: pw / points.length, height: ph + 6, 'aria-label': p.long + ': ' + (p.v == null ? 'no data' : Math.round(p.v * 100) + '%') });
      bindTip(hit, function () {
        return { title: p.long, rows: p.v == null ? [{ value: 'No data', label: p.why || '' }] : [{ color: C.blue, value: Math.round(p.v * 100) + '%', label: valueLabel }].concat(p.extra || []) };
      });
      svg.appendChild(hit);
    });
    return svg;
  }

  /* ---------- html stacked bar rows ---------- */
  function stackRow(label, segs, total) {
    var stack = h('div', { 'class': 'cr-stack', role: 'img', 'aria-label': label + ': ' + segs.map(function (g) { return g.label + ' ' + g.n; }).join(', ') });
    segs.forEach(function (g) {
      if (!g.n) { return; }
      var seg = h('div', { 'class': 'seg', style: 'flex:' + g.n + ' 1 0;background:' + g.color + (g.color === C.neutral ? ';color:var(--nd-ink)' : ''), text: g.n / total >= 0.1 ? String(g.n) : '' });
      bindTip(seg, function () { return { title: label, rows: [{ color: g.color, value: num(g.n) + ' (' + pct(g.n, total) + ')', label: g.label }] }; });
      stack.appendChild(seg);
    });
    return stack;
  }

  var T = D.tiles, sessions = D.sessions;
  var held = sessions.map(function (x) { return { label: dayLabel(x.date), long: dayLabel(x.date, true), s: x }; });

  /* ---------- headline numbers ---------- */
  var tiles = [
    ['People who attended', num(T.people), 'at least one held session'],
    ['Sessions held', num(T.sessions), dayLabel(D.first_session) + ' to ' + dayLabel(D.last_session)],
    ['Organizations', num(T.orgs), 'by email domain'],
    ['Median time in a session', T.median_minutes == null ? '–' : Math.round(T.median_minutes) + ' min', 'per person, per session'],
    ['Said the session was helpful', pct(T.helpful_yes, T.helpful_n), num(T.helpful_yes) + ' of ' + num(T.helpful_n) + ' answers'],
    ['Poll response', T.poll_response_rate == null ? '–' : Math.round(T.poll_response_rate * 100) + '%', 'of people present, on average']
  ];
  root.appendChild(h('div', { 'class': 'cr-tiles' }, tiles.map(function (t) {
    return h('div', { 'class': 'cr-tile' }, h('p', { 'class': 'l', text: t[0] }), h('p', { 'class': 'v', text: t[1] }), h('p', { 'class': 's', text: t[2] }));
  })));

  /* ---------- attendance and retention ---------- */
  var g1 = section('Attendance and retention', 'Who came, and whether they kept coming.');

  var attSeries = [{ k: 'returning', label: 'Returning', color: C.blue }, { k: 'new', label: 'First session', color: C.orange }];
  card(g1, 'Attendance per session', 'People present, split into first-time and returning', function (w) {
    var cols = held.map(function (x) { return { label: x.label, long: x.long, values: { returning: x.s.returning, 'new': x.s['new'] } }; });
    return h('div', {}, legend(attSeries), columns(cols, attSeries, 'Attendance per session', { width: w }));
  }, { head: ['Session', 'Present', 'First session', 'Returning', 'Median minutes'],
       rows: held.map(function (x) { return [x.long, num(x.s.attendees), num(x.s['new']), num(x.s.returning), String(Math.round(x.s.median_minutes))]; }) },
    'First session means the first held session a person attended.');

  var ret = D.retention, retPts = [{ label: dayLabel(held[0].s.date), long: held[0].long, v: 1, extra: [{ value: num(ret.opening_group), label: 'people in the opening group' }] }];
  ret.points.forEach(function (p) {
    retPts.push({ label: dayLabel(p.date), long: dayLabel(p.date, true), v: p.share, extra: [{ value: num(Math.round(p.share * ret.opening_group)), label: 'of ' + num(ret.opening_group) + ' came back' }] });
  });
  card(g1, 'Did the opening group keep coming?', 'Share of the ' + num(ret.opening_group) + ' people at the first session who were present each time', function (w) {
    return line(retPts, 'Share of the opening group present at each session', 'of the opening group present', w);
  }, { head: ['Session', 'Share present', 'People'],
       rows: retPts.map(function (p) { return [p.long, Math.round(p.v * 100) + '%', num(Math.round(p.v * ret.opening_group))]; }) });

  var tierColors = [C.r1, C.r2, C.r3, C.r4], tb = D.tiers.bins;
  card(g1, 'How many sessions did people attend?', 'Everyone who attended, by share of the ' + D.tiers.sessions_held + ' held sessions', function () {
    var segs = tb.map(function (b, i) { return { label: b.label + ' of sessions', n: b.people, color: tierColors[i] }; });
    return h('div', {}, legend(segs), stackRow('Share of sessions attended', segs, D.tiers.people),
      h('div', { 'class': 'cr-note' }, h('b', { text: num(tb[3].people) }), ' people attended more than three quarters of the sessions.'));
  }, { head: ['Share of sessions attended', 'People', 'Share of people'],
       rows: tb.map(function (b) { return [b.label, num(b.people), pct(b.people, D.tiers.people)]; }) },
    null, true);

  /* ---------- learning signals ---------- */
  var g2 = section('Learning signals', 'What people said in the polls at the start of each session. Results are shown only when at least ' + D.min_cell + ' people answered.');

  var readPts = held.map(function (x) {
    var ok = x.s.read_n != null;
    return { label: x.label, long: x.long, v: ok ? x.s.read_yes / x.s.read_n : null, why: 'Fewer than ' + D.min_cell + ' answers, or no poll that day.',
             extra: ok ? [{ value: x.s.read_yes + ' of ' + x.s.read_n, label: 'said yes' }] : [] };
  });
  card(g2, 'Did people do the assigned reading?', 'Share who answered yes', function (w) {
    return line(readPts, 'Share of poll respondents who did the assigned reading, by session', 'did the reading', w);
  }, { head: ['Session', 'Yes', 'Answers', 'Share'],
       rows: held.map(function (x) { return [x.long, num(x.s.read_yes), num(x.s.read_n), x.s.read_n ? pct(x.s.read_yes, x.s.read_n) : '–']; }) },
    'Each point is one session. Read the counts in the tooltip or table, because small groups swing.');

  var confSeries = [{ k: 'conf_high', label: 'High', color: C.blue }, { k: 'conf_med', label: 'Medium', color: C.neutral }, { k: 'conf_low', label: 'Low', color: C.orange }];
  card(g2, 'How confident do people feel?', 'Share of answers by level, per session', function (w) {
    var cols = held.map(function (x) {
      var n = x.s.conf_n, v = {};
      if (n) { v.conf_high = x.s.conf_high / n * 100; v.conf_med = x.s.conf_med / n * 100; v.conf_low = x.s.conf_low / n * 100; }
      return { label: x.label, long: x.long, values: n ? v : {}, noData: !n,
               extra: n ? [{ value: String(n), label: 'answers' }] : [{ value: 'No data', label: 'fewer than ' + D.min_cell + ' answers, or no poll' }] };
    });
    var ser = [{ k: 'conf_low', label: 'Low', color: C.orange }, { k: 'conf_med', label: 'Medium', color: C.neutral }, { k: 'conf_high', label: 'High', color: C.blue }];
    return h('div', {}, legend([ser[2], ser[1], ser[0]]),
      columns(cols, ser, 'Confidence level by session, as a share of answers', { width: w, max: 100, step: 25, noTotals: true, fmt: function (v) { return Math.round(v) + '%'; } }));
  }, { head: ['Session', 'High', 'Medium', 'Low', 'Answers'],
       rows: held.map(function (x) { return [x.long, num(x.s.conf_high), num(x.s.conf_med), num(x.s.conf_low), num(x.s.conf_n)]; }) },
    'High is at the top of each column. Counts, not percentages, are in the table.');

  var rvc = D.read_vs_confidence;
  if (rvc.read && rvc.noread) {
    card(g2, 'Confidence, with and without the reading', 'Same person, same day: those who did the reading and those who did not', function () {
      function segs(g) { return [{ label: 'High', n: g.high, color: C.blue }, { label: 'Medium', n: g.med, color: C.neutral }, { label: 'Low', n: g.low, color: C.orange }]; }
      var rows = [['Did the reading', rvc.read], ['Did not', rvc.noread]];
      return h('div', {}, legend(segs(rvc.read)), h('div', { 'class': 'cr-rows' }, rows.map(function (r) {
        return h('div', { 'class': 'cr-row' }, h('div', { 'class': 'k' }, r[0], h('small', { text: num(r[1].n) + ' answers' })),
          stackRow(r[0], segs(r[1]), r[1].n), h('div', { 'class': 'v' }, h('b', { text: pct(r[1].high, r[1].n) }), h('small', { text: 'High' })));
      })));
    }, { head: ['Group', 'High', 'Medium', 'Low', 'Answers', 'High share'],
         rows: [['Did the reading', rvc.read], ['Did not', rvc.noread]].map(function (r) {
           return [r[0], num(r[1].high), num(r[1].med), num(r[1].low), num(r[1].n), pct(r[1].high, r[1].n)];
         }) },
      'Doing the reading goes with higher confidence. This does not show that one causes the other.', true);
  }

  /* ---------- time and organizations ---------- */
  var g3 = section('Time in session and organizations', null);
  var th = D.time_in_session;
  card(g3, 'How long did people stay?', 'Person-sessions by minutes in the meeting', function (w) {
    var cols = th.map(function (b) { return { label: b.label, long: b.label + ' minutes', values: { n: b.count } }; });
    return columns(cols, [{ k: 'n', label: 'Person-sessions', color: C.blue }], 'Minutes in the meeting, per person per session', { width: w });
  }, { head: ['Minutes in the meeting', 'Person-sessions'], rows: th.map(function (b) { return [b.label, num(b.count)]; }) },
    'One person who attends five sessions counts five times. Sessions last about an hour.');

  var O = D.organizations;
  if (O) {
    card(g3, 'Where people work', 'Organizations with ' + O.min_people + ' or more people (by email domain)', function () {
      var max = O.listed.length ? O.listed[0].people : 1;
      var rows = O.listed.map(function (o) {
        var bar = h('div', { 'class': 'cr-bar' }, h('span', { style: 'width:' + (100 * o.people / max) + '%;background:' + C.blue }));
        var row = h('div', { 'class': 'cr-row cr-row-bar' }, h('div', { 'class': 'k', text: o.label }), bar, h('div', { 'class': 'v' }, h('b', { text: num(o.people) })));
        bindTip(row, function () { return { title: o.label, rows: [{ color: C.blue, value: num(o.people), label: 'people' }] }; });
        return row;
      });
      rows.push(h('div', { 'class': 'cr-row cr-row-bar cr-other' }, h('div', { 'class': 'k', text: 'Everyone else' }),
        h('div', { 'class': 'cr-bar-note', text: num(O.other_orgs) + ' other organizations, each with fewer than ' + O.min_people }),
        h('div', { 'class': 'v' }, h('b', { text: num(O.other_people) }))));
      return h('div', { 'class': 'cr-rows' }, rows);
    }, { head: ['Organization', 'People'], rows: O.listed.map(function (o) { return [o.label, num(o.people)]; }).concat([['Everyone else (' + num(O.other_orgs) + ' organizations)', num(O.other_people)]]) },
      'Smaller groups are combined so that no one can be picked out.');
  } else {
    var box = h('figure', { 'class': 'cr-card' }, h('header', {}, h('div', {}, h('h3', { text: 'Where people work' }))),
      h('p', { 'class': 'cr-note', text: T.orgs + ' organizations are represented. This cohort is small, so they are not listed, to keep individuals from being identified.' }));
    g3.appendChild(box);
  }

  /* ---------- about the data ---------- */
  var about = h('section', { 'class': 'cr-section cr-about' }, h('h2', { text: 'About these numbers' }));
  about.appendChild(h('ul', {}, D.notes.map(function (n) { return h('li', { text: n }); }).concat(
    D.excluded.map(function (e) { return h('li', { text: dayLabel(e.date, true) + ': ' + e.reason }); }))));
  root.appendChild(about);
  root.setAttribute('data-ready', 'true');
})();
