/*
 * Expert Engagement report charts.
 *
 * Reads the aggregate numbers from <script type="application/json" id="expert-data"> (written by
 * _includes/nd/expert_engagement.html from docs/_data/expert_engagement.json) and draws the report into
 * <div id="expert-report">. The data holds no names, email addresses or per-person rows.
 *
 * Same look and behaviour as the cohort reports (assets/js/cohort-charts.js): every chart has a Chart / Table
 * switch, colours come from the theme's --nd-viz-* variables, and text is inserted as text nodes, never as HTML.
 * The styles are the .cr-* rules in _sass/custom/nd/_charts.scss.
 *
 * No dependencies and no network requests. ES5 syntax, so the file needs no build step.
 */
(function () {
  'use strict';

  var root = document.getElementById('expert-report');
  var dataNode = document.getElementById('expert-data');
  if (!root || !dataNode) { return; }
  var D;
  try { D = JSON.parse(dataNode.textContent); } catch (e) { return; }

  var SVGNS = 'http://www.w3.org/2000/svg';
  var C = { blue: 'var(--nd-viz-blue)', orange: 'var(--nd-viz-orange)' };

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
  function pct(a, b) { return b ? Math.round(100 * a / b) + '%' : '–'; }
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

  /* Charts are drawn at the card's real pixel width; redrawn when the width changes. */
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
  function card(parent, title, sub, draw, table, foot) {
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
    parent.appendChild(h('figure', { 'class': 'cr-card' }, head, body, foot ? h('p', { 'class': 'cr-foot', text: foot }) : null));
    set(false);
    redraws.push({ body: body, chart: function () { return bChart.getAttribute('aria-pressed') === 'true'; }, redraw: function () { set(false); }, w: body.clientWidth });
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

  /* ---------- svg stacked columns ---------- */
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
  function columns(cols, series, ariaLabel, width) {
    var W = width, H = 260, m = { l: 40, r: 10, t: 24, b: 34 };
    var totals = cols.map(function (c) {
      return series.reduce(function (a, se) { return a + (c.values[se.k] || 0); }, 0);
    });
    var top = Math.max.apply(null, totals.concat([1]));
    var step = niceStep(top / 4), ymax = step * Math.ceil(top / step), nt = Math.round(ymax / step);
    var pw = W - m.l - m.r, ph = H - m.t - m.b, band = pw / cols.length, cw = Math.min(24, band * 0.6);
    function y(v) { return m.t + ph - ph * v / ymax; }
    var svg = s('svg', { 'class': 'cr-svg', viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': ariaLabel });
    var grid = s('g', { 'class': 'grid' }), axis = s('g', { 'class': 'axis' }), i;
    for (i = 0; i <= nt; i++) {
      var v = step * i;
      (i ? grid : axis).appendChild(s('line', { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v) }));
      svg.appendChild(s('text', { x: m.l - 8, y: y(v) + 4, 'text-anchor': 'end', text: num(Math.round(v)) }));
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
      if (totals[idx] > 0 && cols.length <= 16) {
        svg.appendChild(s('text', { 'class': 'val', x: cx, y: y(totals[idx]) - 7, 'text-anchor': 'middle', text: num(totals[idx]) }));
      }
      if (idx % (cols.length > 10 && W < 520 ? 2 : 1) === 0) { svg.appendChild(s('text', { x: cx, y: H - 12, 'text-anchor': 'middle', text: c.label })); }
      var hit = s('rect', { 'class': 'hit', x: m.l + band * idx, y: m.t, width: band, height: ph + 6, 'aria-label': c.long + ': ' + totals[idx] });
      bindTip(hit, function () {
        return { title: c.long, rows: series.map(function (se) {
          return { color: se.color, value: num(c.values[se.k] || 0), label: se.label };
        }).concat([{ value: num(totals[idx]), label: 'total' }]) };
      });
      svg.appendChild(hit);
    });
    return svg;
  }

  /* ---------- horizontal count bars ---------- */
  function bars(rows, total) {
    if (!rows.length) { return h('p', { 'class': 'cr-note', text: 'Nothing to show yet.' }); }
    var max = Math.max.apply(null, rows.map(function (r) { return r.n; }).concat([1]));
    return h('div', { 'class': 'cr-rows' }, rows.map(function (r) {
      var bar = h('div', { 'class': 'cr-bar' }, h('span', { style: 'width:' + (100 * r.n / max) + '%;background:' + C.blue }));
      var row = h('div', { 'class': 'cr-row cr-row-bar' }, h('div', { 'class': 'k', text: r.label }), bar, h('div', { 'class': 'v' }, h('b', { text: num(r.n) })));
      bindTip(row, function () { return { title: r.label, rows: [{ color: C.blue, value: num(r.n), label: pct(r.n, total) + ' of the group' }] }; });
      return row;
    }));
  }
  function countCard(parent, title, sub, rows, head, foot) {
    card(parent, title, sub, function () { return bars(rows, T.engaged); },
      { head: [head, 'People', 'Share of group'], rows: rows.map(function (r) { return [r.label, num(r.n), pct(r.n, T.engaged)]; }) }, foot);
  }

  var T = D.tiles;

  /* ---------- headline numbers ---------- */
  var tiles = [
    ['Engaging experts', num(T.engaged), 'replied with interest or joined the discussion'],
    ['In the discussion', num(T.joined), pct(T.joined, T.engaged) + ' of the group'],
    ['Wrote back by email', num(T.emailed), pct(T.emailed, T.engaged) + ' of the group'],
    ['Did both', num(T.both), 'email reply and discussion post'],
    ['Chose Content Review', num(T.content_review), 'named a track in their reply']
  ];
  root.appendChild(h('div', { 'class': 'cr-tiles' }, tiles.map(function (t) {
    return h('div', { 'class': 'cr-tile' }, h('p', { 'class': 'l', text: t[0] }), h('p', { 'class': 'v', text: t[1] }), h('p', { 'class': 's', text: t[2] }));
  })));

  /* ---------- engagement ---------- */
  var g1 = section('Engagement', 'Counts of experts who replied with interest or posted in the community discussion. Invitation totals are not shown.');

  var SER = [{ k: 'replied', label: 'Replied by email', color: C.blue }, { k: 'joined', label: 'Posted in the discussion', color: C.orange }];
  card(g1, 'Engagement per day', 'Each person counted once per door, on the day they used it', function (w) {
    var cols = D.days.map(function (d) { return { label: dayLabel(d.d), long: dayLabel(d.d, true), values: { replied: d.replied, joined: d.joined } }; });
    return cols.length ? h('div', {}, legend(SER), columns(cols, SER, 'Engagement per day, by type', w)) : h('p', { 'class': 'cr-note', text: 'No dated engagement yet.' });
  }, { head: ['Day', 'Replied by email', 'Posted in the discussion', 'Total'],
       rows: D.days.map(function (d) { return [dayLabel(d.d, true), num(d.replied), num(d.joined), num(d.replied + d.joined)]; }) },
    D.undated_joins ? D.undated_joins + ' discussion ' + (D.undated_joins === 1 ? 'post has' : 'posts have') + ' no recorded date and ' + (D.undated_joins === 1 ? 'is' : 'are') + ' not in this chart.' : null);

  countCard(g1, 'What they have done', 'Steps from the invitation, among engaging experts', D.steps, 'Step',
    'One person can be counted in several rows.');

  /* ---------- who ---------- */
  var g2 = section('Who is engaging', 'Counts of engaging experts. Groups smaller than ' + D.min_cell + ' are combined or left out so that no one can be picked out.');
  countCard(g2, 'Research Area', 'Most common expertise tags', D.by_tag, 'Research area',
    'Each person carries several tags, so rows add up to more than the group size.');
  countCard(g2, 'Organizations', num(D.n_orgs) + ' distinct organizations among the engaging experts', D.by_org, 'Organization',
    'Organizations are matched by email domain. Only those with at least ' + D.min_cell + ' engaging experts are named.');

  root.appendChild(h('section', { 'class': 'cr-section cr-about' }, h('h2', { text: 'About these numbers' }), h('ul', {},
    h('li', { text: 'Engaging means replying with interest by email, posting an introduction in the community discussion, or both.' }),
    h('li', { text: 'The group is small, so one person moves a bar noticeably. Read the counts, not only the shares.' }),
    h('li', { text: 'As of ' + D.as_of + '.' }))));
  root.setAttribute('data-ready', 'true');
})();
