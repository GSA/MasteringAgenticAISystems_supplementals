/*
 * "Where people come from" chart for the Reports page.
 *
 * Reads the aggregate numbers from <script type="application/json" id="org-data"> (written by
 * _includes/nd/org_chart.html from docs/_data/cohorts/*.json) and draws one column per agency, largest first, into
 * <div id="org-report">. At most TOP agencies are named; everyone else is one combined column. The data holds no
 * names, email addresses or per-person rows.
 *
 * Same look and behaviour as the other reports (assets/js/cohort-charts.js): a Chart / Table switch, colours from the
 * theme's --nd-viz-* variables, and text inserted as text nodes, never as HTML. The styles are the .cr-* rules in
 * _sass/custom/nd/_charts.scss.
 *
 * No dependencies and no network requests. ES5 syntax, so the file needs no build step.
 */
(function () {
  'use strict';

  var TOP = 15;
  var root = document.getElementById('org-report');
  var dataNode = document.getElementById('org-data');
  if (!root || !dataNode) { return; }
  var D;
  try { D = JSON.parse(dataNode.textContent); } catch (e) { return; }
  if (!D.cohorts || !D.cohorts.length) { return; }

  var SVGNS = 'http://www.w3.org/2000/svg';
  var BLUE = 'var(--nd-viz-blue)', GREY = 'var(--nd-viz-neutral)';

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

  /* ---------- tooltip (enhances; values are also in the table) ---------- */
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

  /* ---------- the numbers: merge cohorts, name the top agencies, combine the rest ---------- */
  var byLabel = {}, order = [], covered = 0, otherPeople = 0, otherOrgs = 0, minPeople = 0;
  D.cohorts.forEach(function (c) {
    var o = c.organizations;
    covered += c.people;
    minPeople = Math.max(minPeople, o.min_people);
    otherPeople += o.other_people;
    otherOrgs += o.other_orgs;
    o.listed.forEach(function (x) {
      if (!Object.prototype.hasOwnProperty.call(byLabel, x.label)) { byLabel[x.label] = 0; order.push(x.label); }
      byLabel[x.label] += x.people;
    });
  });
  var named = order.map(function (l) { return { label: l, n: byLabel[l] }; })
    .sort(function (a, b) { return b.n - a.n || (a.label < b.label ? -1 : 1); });
  var dropped = named.slice(TOP);
  named = named.slice(0, TOP);
  dropped.forEach(function (x) { otherPeople += x.n; otherOrgs += 1; });
  var cols = named.slice();
  if (otherPeople) { cols.push({ label: 'All other agencies', n: otherPeople, other: true }); }

  /* ---------- svg columns with agency names on the x axis ---------- */
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
  function lines(label) {   /* up to two lines, split at a space near the middle */
    var i, best = -1, words = label.split(' ');
    if (words.length < 2) { return [label]; }
    for (i = 1; i < words.length; i++) {
      var a = words.slice(0, i).join(' ').length, b = words.slice(i).join(' ').length;
      if (best < 0 || Math.abs(a - b) < Math.abs(best.a - best.b)) { best = { i: i, a: a, b: b }; }
    }
    return [words.slice(0, best.i).join(' '), words.slice(best.i).join(' ')];
  }
  function chart(width) {
    var W = width, rotate = (W - 50) / cols.length < 44, H = rotate ? 330 : 280, m = { l: 40, r: 10, t: 24, b: rotate ? 100 : 50 };
    var top = Math.max.apply(null, cols.map(function (c) { return c.n; }).concat([1]));
    var step = niceStep(top / 4), ymax = step * Math.ceil(top / step), nt = Math.round(ymax / step);
    var pw = W - m.l - m.r, ph = H - m.t - m.b, band = pw / cols.length, cw = Math.min(36, band * 0.62);
    function y(v) { return m.t + ph - ph * v / ymax; }
    var svg = s('svg', { 'class': 'cr-svg', viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': 'People by agency, largest first' });
    var grid = s('g', { 'class': 'grid' }), axis = s('g', { 'class': 'axis' }), i;
    for (i = 0; i <= nt; i++) {
      var v = step * i;
      (i ? grid : axis).appendChild(s('line', { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v) }));
      svg.appendChild(s('text', { x: m.l - 8, y: y(v) + 4, 'text-anchor': 'end', text: num(Math.round(v)) }));
    }
    svg.insertBefore(axis, svg.firstChild);
    svg.insertBefore(grid, svg.firstChild);
    cols.forEach(function (c, idx) {
      var cx = m.l + band * idx + band / 2, x0 = cx - cw / 2, yTop = y(c.n), hh = Math.max(1, y(0) - yTop);
      svg.appendChild(s('path', { 'class': 'mark', fill: c.other ? GREY : BLUE, d: roundedTop(x0, yTop, cw, hh, 4) }));
      svg.appendChild(s('text', { 'class': 'val', x: cx, y: yTop - 7, 'text-anchor': 'middle', text: num(c.n) }));
      if (rotate) {
        svg.appendChild(s('text', { x: cx + 4, y: H - m.b + 16, 'text-anchor': 'end', transform: 'rotate(-45 ' + (cx + 4) + ' ' + (H - m.b + 16) + ')', text: c.label }));
      } else {
        lines(c.label).forEach(function (ln, j) {
          svg.appendChild(s('text', { x: cx, y: H - m.b + 18 + 14 * j, 'text-anchor': 'middle', text: ln }));
        });
      }
      var hit = s('rect', { 'class': 'hit', x: m.l + band * idx, y: m.t, width: band, height: ph + 6, 'aria-label': c.label + ': ' + c.n + ' people' });
      bindTip(hit, function () {
        return { title: c.label, rows: [{ color: c.other ? GREY : BLUE, value: num(c.n), label: pct(c.n, covered) + ' of ' + num(covered) + ' people' }] };
      });
      svg.appendChild(hit);
    });
    return svg;
  }

  /* ---------- card with a Chart / Table twin ---------- */
  var tableRows = cols.map(function (c) { return [c.label, num(c.n), pct(c.n, covered)]; });
  function dataTable() {
    var head = ['Agency', 'People', 'Share'];
    return h('div', { 'class': 'cr-tw' }, h('table', { 'class': 'cr-dt' },
      h('thead', {}, h('tr', {}, head.map(function (t, i) { return h('th', { 'class': i ? 'num' : '', scope: 'col', text: t }); }))),
      h('tbody', {}, tableRows.map(function (r) {
        return h('tr', {}, r.map(function (c, i) { return h('td', { 'class': i ? 'num' : '', text: c }); }));
      }))));
  }
  var body = h('div', { 'class': 'cr-body' });
  var bChart = h('button', { type: 'button', 'aria-pressed': 'true', text: 'Chart' });
  var bTable = h('button', { type: 'button', 'aria-pressed': 'false', text: 'Table' });
  var drawnWidth = 0;
  function set(t) {
    bChart.setAttribute('aria-pressed', String(!t));
    bTable.setAttribute('aria-pressed', String(t));
    hideTip();
    while (body.firstChild) { body.removeChild(body.firstChild); }
    drawnWidth = Math.max(260, Math.floor(body.clientWidth || 560));
    body.appendChild(t ? dataTable() : chart(drawnWidth));
  }
  bChart.addEventListener('click', function () { set(false); });
  bTable.addEventListener('click', function () { set(true); });
  var title = 'Where people come from';
  var sub = 'People by agency (email domain), largest first. Agencies with fewer than ' + minPeople + ' people are combined.';
  var foot = named.length + ' ' + (named.length === 1 ? 'agency has' : 'agencies have') + ' at least ' + minPeople + ' people' +
    (otherOrgs ? '; ' + num(otherOrgs) + ' others are combined' : '') + '. Counts ' + num(covered) + ' of ' + num(D.total_people) +
    ' people: ' + (covered < D.total_people ? 'a cohort with too few people to break out by agency is left out, so that no one can be picked out.' : 'every cohort report.');
  root.appendChild(h('figure', { 'class': 'cr-card' },
    h('header', {}, h('div', {}, h('h3', { text: title }), h('p', { text: sub })),
      h('div', { 'class': 'cr-tools', role: 'group', 'aria-label': title + ' view' }, bChart, bTable)),
    body, h('p', { 'class': 'cr-foot', text: foot })));
  set(false);

  var resizeTimer = null;
  window.addEventListener('resize', function () {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(function () {
      if (bChart.getAttribute('aria-pressed') === 'true' && Math.abs(body.clientWidth - drawnWidth) > 24) { set(false); }
    }, 150);
  });
  root.setAttribute('data-ready', 'true');
})();
