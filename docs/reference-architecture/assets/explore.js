/* Interactive explorer: sigma.js (WebGL) over window.KG_GRAPH with a precomputed layout. */
(function () {
  var D = window.KG_GRAPH;
  var container = document.getElementById("graph");
  var panel = document.getElementById("panel");
  if (!D || !container || typeof Sigma === "undefined" || typeof graphology === "undefined") return;

  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }
  var colors = {};
  function readColors() {
    colors.groups = D.groups.map(function (_, i) { return cssVar("--g" + (i + 1)); });
    colors.ink = cssVar("--ink");
    colors.muted = cssVar("--hair");
    colors.edge = cssVar("--edge");
    colors.edgeFocus = cssVar("--ink-2");
  }
  readColors();

  // ---- graph model -------------------------------------------------------
  var graph = new graphology.Graph({ type: "directed", multi: false, allowSelfLoops: false });
  var byLabel = {};
  D.nodes.forEach(function (n) {
    graph.addNode(n.id, { x: n.x, y: n.y, size: n.s, label: n.l, g: n.g });
    byLabel[n.l.toLowerCase()] = n.id;
  });
  var nodeInfo = {};
  D.nodes.forEach(function (n) { nodeInfo[n.id] = n; });
  var outAdj = {}, inAdj = {};
  D.edges.forEach(function (e) {
    var s = D.nodes[e[0]].id, t = D.nodes[e[1]].id;
    var cats = {};
    e[2].forEach(function (r) { cats[D.relations[r].c] = true; });
    graph.addEdge(s, t, { size: 0.6, rels: e[2], cats: Object.keys(cats) });
    (outAdj[s] = outAdj[s] || []).push([t, e[2]]);
    (inAdj[t] = inAdj[t] || []).push([s, e[2]]);
  });

  // ---- state -------------------------------------------------------------
  var state = { hovered: null, selected: null, focus: null, hiddenGroups: {}, cats: {} };
  document.querySelectorAll(".cat-toggle input").forEach(function (cb) {
    state.cats[cb.value] = cb.checked;
  });
  function focusSet(node) {
    var set = {};
    if (!node) return null;
    set[node] = true;
    graph.forEachNeighbor(node, function (nb) { set[nb] = true; });
    return set;
  }

  var renderer = new Sigma(graph, container, {
    renderEdgeLabels: false,
    labelRenderedSizeThreshold: 10,
    labelDensity: 0.45,
    labelGridCellSize: 140,
    labelFont: "system-ui, -apple-system, 'Segoe UI', sans-serif",
    labelSize: 12,
    labelColor: { color: colors.ink },
    zIndex: true,
    stagePadding: 40,
    minCameraRatio: 0.03,
    maxCameraRatio: 4,
    nodeReducer: function (node, data) {
      var res = Object.assign({}, data);
      res.color = colors.groups[data.g];
      if (state.hiddenGroups[data.g]) { res.hidden = true; return res; }
      if (state.focus) {
        if (state.focus[node]) {
          var anchorNode = state.hovered || state.selected;
          res.zIndex = 2;
          // Force labels only for small neighbourhoods; hubs rely on sigma's label grid.
          res.forceLabel = node === anchorNode || state.focusSize <= 25;
          if (node === anchorNode) res.highlighted = true;
        } else {
          res.color = colors.muted;
          res.label = "";
          res.zIndex = 0;
        }
      }
      return res;
    },
    edgeReducer: function (edge, data) {
      var res = Object.assign({}, data);
      var ends = graph.extremities(edge);
      var g0 = graph.getNodeAttribute(ends[0], "g"), g1 = graph.getNodeAttribute(ends[1], "g");
      var visibleCat = data.cats.some(function (c) { return state.cats[c]; });
      if (state.hiddenGroups[g0] || state.hiddenGroups[g1] || !visibleCat) { res.hidden = true; return res; }
      var anchor = state.hovered || state.selected;
      if (anchor) {
        if (ends[0] === anchor || ends[1] === anchor) {
          res.color = colors.edgeFocus;
          res.size = 1.2;
          res.zIndex = 1;
        } else {
          res.hidden = true;
        }
      } else {
        res.color = colors.edge;
      }
      return res;
    }
  });

  function refresh() {
    state.focus = focusSet(state.hovered || state.selected);
    state.focusSize = state.focus ? Object.keys(state.focus).length : 0;
    renderer.refresh({ skipIndexation: true });
  }

  // ---- details panel -----------------------------------------------------
  function relList(adj, outgoing) {
    var groups = {};
    (adj || []).forEach(function (pair) {
      pair[1].forEach(function (r) {
        var rel = D.relations[r];
        if (!state.cats[rel.c]) return;
        var label = outgoing || rel.n === "alternativeTo" || rel.n === "excludes" ? rel.l : rel.i;
        (groups[label] = groups[label] || []).push(pair[0]);
      });
    });
    return Object.keys(groups).sort().map(function (label) {
      var items = groups[label].sort(function (a, b) {
        return nodeInfo[a].l.localeCompare(nodeInfo[b].l);
      }).map(function (id) {
        return '<li><a href="#' + esc(id) + '" data-node="' + esc(id) + '">' + esc(nodeInfo[id].l) + "</a></li>";
      }).join("");
      return '<div class="reltype">' + esc(label) + "</div><ul>" + items + "</ul>";
    }).join("");
  }
  function showPanel(id) {
    if (!id) {
      panel.innerHTML = '<p class="muted">Hover a node to highlight its neighbourhood; click it for details.</p>';
      return;
    }
    var n = nodeInfo[id];
    panel.innerHTML =
      '<p class="crumbs">' + esc(n.ly) + " · " + esc(n.k) + (n.a ? " · abstract" : "") + "</p>" +
      '<h2><span class="dot g' + (n.g + 1) + '" aria-hidden="true"></span>' + esc(n.l) + "</h2>" +
      "<p>" + esc(n.d) + "</p>" +
      '<p><a href="c/' + esc(id) + '.html">Open catalog page →</a></p>' +
      "<h3>Outgoing</h3>" + (relList(outAdj[id], true) || '<p class="muted small">None in the active categories.</p>') +
      "<h3>Incoming</h3>" + (relList(inAdj[id], false) || '<p class="muted small">None in the active categories.</p>');
  }
  panel.addEventListener("click", function (ev) {
    var a = ev.target.closest("a[data-node]");
    if (a) { ev.preventDefault(); select(a.getAttribute("data-node"), true); }
  });

  function select(id, move) {
    if (id && !graph.hasNode(id)) return;
    state.selected = id;
    showPanel(id);
    refresh();
    if (id) {
      try { history.replaceState(null, "", "#" + id); } catch (e) { /* ignore */ }
      if (move) {
        var d = renderer.getNodeDisplayData(id);
        if (d) renderer.getCamera().animate({ x: d.x, y: d.y, ratio: 0.25 }, { duration: 500 });
      }
    }
  }

  // ---- events --------------------------------------------------------------
  renderer.on("enterNode", function (e) { state.hovered = e.node; refresh(); container.style.cursor = "pointer"; });
  renderer.on("leaveNode", function () { state.hovered = null; refresh(); container.style.cursor = ""; });
  renderer.on("clickNode", function (e) { select(e.node, false); });
  renderer.on("clickStage", function () { select(null, false); });

  document.querySelectorAll(".legend-item").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var g = Number(btn.getAttribute("data-group"));
      state.hiddenGroups[g] = !state.hiddenGroups[g];
      btn.setAttribute("aria-pressed", state.hiddenGroups[g] ? "false" : "true");
      refresh();
    });
  });
  document.querySelectorAll(".cat-toggle input").forEach(function (cb) {
    cb.addEventListener("change", function () {
      state.cats[cb.value] = cb.checked;
      if (state.selected) showPanel(state.selected);
      refresh();
    });
  });

  var find = document.getElementById("find");
  var names = document.getElementById("names");
  names.innerHTML = D.nodes.map(function (n) { return '<option value="' + esc(n.l) + '">'; }).join("");
  find.addEventListener("change", function () {
    var id = byLabel[find.value.trim().toLowerCase()];
    if (id) select(id, true);
  });
  document.getElementById("reset").addEventListener("click", function () {
    state.hiddenGroups = {};
    document.querySelectorAll(".legend-item").forEach(function (b) { b.setAttribute("aria-pressed", "true"); });
    select(null, false);
    renderer.getCamera().animatedReset({ duration: 400 });
  });

  function applyTheme() {
    readColors();
    renderer.setSetting("labelColor", { color: colors.ink });
    refresh();
  }
  window.addEventListener("ara-theme-change", applyTheme);
  if (window.matchMedia) {
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", applyTheme);
  }

  var start = decodeURIComponent((location.hash || "").slice(1));
  if (start && graph.hasNode(start)) setTimeout(function () { select(start, true); }, 50);
})();
