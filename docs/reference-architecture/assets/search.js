/* Client-side component search over window.KG_SEARCH: [id, label, layer, aliases, definition]. */
(function () {
  var input = document.getElementById("search");
  var list = document.getElementById("results");
  var data = window.KG_SEARCH || [];
  if (!input || !list) return;
  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function score(row, q) {
    var label = row[1].toLowerCase();
    if (label === q) return 0;
    if (label.indexOf(q) === 0) return 1;
    if (label.indexOf(q) >= 0) return 2;
    if (row[3].toLowerCase().indexOf(q) >= 0) return 3;
    if (row[4].toLowerCase().indexOf(q) >= 0) return 4;
    return -1;
  }
  input.addEventListener("input", function () {
    var q = input.value.trim().toLowerCase();
    if (q.length < 2) { list.innerHTML = ""; return; }
    var hits = [];
    for (var i = 0; i < data.length; i++) {
      var s = score(data[i], q);
      if (s >= 0) hits.push([s, data[i]]);
    }
    hits.sort(function (a, b) { return a[0] - b[0] || a[1][1].localeCompare(b[1][1]); });
    list.innerHTML = hits.slice(0, 30).map(function (h) {
      var r = h[1];
      return '<li><a href="c/' + esc(r[0]) + '.html">' + esc(r[1]) + '</a> <span class="muted small">' +
        esc(r[2]) + '</span><span class="def">' + esc(r[4].slice(0, 180)) + "</span></li>";
    }).join("") + (hits.length > 30 ? '<li class="muted">…and ' + (hits.length - 30) + " more</li>" : "") +
      (hits.length === 0 ? '<li class="muted">No matches.</li>' : "");
  });
})();
