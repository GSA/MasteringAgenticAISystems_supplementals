/* Theme: follows the OS by default; the header toggle stores a per-viewer choice. */
(function () {
  var root = document.documentElement;
  try {
    var saved = localStorage.getItem("ara-theme");
    if (saved === "light" || saved === "dark") root.dataset.theme = saved;
  } catch (e) { /* storage unavailable: follow the OS */ }
  function isDark() {
    if (root.dataset.theme) return root.dataset.theme === "dark";
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  }
  document.addEventListener("DOMContentLoaded", function () {
    var btn = document.querySelector(".theme-toggle");
    if (!btn) return;
    btn.addEventListener("click", function () {
      var next = isDark() ? "light" : "dark";
      root.dataset.theme = next;
      try { localStorage.setItem("ara-theme", next); } catch (e) { /* ignore */ }
      window.dispatchEvent(new CustomEvent("ara-theme-change"));
    });
  });
  window.araIsDark = isDark;
})();
