(function () {
  function currentPage() {
    const path = window.location.pathname;

    if (path.endsWith("/admin.html")) return "admin";
    if (path.endsWith("/graph.html")) return "graph";
    if (path.endsWith("/agents.html")) return "agents";

    return "chat";
  }

  function pageLabel(page) {
    if (page === "admin") return "Admin";
    if (page === "graph") return "Graph";
    if (page === "agents") return "Agent Studio";
    return "Chat";
  }

  function activeClass(page, target) {
    return page === target ? " active" : "";
  }

  function installShellNav() {
    if (document.getElementById("imShellNav")) {
      return;
    }

    const page = currentPage();

    document.body.classList.add("im-shell-ready");
    document.body.classList.add("im-shell-page-" + page);

    if (document.querySelector(".app")) {
      document.body.classList.add("im-has-app");
    } else {
      document.body.classList.add("im-no-app");
    }

    const nav = document.createElement("div");
    nav.id = "imShellNav";
    nav.className = "im-shell-nav";
    nav.innerHTML = `
      <div class="im-shell-brand">
        <div class="im-shell-brand-main">Infra<span>RAG</span></div>
        <div class="im-shell-brand-sub">Private RAG + Agent Studio</div>
      </div>

      <div class="im-shell-links" aria-label="InfraRAG navigation">
        <a class="im-shell-link${activeClass(page, "chat")}" href="/">Chat</a>
        <a class="im-shell-link${activeClass(page, "admin")}" href="/admin.html">Admin</a>
        <a class="im-shell-link${activeClass(page, "graph")}" href="/graph.html">Graph</a>
        <a class="im-shell-link${activeClass(page, "agents")}" href="/agents.html">Agents</a>
      </div>

      <div class="im-shell-spacer"></div>
      <div class="im-shell-page-label">${pageLabel(page)}</div>
    `;

    document.body.prepend(nav);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", installShellNav);
  } else {
    installShellNav();
  }
})();
