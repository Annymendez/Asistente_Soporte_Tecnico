/* Asistente RAG — lógica del chat (Instagram Direct) */
(() => {
  "use strict";

  const $ = (s) => document.querySelector(s);
  const el = {
    chat: $("#chat"), messages: $("#messages"), input: $("#input"), send: $("#send"),
    attach: $("#attach"), file: $("#file"), docs: $("#docs"), status: $("#status"),
    gear: $("#gear"), settings: $("#settings"), sKey: $("#s-key"), sModel: $("#s-model"),
    sK: $("#s-k"), sSave: $("#s-save"), sClose: $("#s-close"), sReset: $("#s-reset"), sMsg: $("#s-msg"),
  };
  const ROBOT = $("#robot").innerHTML;
  $("#avatar-inner").innerHTML = ROBOT;

  const st = {
    ready: false, hasKey: false, model: "", k: 10,
    indexing: false, asking: false,
    handled: 0, first: true, bubble: null, timer: null, typing: null,
  };

  /* ── utilidades ─────────────────────────────────────────── */
  async function api(url, opts = {}) {
    const r = await fetch(url, opts);
    let data = {};
    try { data = await r.json(); } catch (_) { /* sin cuerpo JSON */ }
    if (!r.ok) {
      const e = new Error(data.error || `Error ${r.status}`);
      e.status = r.status; e.code = data.code;
      throw e;
    }
    return data;
  }
  const scroll = () => { el.messages.scrollTop = el.messages.scrollHeight; };
  const div = (cls) => { const d = document.createElement("div"); if (cls) d.className = cls; return d; };

  /* ── Markdown mínimo (escapa HTML antes de dar formato) ─── */
  const esc = (s) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const inline = (s) => s
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*([^*\s][^*]*?)\*(?!\w)/g, "$1<em>$2</em>");

  function table(rows) {
    const cells = (r) => r.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
    const hasHead = rows.length > 1 && /^\s*\|?\s*:?-{2,}/.test(rows[1]);
    const head = hasHead ? cells(rows[0]) : null;
    const body = (hasHead ? rows.slice(2) : rows).map(cells);
    return '<div class="tbl"><table>' +
      (head ? "<thead><tr>" + head.map((c) => `<th>${inline(esc(c))}</th>`).join("") + "</tr></thead>" : "") +
      "<tbody>" + body.map((r) => "<tr>" + r.map((c) => `<td>${inline(esc(c))}</td>`).join("") + "</tr>").join("") + "</tbody></table></div>";
  }

  function md(src) {
    const math = [];   // protegemos las fórmulas para que el formato no las dañe
    src = src.replace(/\\\[[\s\S]+?\\\]|\\\([\s\S]+?\\\)|\$\$[\s\S]+?\$\$/g, (m) => { math.push(m); return `@@M${math.length - 1}@@`; });
    const lines = src.replace(/\r/g, "").split("\n");
    const isUl = (l) => /^\s*[-*•]\s+/.test(l), isOl = (l) => /^\s*\d+[.)]\s+/.test(l), isTb = (l) => /^\s*\|.*\|\s*$/.test(l);
    let html = "", i = 0;
    while (i < lines.length) {
      const line = lines[i];
      if (/^\s*$/.test(line)) { i++; continue; }
      if (isTb(line)) { const rows = []; while (i < lines.length && isTb(lines[i])) rows.push(lines[i++]); html += table(rows); continue; }
      if (isUl(line)) { const it = []; while (i < lines.length && isUl(lines[i])) it.push(lines[i++].replace(/^\s*[-*•]\s+/, "")); html += "<ul>" + it.map((t) => `<li>${inline(esc(t))}</li>`).join("") + "</ul>"; continue; }
      if (isOl(line)) { const it = []; while (i < lines.length && isOl(lines[i])) it.push(lines[i++].replace(/^\s*\d+[.)]\s+/, "")); html += "<ol>" + it.map((t) => `<li>${inline(esc(t))}</li>`).join("") + "</ol>"; continue; }
      const h = line.match(/^\s{0,3}#{1,6}\s+(.*)/);
      if (h) { html += `<p class="h">${inline(esc(h[1]))}</p>`; i++; continue; }
      const para = [lines[i++]];
      while (i < lines.length && !/^\s*$/.test(lines[i]) && !isTb(lines[i]) && !isUl(lines[i]) && !isOl(lines[i]) && !/^\s{0,3}#{1,6}\s/.test(lines[i])) para.push(lines[i++]);
      html += "<p>" + para.map((t) => inline(esc(t))).join("<br>") + "</p>";
    }
    return html.replace(/@@M(\d+)@@/g, (_, k) => esc(math[k]));
  }

  function typeset(node) {
    if (!window.renderMathInElement) return;
    try {
      window.renderMathInElement(node, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "\\[", right: "\\]", display: true },
          { left: "\\(", right: "\\)", display: false },
        ],
        throwOnError: false,
      });
    } catch (_) { /* si KaTeX falla, se deja el texto tal cual */ }
  }

  /* ── filas de chat ──────────────────────────────────────── */
  function makeRow(role) {
    const row = div("row " + role);
    if (role === "bot") { const av = div("av-sm"); av.innerHTML = ROBOT; row.appendChild(av); }
    const col = div("col");
    const bubble = div("bubble");
    col.appendChild(bubble); row.appendChild(col);
    el.messages.appendChild(row);
    return { row, col, bubble };
  }

  function addUser(text) {
    const r = makeRow("user");
    r.bubble.textContent = text;
    scroll();
  }

  const URGENCIA = {
    bajo: { label: "Urgencia baja", color: "#3fb97a" },
    medio: { label: "Urgencia media", color: "#e2ab3d" },
    alto: { label: "Urgencia alta", color: "#e5484d" },
  };

  function soporteHTML(r) {
    const u = URGENCIA[(r.nivel_urgencia || "").toLowerCase()] || { label: r.nivel_urgencia || "—", color: "#8e8e8e" };
    const pasos = Array.isArray(r.pasos_solucion) ? r.pasos_solucion : [];
    let html = `<p class="h">Diagnóstico</p><p>${esc(r.diagnostico || "—")}</p>`;
    if (pasos.length) {
      html += `<p class="h" style="margin-top:10px">Pasos a seguir</p><ol>${pasos.map((p) => `<li>${esc(p)}</li>`).join("")}</ol>`;
    }
    html += `<p style="margin-top:10px"><span class="badge" style="background:${u.color}22;color:${u.color};border-color:${u.color}55">${esc(u.label)}</span></p>`;
    if (r.fuente) html += `<p style="margin-top:6px;color:var(--muted);font-size:13px">Fuente citada: ${esc(r.fuente)}</p>`;
    if (r.requiere_soporte_humano) {
      html += `<p class="warn">⚠️ Se recomienda escalar este caso a soporte humano.</p>`;
    }
    return html;
  }

  function ctxDetails(r) {
    const d = document.createElement("details");
    d.className = "ctx";
    const s = document.createElement("summary");
    s.textContent = "Ver fragmentos y mensajes enviados al modelo";
    d.appendChild(s);
    const h1 = document.createElement("h4");
    h1.textContent = `Fragmentos recuperados (${r.fragmentos.length}) · ~${(r.prompt || "").length / 4 | 0} tokens aprox. de contexto`;
    d.appendChild(h1);
    r.fragmentos.forEach((f, i) => {
      const box = div("frag");
      const b = document.createElement("b");
      b.textContent = `[${i + 1}] ${f.fuente} — Pág. ${f.pagina}`;
      const pre = document.createElement("pre");
      pre.textContent = f.texto;
      box.append(b, pre);
      d.appendChild(box);
    });
    const h2 = document.createElement("h4");
    h2.textContent = "Mensajes enviados al modelo (system + few-shot + turno real)";
    const pre = document.createElement("pre");
    pre.textContent = r.prompt;
    d.append(h2, pre);
    return d;
  }

  function addBot(text, o = {}) {
    const r = makeRow("bot");
    if (o.error) r.bubble.classList.add("error");
    if (o.html) { r.bubble.innerHTML = o.html; }
    else if (o.md) { r.bubble.innerHTML = md(text); typeset(r.bubble); }
    else { r.bubble.classList.add("plain"); r.bubble.textContent = text; }
    if (o.sources) { const m = div("meta"); m.textContent = "Fuentes: " + o.sources; r.col.appendChild(m); }
    if (o.ctx) r.col.appendChild(ctxDetails(o.ctx));
    if (o.quick) {
      const q = div("quick");
      o.quick.forEach(({ label, run }) => {
        const b = document.createElement("button");
        b.textContent = label;
        b.onclick = () => { q.remove(); addUser(label); run(); };
        q.appendChild(b);
      });
      r.col.appendChild(q);
    }
    scroll();
    return r;
  }

  function showTyping() { const r = makeRow("bot"); r.bubble.classList.add("typing"); r.bubble.innerHTML = "<span></span><span></span><span></span>"; st.typing = r.row; scroll(); }
  function hideTyping() { if (st.typing) { st.typing.remove(); st.typing = null; } }

  /* ── documentos ─────────────────────────────────────────── */
  function renderDocs(docs) {
    el.docs.replaceChildren();
    docs.forEach((name) => {
      const chip = div("doc-chip");
      const dot = document.createElement("i"); dot.textContent = "PDF";
      const t = document.createElement("span"); t.textContent = name; t.title = name;
      chip.append(dot, t);
      el.docs.appendChild(chip);
    });
  }

  /* ── estado / interfaz ──────────────────────────────────── */
  function refreshUI() {
    const busy = st.indexing || st.asking;
    el.status.textContent = st.asking ? "Escribiendo…" : st.indexing ? "Indexando…" : "Activo(a) ahora";
    el.send.disabled = busy || !el.input.value.trim();
    el.attach.disabled = st.indexing;
  }

  function makeProgress() {
    document.querySelectorAll(".quick").forEach((q) => q.remove());
    const r = makeRow("bot");
    r.bubble.innerHTML = '<div class="ptxt"></div><div class="bar"><i></i></div>';
    scroll();
    return r;
  }
  function updateProgress(t) {
    const b = st.bubble.bubble;
    b.querySelector(".ptxt").textContent = `${t.message}  ${Math.round(t.progress * 100)}%`;
    b.querySelector(".bar > i").style.width = `${t.progress * 100}%`;
  }

  function finishTask(t) {
    let text, error = false;
    if (t.state === "error") {
      text = "Algo salió mal: " + (t.error || "error desconocido"); error = true;
    } else {
      const r = t.result || {};
      if (r.stats) {
        text = `Listo. Indexé ${r.stats.fragmentos} fragmentos (${r.stats.paginas} páginas). Ya puedes preguntarme sobre el contenido.`;
        if (r.stats.avisos && r.stats.avisos.length) text += "\n\nAvisos:\n" + r.stats.avisos.map((a) => "• " + a).join("\n");
      } else if (r.loaded > 0) {
        text = `Recuperé mi memoria anterior: ${r.loaded} fragmentos de ${r.docs.join(", ")}. ¿Qué quieres saber?`;
      } else {
        text = "Listo.";
      }
    }
    if (st.bubble) {
      const b = st.bubble.bubble;
      b.classList.add("plain"); b.classList.toggle("error", error); b.textContent = text;
      st.bubble = null;
    } else {
      addBot(text, { error });
    }
    st.handled = t.id;
    scroll();
  }

  function greet(s) {
    const t = s.task;
    const running = t.state === "running";
    addBot("¡Hola! 👋 Soy tu asistente de soporte técnico. Agrega el manual (PDF) de un producto con el botón + y cuéntame el problema que tienes.", {
      quick: !s.ready && !running ? [
        { label: "Agregar PDF", run: () => el.file.click() },
        { label: "Indexar carpeta manuales/", run: syncFolder },
      ] : null,
    });
    if (!running) st.handled = t.id;
    if (s.ready && !running) addBot(`Tengo ${s.count} fragmentos de ${s.docs.join(", ")}. Cuéntame el problema que tienes.`);
  }

  function applyStatus(s) {
    st.ready = s.ready; st.hasKey = s.has_key; st.model = s.model; st.k = s.k;
    renderDocs(s.docs);
    if (st.first) { st.first = false; greet(s); }
    const t = s.task;
    if (t.state === "running") {
      if (!st.bubble) st.bubble = makeProgress();
      updateProgress(t);
      st.indexing = true;
    } else {
      if (t.id > st.handled) finishTask(t);
      st.indexing = false;
    }
    refreshUI();
  }

  async function tick() {
    clearTimeout(st.timer);
    try {
      const s = await api("/api/status");
      applyStatus(s);
      if (s.task.state === "running") st.timer = setTimeout(tick, 700);
    } catch (_) {
      st.timer = setTimeout(tick, 2000);
    }
  }

  /* ── acciones ───────────────────────────────────────────── */
  async function upload(files) {
    const pdfs = [...files].filter((f) => /\.pdf$/i.test(f.name));
    if (!pdfs.length) { addBot("Solo puedo leer manuales en PDF.", { error: true }); return; }
    addUser(pdfs.map((f) => "📄 " + f.name).join("\n"));
    const fd = new FormData();
    pdfs.forEach((f) => fd.append("files", f));
    try {
      const r = await api("/api/upload", { method: "POST", body: fd });
      if (r.omitidos && r.omitidos.length) addBot("Omití: " + r.omitidos.join("; "));
    } catch (e) {
      addBot(e.message, { error: true });
    }
    tick();
  }

  async function syncFolder() {
    try {
      const r = await api("/api/sync", { method: "POST" });
      if (!r.started) addBot(r.message || "No hay manuales nuevos en la carpeta manuales/.");
    } catch (e) {
      addBot(e.message, { error: true });
    }
    tick();
  }

  async function sendMessage() {
    const q = el.input.value.trim();
    if (!q || st.asking || st.indexing) return;
    el.input.value = "";
    addUser(q);
    if (!st.ready) { addBot("Todavía no tengo documentos. Toca el botón + y agrega un manual.", {
      quick: [{ label: "Indexar carpeta manuales/", run: syncFolder }] }); refreshUI(); return; }
    if (!st.hasKey) {
      addBot("Necesito tu API key de Groq. Pégala en Ajustes (el engranaje de arriba) o guárdala en el archivo .env.", { error: true });
      openSettings(); refreshUI(); return;
    }
    st.asking = true; refreshUI(); showTyping();
    try {
      const r = await api("/api/ask", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: q }),
      });
      hideTyping();
      if (r.respuesta) {
        const row = addBot("", { html: soporteHTML(r.respuesta), sources: r.fuentes, ctx: r });
        if (r.error_parseo) {
          const warn = div("meta");
          warn.style.color = "#e5484d";
          warn.textContent = "Aviso: " + r.error_parseo;
          row.col.appendChild(warn);
        }
      } else {
        addBot("No pude interpretar la respuesta del modelo como JSON" +
          (r.error_parseo ? " (" + r.error_parseo + ")" : "") + ". Esto es lo que devolvió:\n\n" + r.respuesta_cruda,
          { error: true, sources: r.fuentes, ctx: r });
      }
    } catch (e) {
      hideTyping();
      if (e.code === "no_key") { st.hasKey = false; openSettings(); }
      addBot("Algo salió mal: " + e.message, { error: true });
    } finally {
      st.asking = false; refreshUI(); el.input.focus();
    }
  }

  /* ── ajustes ────────────────────────────────────────────── */
  let armTimer = null;
  function disarmReset() { clearTimeout(armTimer); el.sReset.classList.remove("armed"); el.sReset.textContent = "Borrar índice"; }

  function openSettings() {
    el.sKey.value = "";
    el.sKey.placeholder = st.hasKey ? "•••••• (ya guardada)" : "gsk_…";
    if (![...el.sModel.options].some((o) => o.value === st.model)) el.sModel.add(new Option(st.model, st.model));
    el.sModel.value = st.model;
    el.sK.value = st.k;
    el.sMsg.textContent = "";
    disarmReset();
    el.settings.hidden = false;
  }
  const closeSettings = () => { el.settings.hidden = true; disarmReset(); el.input.focus(); };

  el.gear.onclick = openSettings;
  el.sClose.onclick = closeSettings;
  el.settings.addEventListener("click", (e) => { if (e.target === el.settings) closeSettings(); });
  el.sSave.onclick = async () => {
    try {
      const r = await api("/api/settings", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ api_key: el.sKey.value, model: el.sModel.value, k: el.sK.value }),
      });
      st.hasKey = r.has_key; st.model = r.model; st.k = r.k;
      closeSettings();
    } catch (e) { el.sMsg.style.color = "#ff8a95"; el.sMsg.textContent = e.message; }
  };
  el.sReset.onclick = async () => {   // sin confirm(): el Simple Browser de VS Code bloquea los diálogos
    if (!el.sReset.classList.contains("armed")) {
      el.sReset.classList.add("armed"); el.sReset.textContent = "¿Seguro? Toca otra vez";
      armTimer = setTimeout(disarmReset, 3500); return;
    }
    try {
      await api("/api/reset", { method: "POST" });
      closeSettings();
      addBot("Borré el índice. Agrega un PDF para empezar de nuevo.");
      tick();
    } catch (e) { el.sMsg.style.color = "#ff8a95"; el.sMsg.textContent = e.message; disarmReset(); }
  };

  /* ── eventos ────────────────────────────────────────────── */
  el.send.onclick = sendMessage;
  el.input.addEventListener("input", refreshUI);
  el.input.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.isComposing) { e.preventDefault(); sendMessage(); } });
  el.attach.onclick = () => el.file.click();
  el.file.onchange = () => { if (el.file.files.length) upload(el.file.files); el.file.value = ""; };

  let dragDepth = 0;
  el.chat.addEventListener("dragenter", (e) => { e.preventDefault(); dragDepth++; el.chat.classList.add("drag"); });
  el.chat.addEventListener("dragover", (e) => e.preventDefault());
  el.chat.addEventListener("dragleave", () => { if (--dragDepth <= 0) { dragDepth = 0; el.chat.classList.remove("drag"); } });
  el.chat.addEventListener("drop", (e) => {
    e.preventDefault(); dragDepth = 0; el.chat.classList.remove("drag");
    if (e.dataTransfer && e.dataTransfer.files.length) upload(e.dataTransfer.files);
  });

  refreshUI();
  tick();
  el.input.focus();
})();
