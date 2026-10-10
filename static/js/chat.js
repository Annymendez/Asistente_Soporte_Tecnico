const $mensajes = document.getElementById("mensajes");
const $formPregunta = document.getElementById("form-pregunta");
const $inputPregunta = document.getElementById("input-pregunta");
const $btnEnviar = document.getElementById("btn-enviar");
const $badgeTurnos = document.getElementById("badge-turnos");
const $btnNuevoChat = document.getElementById("btn-nuevo-chat");
const $listaFuentes = document.getElementById("lista-fuentes");
const $estadoIndexacion = document.getElementById("estado-indexacion");
const $formUpload = document.getElementById("form-upload");
const $inputArchivos = document.getElementById("input-archivos");
const $inputApiKey = document.getElementById("input-api-key");
const $inputK = document.getElementById("input-k");
const $btnGuardarConfig = document.getElementById("btn-guardar-config");
const $plantillaRespuesta = document.getElementById("plantilla-respuesta");

let turnos = 0;
let sondeoActivo = null;

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.textContent === s ? s.replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c])) : s;
}

function agregarMensajeUsuario(texto) {
  const div = document.createElement("div");
  div.className = "mensaje-usuario";
  div.textContent = texto;
  $mensajes.appendChild(div);
  $mensajes.scrollTop = $mensajes.scrollHeight;
}

function agregarMensajeError(texto) {
  const div = document.createElement("div");
  div.className = "mensaje-error";
  div.textContent = texto;
  $mensajes.appendChild(div);
  $mensajes.scrollTop = $mensajes.scrollHeight;
}

function agregarRespuesta(data) {
  const nodo = $plantillaRespuesta.content.cloneNode(true);
  const tarjeta = nodo.querySelector(".tarjeta-respuesta");
  const r = data.respuesta;

  if (!r) {
    // El modelo no devolvió un JSON válido: mostramos el texto crudo tal cual.
    tarjeta.querySelector(".urgencia-badge").remove();
    tarjeta.querySelector(".diagnostico").textContent =
      "No se pudo interpretar la respuesta del modelo como JSON. Respuesta cruda:";
    tarjeta.querySelector(".pasos").innerHTML =
      `<li>${escapeHtml(data.respuesta_cruda || "(sin contenido)")}</li>`;
    tarjeta.querySelector(".fuente-campo").remove();
    tarjeta.querySelector(".alerta-humano").hidden = false;
  } else {
    const badge = tarjeta.querySelector(".urgencia-badge");
    badge.textContent = `Urgencia: ${r.nivel_urgencia || "?"}`;
    badge.classList.add((r.nivel_urgencia || "bajo").toLowerCase());

    tarjeta.querySelector(".diagnostico").textContent = r.diagnostico || "";

    const $pasos = tarjeta.querySelector(".pasos");
    (r.pasos_solucion || []).forEach((p) => {
      const li = document.createElement("li");
      li.textContent = p;
      $pasos.appendChild(li);
    });

    tarjeta.querySelector(".fuente").textContent = r.fuente || "No especificada";

    if (r.requiere_soporte_humano) {
      tarjeta.querySelector(".alerta-humano").hidden = false;
    } else {
      tarjeta.querySelector(".alerta-humano").remove();
    }
  }

  tarjeta.querySelector(".debug-fuentes").textContent = data.fuentes || "(sin fragmentos)";
  tarjeta.querySelector(".debug-prompt").textContent = data.prompt_debug || "";

  const $toggle = tarjeta.querySelector(".toggle-debug");
  const $panel = tarjeta.querySelector(".debug-panel");
  $toggle.addEventListener("click", () => {
    $panel.hidden = !$panel.hidden;
    $toggle.textContent = $panel.hidden
      ? "Ver fragmentos y prompt enviado al modelo ▾"
      : "Ocultar fragmentos y prompt enviado al modelo ▴";
  });

  $mensajes.appendChild(nodo);
  $mensajes.scrollTop = $mensajes.scrollHeight;
}

async function enviarPregunta(pregunta) {
  $btnEnviar.disabled = true;
  try {
    const resp = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pregunta }),
    });
    const data = await resp.json();
    if (!resp.ok) {
      agregarMensajeError(data.error || "Ocurrió un error al consultar al asistente.");
      return;
    }
    agregarRespuesta(data);
    turnos = data.num_turno || turnos + 1;
    $badgeTurnos.textContent = `${turnos} turno${turnos === 1 ? "" : "s"}`;
  } catch (e) {
    agregarMensajeError("No se pudo conectar con el servidor: " + e.message);
  } finally {
    $btnEnviar.disabled = false;
  }
}

$formPregunta.addEventListener("submit", (ev) => {
  ev.preventDefault();
  const texto = $inputPregunta.value.trim();
  if (!texto) return;
  agregarMensajeUsuario(texto);
  $inputPregunta.value = "";
  enviarPregunta(texto);
});

$inputPregunta.addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && !ev.shiftKey) {
    ev.preventDefault();
    $formPregunta.requestSubmit();
  }
});

$btnNuevoChat.addEventListener("click", async () => {
  await fetch("/api/reset_chat", { method: "POST" });
  turnos = 0;
  $badgeTurnos.textContent = "0 turnos";
  $mensajes.innerHTML = `<div class="mensaje-sistema">Nueva conversación iniciada. El historial anterior ya no se usará para las próximas preguntas.</div>`;
});

$formUpload.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const archivos = $inputArchivos.files;
  if (!archivos.length) return;
  const fd = new FormData();
  for (const f of archivos) fd.append("archivos", f);

  $estadoIndexacion.textContent = "Subiendo…";
  const resp = await fetch("/api/upload", { method: "POST", body: fd });
  const data = await resp.json();
  if (!resp.ok) {
    $estadoIndexacion.textContent = "Error: " + (data.error || "no se pudo subir.");
    return;
  }
  $inputArchivos.value = "";
  if (data.rechazados && data.rechazados.length) {
    $estadoIndexacion.textContent = `Formato no soportado: ${data.rechazados.join(", ")}`;
  }
  iniciarSondeoEstado();
});

$btnGuardarConfig.addEventListener("click", async () => {
  await fetch("/api/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      groq_api_key: $inputApiKey.value.trim(),
      k: parseInt($inputK.value, 10) || 6,
    }),
  });
  $estadoIndexacion.textContent = "Configuración guardada.";
  setTimeout(() => { if ($estadoIndexacion.textContent === "Configuración guardada.") $estadoIndexacion.textContent = ""; }, 2500);
});

function actualizarListaFuentes(fuentes) {
  $listaFuentes.innerHTML = "";
  if (!fuentes || !fuentes.length) {
    $listaFuentes.innerHTML = `<li class="vacio">Sin manuales todavía.</li>`;
    return;
  }
  for (const f of fuentes) {
    const li = document.createElement("li");
    li.textContent = f;
    $listaFuentes.appendChild(li);
  }
}

async function consultarEstado() {
  try {
    const resp = await fetch("/api/status");
    const data = await resp.json();
    actualizarListaFuentes(data.fuentes);

    if (data.indexando) {
      const pct = Math.round((data.progreso || 0) * 100);
      $estadoIndexacion.textContent = `${data.mensaje || "Indexando…"} (${pct}%)`;
    } else if (data.error) {
      $estadoIndexacion.textContent = "Error: " + data.error;
    } else if (data.mensaje) {
      $estadoIndexacion.textContent = data.mensaje;
    }
    return data.indexando;
  } catch {
    return false;
  }
}

function iniciarSondeoEstado() {
  if (sondeoActivo) return; // ya está corriendo, no duplicar el intervalo
  sondeoActivo = setInterval(consultarEstado, 1500);
}

// Autoajuste de altura del textarea
$inputPregunta.addEventListener("input", () => {
  $inputPregunta.style.height = "auto";
  $inputPregunta.style.height = Math.min($inputPregunta.scrollHeight, 120) + "px";
});

// Se consulta el estado de forma continua mientras la página esté abierta
// (y no solo cuando el propio navegador disparó la indexación), para no
// perderse el progreso de una indexación que ya estaba en curso al cargar
// (ej. el bootstrap automático al reiniciarse el servicio en Render).
consultarEstado();
iniciarSondeoEstado();