"use strict";
const $ = (s) => document.querySelector(s);
const el = (tag, attrs = {}, ...kids) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) k === "class" ? (n.className = v) : k.startsWith("on") ? n.addEventListener(k.slice(2), v) : n.setAttribute(k, v);
  for (const c of kids) n.append(c);
  return n;
};
const isTelugu = (s) => /[ఀ-౿]/.test(s);

const EXAMPLES = {
  "Welcome (mixed)": "[sfx:chime] నమస్కారం! Welcome to Vani, మీ AI voice assistant. [pause:400ms] Telugu లో మాట్లాడినా, English లో మాట్లాడినా, I understand both!",
  "Rainy-night story": "[bg:rain] ఒకప్పుడు, ఒక చిన్న గ్రామంలో, చీకటి రాత్రి వర్షం కురుస్తోంది. [pause:600ms] [sfx:thunder] అకస్మాత్తుగా, తలుపు దగ్గర చప్పుడు వినిపించింది. [sfx:knock] [pause:500ms] [voice:male] ఎవరు అక్కడ? [bg:off]",
  "Temple morning": "[bg:birds] [sfx:temple_bell] శుభోదయం! [pause:300ms] ఈ రోజు మీకు ఆనందంగా ఉండాలని కోరుకుంటున్నాను. [sfx:conch] [bg:off]",
  "Quiz show (English)": "[sfx:drumroll] And the winner of today's quiz is... [pause:700ms] Team Hyderabad! [sfx:applause] [sfx:fanfare]",
  "Festival announcement": "[bg:drum] సంక్రాంతి పండుగ శుభాకాంక్షలు! This year's celebration starts at 6 PM sharp. అందరూ రండి! [sfx:applause] [bg:off]",
};
const TAGS = ["[sfx:applause]", "[sfx:chime]", "[sfx:thunder]", "[bg:rain]", "[bg:off]", "[pause:500ms]", "[voice:male]", "[voice:female]", "[lang:te]", "[lang:en]"];

// ---------- shared voice settings
function voiceParams() {
  const p = {
    gender: $("#gender").value, language: $("#language").value, english_accent: $("#accent").value,
    rate: +$("#rate").value, pitch: +$("#pitch").value, format: $("#format").value,
  };
  if ($("#provider").value) p.provider = $("#provider").value;
  return p;
}
for (const [id, unit] of [["rate", "%"], ["pitch", "Hz"]]) {
  $("#" + id).addEventListener("input", (e) => ($("#" + id + "Out").textContent = (e.target.value > 0 ? "+" : "") + e.target.value + unit));
}

async function api(path, body) {
  const res = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {});
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail || data));
  return data;
}
const toUrl = (b64, mime) => URL.createObjectURL(new Blob([Uint8Array.from(atob(b64), (c) => c.charCodeAt(0))], { type: mime }));

let currentAudio = null;
function play(url) {
  if (currentAudio) currentAudio.pause();
  currentAudio = new Audio(url);
  currentAudio.play().catch(() => {});
  return currentAudio;
}
async function busy(btn, fn) {
  const label = btn.textContent;
  btn.disabled = true; btn.textContent = "Working…";
  try { return await fn(); } catch (e) { toast(e.message); } finally { btn.disabled = false; btn.textContent = label; }
}
function toast(msg) {
  const t = el("div", { class: "card", role: "alert", style: "position:fixed;bottom:16px;left:16px;right:16px;max-width:560px;margin:auto;border-color:var(--sfx);z-index:9" }, msg);
  document.body.append(t);
  setTimeout(() => t.remove(), 6000);
}

// ---------- tabs
document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === b));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === "tab-" + b.dataset.tab));
}));

// ---------- status
api("/api/health").then((h) => {
  const pill = (on, text) => el("span", { class: "pill " + (on ? "on" : "off") }, text);
  $("#status").append(
    pill(true, "Neural TTS"),
    pill(h.agent === "claude", h.agent === "claude" ? "Claude agent" : "Agent: offline mode"),
    pill(h.elevenlabs, "ElevenLabs"),
  );
}).catch(() => {});

// ---------- agent chat
const history = [];
function addMsg(role, text, extra) {
  const bubble = el("div", { class: "bubble" }, text);
  if (isTelugu(text)) bubble.setAttribute("lang", "te");
  if (extra) bubble.append(extra);
  const m = el("div", { class: "msg " + role }, bubble);
  $("#messages").append(m);
  $("#messages").scrollTop = $("#messages").scrollHeight;
  return bubble;
}
async function sendChat(text) {
  text = text.trim();
  if (!text) return;
  addMsg("user", text);
  const pending = addMsg("bot", "");
  pending.classList.add("typing");
  try {
    const r = await api("/api/agent/chat", { ...voiceParams(), format: "mp3", message: text, history });
    history.push({ role: "user", content: text }, { role: "assistant", content: r.reply });
    pending.classList.remove("typing");
    pending.textContent = r.transcript || r.reply;
    if (isTelugu(r.reply)) pending.setAttribute("lang", "te");
    const meta = el("span", { class: "meta" }, r.source === "claude" ? "Claude" : "offline mode");
    pending.append(meta);
    if (r.audio) {
      const url = toUrl(r.audio, r.mime);
      pending.append(el("button", { class: "replay", type: "button", onclick: () => play(url) }, "▶ Replay"));
      play(url);
    } else if (r.audio_error) {
      meta.textContent += " · audio unavailable: " + r.audio_error;
    }
  } catch (e) {
    pending.classList.remove("typing");
    pending.textContent = "⚠ " + e.message;
  }
}
$("#chatForm").addEventListener("submit", (e) => {
  e.preventDefault();
  const v = $("#chatInput").value;
  $("#chatInput").value = "";
  sendChat(v);
});

// Browser speech recognition (Chrome/Edge support te-IN and en-IN).
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
if (!SR) {
  $("#micBtn").title = "Speech recognition isn't supported in this browser; type instead.";
  $("#micBtn").disabled = true;
} else {
  let rec = null;
  $("#micBtn").addEventListener("click", () => {
    if (rec) { rec.stop(); return; }
    rec = new SR();
    rec.lang = $("#micLang").value;
    rec.interimResults = true;
    rec.onresult = (ev) => {
      const r = ev.results[ev.results.length - 1];
      $("#chatInput").value = r[0].transcript;
      if (r.isFinal) { const v = $("#chatInput").value; $("#chatInput").value = ""; sendChat(v); }
    };
    rec.onend = () => { $("#micBtn").classList.remove("listening"); rec = null; };
    rec.onerror = (e) => toast("Mic: " + e.error);
    if (currentAudio) currentAudio.pause();
    $("#micBtn").classList.add("listening");
    rec.start();
  });
}

// ---------- studio
for (const name of Object.keys(EXAMPLES)) $("#examples").append(el("option", { value: name }, name));
$("#examples").addEventListener("change", (e) => { if (e.target.value) $("#script").value = EXAMPLES[e.target.value]; });
$("#script").value = EXAMPLES["Welcome (mixed)"];
for (const tag of TAGS) {
  $("#tagChips").append(el("button", { class: "chip", type: "button", onclick: () => {
    const ta = $("#script");
    const pos = ta.selectionStart ?? ta.value.length;
    ta.value = ta.value.slice(0, pos) + " " + tag + " " + ta.value.slice(pos);
    ta.focus();
  } }, tag));
}

function showStudio(r) {
  $("#studioOut").hidden = false;
  const url = toUrl(r.audio, r.mime);
  $("#studioAudio").src = url;
  $("#studioAudio").play().catch(() => {});
  $("#studioDl").href = url;
  $("#studioDl").download = "vani." + (r.mime === "audio/mpeg" ? "mp3" : "wav");
  $("#studioDur").textContent = r.duration.toFixed(1) + " s";
  const tl = $("#timeline"); tl.replaceChildren();
  const segList = $("#segments"); segList.replaceChildren();
  const total = r.duration || 1;
  const pct = (x) => (100 * x / total).toFixed(2) + "%";
  const items = r.timeline || [{ kind: "speech", start: 0, duration: r.duration, detail: { segments: r.segments } }];
  for (const it of items) {
    if (it.kind === "speech") {
      // Split speech bar proportionally by segment text length to show language switches.
      const segs = it.detail.segments || [];
      const totalLen = segs.reduce((a, s) => a + s.text.length, 0) || 1;
      let off = it.start;
      for (const s of segs) {
        const d = it.duration * s.text.length / totalLen;
        tl.append(el("div", { class: "tl speech " + s.language, style: `left:${pct(off)};width:${pct(d)}`, title: s.text }, s.text));
        off += d;
        segList.append(el("li", {}, el("span", { class: "lang " + s.language }, s.language.toUpperCase()), s.text, el("span", { class: "muted small" }, "  · " + s.voice)));
      }
    } else {
      const label = it.kind === "pause" ? "pause" : (it.detail.effect || it.detail.prompt || it.kind);
      tl.append(el("div", { class: "tl " + it.kind, style: `left:${pct(it.start)};width:${pct(it.duration)}`, title: label }, label));
    }
  }
}
$("#produceBtn").addEventListener("click", (e) => busy(e.target, async () => showStudio(await api("/api/produce", { ...voiceParams(), script: $("#script").value }))));
$("#speakBtn").addEventListener("click", (e) => busy(e.target, async () => {
  const text = $("#script").value.replace(/\[(sfx|bg|pause|voice|lang)(:[^\]]*)?\]/gi, " ");
  showStudio(await api("/api/tts", { ...voiceParams(), text }));
}));

// ---------- sound fx
async function playSfx(prompt, duration, btn) {
  const r = await api("/api/sfx", { prompt, duration: duration || null, format: "wav" });
  const url = toUrl(r.audio, r.mime);
  $("#sfxAudio").hidden = false;
  $("#sfxAudio").src = url;
  $("#sfxInfo").textContent = r.info.source === "elevenlabs"
    ? `Generated by ElevenLabs · ${r.duration.toFixed(1)} s`
    : `Built-in effect “${r.info.effect}” · ${r.duration.toFixed(1)} s`;
  document.querySelectorAll(".sfx-btn").forEach((b) => b.classList.remove("playing"));
  if (btn) btn.classList.add("playing");
  const a = play(url);
  a.onended = () => btn && btn.classList.remove("playing");
}
$("#sfxForm").addEventListener("submit", (e) => {
  e.preventDefault();
  busy(e.submitter || $("#sfxForm button"), () => playSfx($("#sfxPrompt").value, parseFloat($("#sfxDur").value)));
});
api("/api/sfx").then((list) => {
  const names = { ui: "Interface", transition: "Transitions", nature: "Nature & ambience", people: "People & objects", music: "Music & culture", impact: "Impacts" };
  const byCat = {};
  for (const s of list) (byCat[s.category] ||= []).push(s);
  for (const [cat, items] of Object.entries(byCat)) {
    $("#sfxGrid").append(el("div", { class: "cat" }, names[cat] || cat));
    const grid = el("div", { class: "sfx-grid" });
    for (const s of items) {
      const b = el("button", { class: "sfx-btn", type: "button", title: `[sfx:${s.name}]` + (s.loopable ? " — loops as [bg:" + s.name + "]" : "") },
        s.name.replace("_", " "), el("span", { class: "te", lang: "te" }, s.label_te));
      b.addEventListener("click", () => playSfx(s.name, null, b).catch((e) => toast(e.message)));
      grid.append(b);
    }
    $("#sfxGrid").append(grid);
  }
});
