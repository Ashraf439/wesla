"use strict";
const $ = id => document.getElementById(id);
const make = (tag, text, cls) => {
  const node = document.createElement(tag);
  if (text != null) node.textContent = text;
  if (cls) node.className = cls;
  return node;
};
const notice = text => { $("status").textContent = text; };
function button(label, action, cls) {
  const b = make("button", label, cls);
  b.type = "button";
  b.onclick = () => Promise.resolve().then(action).catch(e => notice(e.message));
  return b;
}
const glyphs = {
  copy: '<rect x="8" y="8" width="12" height="13" rx="2"/><path d="M15 8V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h3"/>',
  check: '<path d="m4 12 5 5L20 6"/>',
  download: '<path d="M12 3v12m-5-5 5 5 5-5M4 16v4h16v-4"/>',
  like: '<path d="M3 10h4v11H3zM7 11l5-8c3 0 3 3 2 6h5c2 0 2 2 2 3l-2 7c0 1-1 2-3 2h-5l-4-2z"/>',
  dislike: '<g transform="rotate(180 12 12)"><path d="M3 10h4v11H3zM7 11l5-8c3 0 3 3 2 6h5c2 0 2 2 2 3l-2 7c0 1-1 2-3 2h-5l-4-2z"/></g>',
  share: '<path d="m14 4 7 6-7 6v-4c-5 0-8 2-11 6 1-8 5-11 11-11z"/>',
  sources: '<path d="M12 6c-3-2-6-2-10-1v14c4-1 7-1 10 1 3-2 6-2 10-1V5c-4-1-7-1-10 1Zm0 0v14"/>',
  picker: '<path d="m3 7 6-3 6 3-6 3zM3 11l6 3 6-3M3 15l6 3 6-3m1-3 3 3 3-3"/>'
};
function icon(control, label, name) {
  control.classList.add("icon-button");
  control.title = label;
  control.setAttribute("aria-label", label);
  control.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' + glyphs[name] + '</svg>';
  return control;
}
icon($("download"), "Download chat", "download");
let chats;
try { chats = JSON.parse(localStorage.getItem("wesla.chats")) || []; } catch { chats = []; }
let current = chats[0] || fresh(), busy = false;
function fresh() { return {id: crypto.randomUUID(), messages: [], model: ""}; }
function save() {
  chats = [current, ...chats.filter(c => c.id !== current.id)];
  try { localStorage.setItem("wesla.chats", JSON.stringify(chats)); }
  catch { notice("Browser storage is full. Download this chat to keep a copy."); }
}
async function api(path, body) {
  const r = await fetch("/api/" + path, body === undefined ? {} : {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  if (!r.ok) {
    const data = await r.json().catch(() => ({}));
    throw Error(typeof data.detail === "string" ? data.detail : "Request failed (" + r.status + ").");
  }
  return r;
}
function modal(title) {
  $("dialogTitle").textContent = title;
  $("dialogBody").replaceChildren();
  if (!$("dialog").open) $("dialog").showModal();
  return $("dialogBody");
}
$("close").onclick = () => $("dialog").close();
function downloadMenu(title, payload, name) {
  const box = modal(title);
  for (const [ext,label] of [["pdf","PDF document"],["docx","Word document (.docx)"],["xlsx","Excel workbook (.xlsx)"],["json","JSON data"]]) {
    box.append(button(label, async () => {
      const r = await api("export", {...payload, format:ext});
      const url = URL.createObjectURL(await r.blob());
      const a = make("a"); a.href = url; a.download = name + "." + ext; a.click();
      setTimeout(() => URL.revokeObjectURL(url),30000); $("dialog").close(); notice("Download started.");
    }));
  }
}
function textWithSources(m) { return [m.text,...(m.sources || [])].join("\n\n"); }
async function share(m) {
  const text = textWithSources(m);
  if (navigator.share) { try { await navigator.share({text}); } catch(e) { if(e.name !== "AbortError") throw e; } }
  else { await navigator.clipboard.writeText(text); notice("Answer and sources copied for sharing."); }
}
function sourceList(m) {
  const box = modal("Sources");
  if (!m.sources.length) box.append(make("p","This answer has no cited sources."));
  for (const url of m.sources) {
    if (/^https?:\/\//i.test(url)) {
      const a = make("a",url); a.href=url; a.target="_blank"; a.rel="noopener noreferrer"; box.append(a);
    } else box.append(make("p",url));
  }
}
async function feedback(m, rating, correction="") {
  await api("feedback",{id:m.id,rating,correction});
  m.vote = rating; save(); draw(); $("dialog").close(); notice("Feedback saved.");
}
function copyButton(label, text) {
  const control = button(label, async () => {
    await navigator.clipboard.writeText(text);
    icon(control, label, "check");
    notice(label.slice(5) + " copied.");
    setTimeout(() => icon(control, label, "copy"), 1400);
  }, "action");
  return icon(control, label, "copy");
}
function question(m) {
  const group = make("div",null,"query");
  group.append(make("div",m.text,"user"), copyButton("Copy question",m.text));
  return group;
}
function answer(m, index) {
  const article = make("article",null,"answer"), by = make("div",null,"by");
  const logo = make("img"); logo.src="/assets/wesla.svg"; logo.alt="";
  by.append(logo,document.createTextNode("Wesla"));
  if(Number.isFinite(m.seconds)) by.append(make("span","· "+m.seconds.toFixed(1)+" s","elapsed"));
  const content = make("div",null,"content");
  // Markdown is rendered by the server with raw HTML disabled.
  content.innerHTML = m.html;
  for (const pre of content.querySelectorAll("pre")) {
    const code = pre.querySelector("code") || pre;
    const lang = [...code.classList].find(name => name.startsWith("language-"));
    const block = make("div", null, "code-block");
    const toolbar = make("div", null, "code-toolbar");
    toolbar.append(make("span", lang?.slice(9) || "Code"), copyButton("Copy code", code.textContent));
    pre.replaceWith(block);
    block.append(toolbar, pre);
  }
  for (const table of [...content.querySelectorAll("table")]) {
    const wrap = make("div",null,"tablewrap");
    table.replaceWith(wrap); wrap.append(table);
    const rows = Array.from(table.rows, row => Array.from(row.cells,c => c.innerText || c.textContent));
    const toolbar = make("div", null, "table-toolbar");
    const tsvCell = value => /["\t\r\n]/.test(value) ? '"' + value.replaceAll('"','""') + '"' : value;
    toolbar.append(icon(button("Download table",() => downloadMenu("Download table",{rows},"wesla-table"),"action tablebtn"),"Download table","download"),
      copyButton("Copy table", rows.map(row => row.map(tsvCell).join("\t")).join("\n")));
    wrap.after(toolbar);
  }
  const actions = make("div",null,"actions");
  const up = button("Like",()=>feedback(m,1)), down = button("Dislike",()=>{
    const box = modal("Give feedback"), input = make("textarea");
    input.placeholder="Optional correction"; input.setAttribute("aria-label","Optional correction");
    box.append(input,button("Submit dislike",()=>feedback(m,-1,input.value)));
  });
  if(m.vote) { up.disabled=down.disabled=true; (m.vote===1?up:down).className="selected"; }
  actions.append(icon(up,"Like","like"),icon(down,"Dislike","dislike"),icon(button("Share",()=>share(m)),"Share","share"),
    icon(button("Sources",()=>sourceList(m)),m.sources.length+" sources","sources"),
    button("•••",()=>{
      const box=modal("Answer options");
      box.append(button("Copy answer",async()=>{await navigator.clipboard.writeText(textWithSources(m)); $("dialog").close();notice("Copied.");}),
        button("Download answer",()=>downloadMenu("Download answer",{messages:[m]},"wesla-answer")),
        button("Rewrite in a new chat",()=>{
          if(busy) return;
          const previous=current.messages.slice(0,index), q=previous.pop();
          if(!q || q.role!=="user") return;
          current={...fresh(),model:current.model,messages:previous}; $("dialog").close(); draw(); send(q.text);
        }),button("Routing, verification & learning",()=>{
          modal("Answer details").append(make("pre",JSON.stringify({route:m.route,...m.details,decisions:m.decisions},null,2)));
        }));
    },"more"));
  actions.lastChild.setAttribute("aria-label","More answer options");
  article.append(by,content,actions); return article;
}
function chatName(chat) {
  return chat.title || chat.messages[0]?.text || "New chat";
}

function renameChat(chat) {
  const box = modal("Rename chat");
  const input = make("input");
  input.value = chatName(chat);
  input.setAttribute("aria-label", "Chat name");
  const apply = button("Save name", () => {
    const title = input.value.trim();
    if (!title) return;
    const updated = chats.map(c => c.id === chat.id ? {...c, title} : c);
    localStorage.setItem("wesla.chats", JSON.stringify(updated));
    chats = updated;
    if (current.id === chat.id) current = chats.find(c => c.id === chat.id);
    $("dialog").close();
    draw();
  });
  input.oninput = () => { apply.disabled = !input.value.trim(); };
  input.onkeydown = e => { if (e.key === "Enter" && !e.isComposing) apply.click(); };
  box.append(input, apply);
  input.focus(); input.select();
}

function confirmDeleteChat(chat) {
  const box = modal("Delete chat?");
  box.append(make("p", "Remove “" + chatName(chat) + "” from this browser’s chat history?"),
    button("Cancel", () => $("dialog").close()),
    button("Delete chat", () => {
      const remaining = chats.filter(c => c.id !== chat.id);
      localStorage.setItem("wesla.chats", JSON.stringify(remaining));
      chats = remaining;
      if (current.id === chat.id) {
        current = chats[0] || fresh();
        $("question").value = "";
      }
      $("dialog").close();
      draw();
    }));
}

function historyRow(chat) {
  const row = make("div", null, "history-row");
  const name = chatName(chat);
  const open = button(name, () => {
    if (busy || uploading) return;
    if (location.hash === "#settings") location.hash = "chat";
    current = chat;
    $("question").value = "";
    notice(""); draw();
    if (innerWidth <= 750) document.body.classList.add("closed");
  });
  open.title = name;
  if (chat.id === current.id) row.classList.add("active");
  const more = button("•••", () => {
    if (chat.id === current.id && (busy || uploading)) return;
    const box = modal("Chat options");
    box.append(button("Rename", () => renameChat(chat)), button("Delete", () => confirmDeleteChat(chat)));
  }, "history-more");
  more.title = "Options for " + name;
  more.setAttribute("aria-label", more.title);
  more.disabled = chat.id === current.id && (busy || uploading);
  row.append(open, more);
  return row;
}

function draw() {
  $("files").replaceChildren(...(current.files||[]).map((f,i)=>{
    const b=button(f.name+" ×",()=>{if(busy)return;current.files.splice(i,1);if(current.messages.length)save();draw();});
    b.disabled=busy; b.title="Remove "+f.name;return b;
  }));
  $("thread").replaceChildren(...current.messages.map((m,i)=>m.role==="assistant"?answer(m,i):question(m)));
  $("welcome").hidden=!!current.messages.length;
  $("title").textContent=chatName(current);
  $("download").disabled=!current.messages.length;
  renderPicker();
  $("recent").replaceChildren(...chats.map(historyRow));
}
let active, uploading=false;
function theme(mode) {
  document.documentElement.dataset.theme=mode;
  $("appearance").value=mode;
  localStorage.setItem("wesla.theme",mode);
}
theme(localStorage.getItem("wesla.theme")==="light"?"light":"dark");
$("appearance").onchange = e => theme(e.target.value);
function showPage() {
  const settings = location.hash === "#settings";
  document.body.classList.toggle("settings-open", settings);
  $("settings").hidden = !settings;
  document.title = settings ? "Settings · WESLA" : "Wesla · Chat";
  if (innerWidth <= 750) document.body.classList.add("closed");
  if (settings) $("settings-heading").focus();
}
addEventListener("hashchange", showPage);
showPage();
$("attach").onclick=()=>{if(!busy&&!uploading)$("fileInput").click();};
$("fileInput").onchange=async e=>{
  const files=[...e.target.files];e.target.value="";
  const chat=current;
  if(files.length+(chat.files||[]).length>4){notice("Attach up to four files per chat.");return;}
  uploading=true;$("attach").disabled=$("new").disabled=$("send").disabled=true;
  try {
    for(const file of files){
      if(file.size>15000000)throw Error("Maximum file size is 15 MB.");
      notice("Reading "+file.name+"…");
      const form=new FormData();form.append("file",file);
      const r=await fetch("/api/upload",{method:"POST",body:form});
      const data=await r.json();if(!r.ok)throw Error(data.detail);
      (chat.files??=[]).push(data);
    }
    notice("Files ready. Ask a question about them.");
  }catch(e){notice(e.message);}
  finally{uploading=false;$("attach").disabled=$("new").disabled=$("send").disabled=false;if(chat.messages.length&&chat===current)save();draw();}
};
async function stop(){
  const run=active;if(!run||run.stopping)return;
  run.stopping=true;notice("Stopping…");
  try{
    const r=await api("stop/"+run.id,{});const result=await r.json();
    if(result.stopped)run.controller.abort();
    else run.stopping=false;
  }catch(e){run.stopping=false;notice(e.message);}
}
async function send(text) {
  text=text.trim(); if(!text||busy||uploading)return;
  busy=true; $("new").disabled=$("model").disabled=$("attach").disabled=true;
  $("send").textContent="■";$("send").setAttribute("aria-label","Stop response");
  const history=current.messages.slice(); current.messages.push({role:"user",text}); $("question").value="";
  const run={id:crypto.randomUUID(),controller:new AbortController(),start:performance.now(),stage:"Submitting your question"};
  active=run;
  const tick=async()=>{
    if(run.polling||active!==run)return;
    run.polling=true;
    try{
      const r=await api("activity/"+run.id); const data=await r.json();
      if(active!==run)return;
      if(data.stage)run.stage=data.stage;
      notice((run.stopping?"Stopping":run.stage)+" · "+Math.floor((performance.now()-run.start)/1000)+" s");
    }catch{}finally{run.polling=false;}
  };
  draw(); notice(run.stage); $("scroll").scrollTop=$("scroll").scrollHeight;
  const timer=setInterval(tick,450);
  try {
    const body={request_id:run.id,text,model:current.model,files:current.files||[],history:history.map(({role,text})=>({role,text}))};
    const r=await fetch("/api/ask",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body),signal:run.controller.signal});
    const data=await r.json();if(!r.ok)throw Error(data.detail||"Request failed.");
    current.messages.push(data); notice(""); save();
  } catch(e) {current.messages.pop(); $("question").value=text; notice(run.stopping?"Response stopped.":e.message);}
  finally {clearInterval(timer);active=null;busy=false; $("send").textContent="↑";$("send").setAttribute("aria-label","Send message");
    $("new").disabled=$("model").disabled=$("attach").disabled=false; draw(); $("scroll").scrollTop=$("scroll").scrollHeight;}
}
$("composer").onsubmit=e=>{e.preventDefault();if(busy)stop();else send($("question").value);};
$("question").onkeydown=e=>{if(e.key==="Enter"&&!e.shiftKey&&!e.isComposing){e.preventDefault();send(e.target.value);}};
$("download").onclick=()=>downloadMenu("Download conversation",{messages:current.messages},"wesla-chat");
$("new").onclick=()=>{if(location.hash==="#settings")location.hash="chat";current=fresh();$("question").value="";notice("");draw();if(innerWidth<=750)document.body.classList.add("closed");$("question").focus();};
$("toggle").onclick=()=>document.body.classList.toggle("closed");
let installedModels = [];
function openPicker(open) {
  open = open && !$("model").disabled;
  $("models").hidden = !open;
  $("model").setAttribute("aria-expanded",String(open));
}
function renderPicker() {
  icon($("model"),"Select model · "+(current.model||"Auto"),"picker");
  $("models").replaceChildren(...["",...installedModels].map(name=>{
    const selected = name === (current.model||"");
    const choice = button(name||"Auto · WESLA",()=>{
      current.model=name;
      if(current.messages.length)save();
      renderPicker();openPicker(false);$("model").focus();
    });
    choice.setAttribute("aria-pressed",String(selected));
    return choice;
  }));
  if(busy)openPicker(false);
}
const picker=$("picker");
picker.onpointerenter=e=>{if(e.pointerType==="mouse")openPicker(true);};
picker.onpointerleave=()=>{if(!$("models").contains(document.activeElement))openPicker(false);};
$("model").onclick=e=>openPicker(e.detail && matchMedia("(hover:hover)").matches ? true : $("models").hidden);
picker.onkeydown=e=>{
  if(e.key==="Escape"){openPicker(false);$("model").focus();e.preventDefault();}
  if(e.key==="ArrowDown"||e.key==="ArrowUp"){
    e.preventDefault();openPicker(true);
    const options=[...$("models").children], index=options.indexOf(document.activeElement);
    const next=index<0?(e.key==="ArrowDown"?0:options.length-1):(index+(e.key==="ArrowDown"?1:-1)+options.length)%options.length;
    options[next]?.focus();
  }
};
picker.onfocusout=e=>{if(!picker.contains(e.relatedTarget))openPicker(false);};
document.addEventListener("pointerdown",e=>{if(!picker.contains(e.target))openPicker(false);});
document.querySelectorAll(".suggest button").forEach((b,i)=>b.onclick=()=>{
  $("question").value=["Explain how retrieval-augmented generation works.","Compare semantic search and keyword search in a table.","Research the latest developments in local language models."][i];$("question").focus();
});
if(innerWidth<=750)document.body.classList.add("closed");
draw();
api("models").then(r=>r.json()).then(models=>{
  installedModels=models;
  if(!models.includes(current.model))current.model="";
  renderPicker();
}).catch(()=>notice("Ollama model list unavailable. Auto routing remains available."));

