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
async function api(path, body, method = body === undefined ? "GET" : "POST") {
  const r = await fetch("/api/" + path, body === undefined ? {method} : {method,headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
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
function fileCard(file) {
  const name = file.name || file;
  const canPreview = file.id || file.chunks?.length;
  const card = canPreview ? button("", () => openPreview(file, card), "file-card") : make("div", null, "file-card");
  const extension = name.split(".").at(-1).toUpperCase();
  const type = name.includes(".") ? extension.slice(0, 8) : "FILE";
  const label = make("span", null, "file-card-label");
  label.append(make("strong", name), make("small", file.removed ? "Removed" : type));
  card.append(make("span",type,"file-card-icon"), label);
  if (canPreview) card.setAttribute("aria-label", "Preview " + name);
  card.title = name;
  return card;
}

let previewState;
function openPreview(file, trigger) {
  closeSideChat();
  previewState = {file, trigger, part:0, sheet:0, zoom:100};
  $("file-preview-title").textContent = file.name;
  $("file-preview-info").textContent = file.size ? Math.ceil(file.size / 1024) + " KB" : "";
  const download = $("file-preview-download");
  download.hidden = !file.id;
  if (file.id) download.href = "/api/artifacts/" + file.id + "/download";
  $("file-preview").hidden = false;
  document.body.classList.add("preview-open");
  $("file-preview-close").focus();
  loadPreview(previewState);
}

function closePreview() {
  if (!previewState) return;
  const {trigger} = previewState;
  previewState = null;
  $("file-preview").hidden = true;
  document.body.classList.remove("preview-open");
  $("file-preview-body").replaceChildren();
  $("file-preview-controls").replaceChildren();
  if (trigger.isConnected) trigger.focus();
}
$("file-preview-close").onclick = closePreview;
document.addEventListener("keydown", event => {
  if (event.key === "Escape" && !$("dialog").open) closePreview();
});

async function loadPreview(view) {
  const body = $("file-preview-body"), controls = $("file-preview-controls");
  controls.replaceChildren();
  body.textContent = "Loading preview…";
  let data;
  try {
    data = view.file.id
      ? await (await api("artifacts/" + view.file.id + "/preview?part=" + view.part + "&sheet=" + view.sheet)).json()
      : {kind:"document",blocks:view.file.chunks.map(chunk => ({label:chunk.source,text:chunk.text}))};
  } catch (error) {
    if (previewState === view) body.textContent = error.message;
    return;
  }
  if (previewState !== view) return;
  body.replaceChildren();
  body.scrollTop = body.scrollLeft = 0;
  const content = "/api/artifacts/" + view.file.id + "/preview/content?part=" + view.part;
  if (data.kind === "pdf" || data.kind === "image") {
    const img = make("img",null,data.kind === "pdf" ? "preview-pdf" : "preview-image");
    img.src = content;
    img.alt = data.kind === "pdf" ? "PDF page " + (view.part + 1) : data.name;
    img.onerror = () => img.replaceWith(make("p","This image could not be rendered. Download the original to open it."));
    body.append(img);
    if (data.kind === "pdf") {
      const scale = make("span");
      const zoom = amount => {
        view.zoom = Math.max(50, Math.min(200, view.zoom + amount));
        img.style.width = view.zoom + "%";
        scale.textContent = view.zoom + "%";
        smaller.disabled = view.zoom === 50;
        larger.disabled = view.zoom === 200;
      };
      const smaller = button("−",() => zoom(-25),"action");
      const larger = button("+",() => zoom(25),"action");
      smaller.setAttribute("aria-label","Zoom out");
      larger.setAttribute("aria-label","Zoom in");
      controls.append(smaller,scale,larger);
      zoom(0);
    }
  } else if (data.kind === "audio" || data.kind === "video") {
    const player = make(data.kind);
    player.controls = true;
    player.src = content;
    player.onerror = () => player.replaceWith(make("p","Your browser could not play this file. Download the original to open it."));
    body.append(player);
  } else if (data.kind === "table") {
    if (data.sheets?.length) {
      const sheet = make("select");
      sheet.setAttribute("aria-label","Worksheet");
      data.sheets.forEach((name,index)=>sheet.add(new Option(name,index)));
      sheet.value = view.sheet;
      sheet.onchange = () => { view.sheet = Number(sheet.value); view.part = 0; loadPreview(view); };
      controls.append(sheet);
    }
    const table = make("table",null,"preview-table");
    const rows = make("tbody");
    data.rows.forEach((cells,index)=>{
      const row = make("tr");
      const number = make("th",data.first_row+index);
      number.scope = "row";
      row.append(number,...cells.map(cell=>make("td",cell)));
      rows.append(row);
    });
    table.append(rows);
    body.append(table);
    if (!data.rows.length) body.append(make("p","This sheet has no rows to show."));
    if (data.clipped_columns) body.append(make("p","Preview shows the first 30 columns."));
  } else if (data.kind === "document") {
    const documentText = make("div",null,"preview-document");
    documentText.append(...data.blocks.map(block => {
      const section = make("section");
      section.append(make("small",block.label),make("p",block.text));
      return section;
    }));
    body.append(documentText);
    if (!data.blocks.length) documentText.append(make("p","No readable text in this file."));
  } else if (data.kind === "text") {
    body.append(make("pre",data.text));
  } else {
    body.append(make("p",data.reason));
  }
  if (data.truncated) body.append(make("p","Preview is truncated. Download the original to see the rest."));
  if (data.kind === "pdf" || data.kind === "table") {
    const move = amount => { view.part += amount; loadPreview(view); };
    const previous = button("Previous",() => move(-1),"action");
    const next = button("Next",() => move(1),"action");
    previous.disabled = view.part === 0;
    next.disabled = data.kind === "pdf" ? view.part + 1 >= data.pages : !data.more;
    const position = data.kind === "pdf"
      ? "Page " + (view.part + 1) + " of " + data.pages
      : data.rows.length ? "Rows " + data.first_row + "–" + (data.first_row + data.rows.length - 1) : "No rows";
    controls.prepend(previous,make("span",position),next);
  }
}

function question(m) {
  const group = make("div",null,"query");
  const bubble = make("div",m.text,"user");
  if (m.attachments?.length) {
    const attachments = make("div", null, "query-files");
    attachments.append(...m.attachments.map(fileCard));
    group.append(attachments);
  }
  const actions = make("div",null,"actions");
  actions.append(copyButton("Copy question",m.text));
  const reusable = (m.attachments || []).filter(file => file.id || file.chunks);
  if (reusable.length) {
    const reuse = button("Reuse files", () => attachFiles(reusable));
    reuse.disabled = busy || uploading;
    actions.append(reuse);
  }
  if (m.quote) group.append(make("blockquote",m.quote,"sent-quote"));
  group.append(bubble, actions);
  return group;
}
function answer(m, index) {
  const article = make("article",null,"answer"), by = make("div",null,"by");
  article.dataset.index = index;
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
          current={...fresh(),model:current.model,messages:previous,
            files:(q.attachments || []).filter(file => file.id || file.chunks)}; $("dialog").close(); draw(); send(q.text);
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
    if (location.hash !== "#chat") location.hash = "chat";
    closeSideChat();
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

let selectedExcerpt, selectionFrame, sideChat;

function updateSelectionActions() {
  selectionFrame = 0;
  const toolbar = $("selection-actions");
  const selection = window.getSelection();
  const node = selection?.anchorNode;
  const anchor = (node?.nodeType === Node.ELEMENT_NODE ? node : node?.parentElement);
  const content = anchor?.closest("#messages .answer .content");
  const text = selection?.toString().trim();
  if (!text || !selection.rangeCount || busy || uploading || !content ||
      !content.contains(selection.focusNode) || $("scroll").getBoundingClientRect().height === 0 ||
      $("dialog").open) {
    toolbar.hidden = true;
    selectedExcerpt = null;
    return;
  }
  const bounds = selection.getRangeAt(0).getBoundingClientRect();
  const visible = $("scroll").getBoundingClientRect();
  if (bounds.bottom < visible.top || bounds.top > visible.bottom) {
    toolbar.hidden = true;
    selectedExcerpt = null;
    return;
  }
  const index = Number(content.closest("article").dataset.index);
  const prior = current.messages[index - 1];
  const files = prior?.forgotten || current.messages[index]?.forgotten ? [] :
    (prior?.attachments || []).filter(file => file.id || file.chunks);
  selectedExcerpt = {text, files, trigger:content};
  toolbar.hidden = false;
  toolbar.style.left = Math.max(8,Math.min(innerWidth-toolbar.offsetWidth-8,
    bounds.left+bounds.width/2-toolbar.offsetWidth/2))+"px";
  toolbar.style.top = Math.max(8,bounds.top-toolbar.offsetHeight-8)+"px";
}

function scheduleSelectionActions() {
  if (!selectionFrame) selectionFrame = requestAnimationFrame(updateSelectionActions);
}
document.addEventListener("selectionchange",scheduleSelectionActions);
document.addEventListener("scroll",scheduleSelectionActions,true);
window.addEventListener("resize",scheduleSelectionActions);
$("selection-actions").onpointerdown = event => event.preventDefault();

function takeSelection() {
  const excerpt = selectedExcerpt;
  if (!excerpt || busy || uploading) return null;
  if (excerpt.text.length > 6000) {
    notice("Select up to 6,000 characters to ask about an excerpt.");
    return null;
  }
  selectedExcerpt = null;
  $("selection-actions").hidden = true;
  window.getSelection().removeAllRanges();
  return excerpt;
}

function renderDraftQuote() {
  $("quote-draft").replaceChildren();
  if (!current.quote) return;
  const remove = button("×",() => {
    delete current.quote;
    save();
    renderDraftQuote();
    $("question").focus();
  });
  remove.setAttribute("aria-label","Remove quoted text");
  remove.title = "Remove quoted text";
  $("quote-draft").append(make("blockquote",current.quote.text),remove);
}

$("selection-add").onclick = () => {
  const excerpt = takeSelection();
  if (!excerpt) return;
  current.quote = {text:excerpt.text,files:excerpt.files};
  save();
  renderDraftQuote();
  $("question").focus();
};

function closeSideChat() {
  const side = sideChat;
  if (!side) return;
  sideChat = null;
  if (side.run) {
    api("stop/"+side.run.id,{}).catch(()=>{});
    side.run.controller.abort();
  }
  $("side-chat").hidden = true;
  document.body.classList.remove("side-chat-open");
  if (side.trigger.isConnected) $("question").focus();
}
$("side-chat-close").onclick = closeSideChat;
document.addEventListener("keydown", event => {
  if (event.key === "Escape" && !$("dialog").open) {
    closeSideChat();
    window.getSelection()?.removeAllRanges();
  }
});

$("selection-side").onclick = () => {
  const excerpt = takeSelection();
  if (!excerpt) return;
  closePreview();
  closeSideChat();
  sideChat = {...excerpt,model:current.model,history:[]};
  $("side-quote").textContent = excerpt.text;
  $("side-question").value = "";
  $("side-status").textContent = "";
  $("side-send").textContent = "Send";
  $("side-messages").replaceChildren();
  $("side-chat").hidden = false;
  document.body.classList.add("side-chat-open");
  $("side-question").focus();
};

$("side-composer").onsubmit = async event => {
  event.preventDefault();
  const side = sideChat;
  if (!side) return;
  if (side.run) {
    const run = side.run;
    try {
      const stopped = await (await api("stop/"+run.id,{})).json();
      if (stopped.stopped) run.controller.abort();
    } catch (error) {
      if (sideChat === side && side.run === run) $("side-status").textContent = error.message;
    }
    return;
  }
  const text = $("side-question").value.trim();
  if (!text) return;
  const run = {id:crypto.randomUUID(),controller:new AbortController()};
  side.run = run;
  $("side-status").textContent = "Working…";
  $("side-send").textContent = "Stop";
  $("side-question").value = "";
  const messages = $("side-messages");
  const question = make("p",text,"user");
  messages.append(question);
  messages.scrollTop = messages.scrollHeight;
  try {
    const body = {request_id:run.id,text,selected_text:side.text,model:side.model,files:side.files,history:side.history};
    const response = await fetch("/api/ask",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify(body),signal:run.controller.signal});
    const answer = await response.json();
    if (!response.ok) throw Error(answer.detail || "Request failed.");
    if (sideChat !== side) return;
    side.history.push({role:"user",text},{role:"assistant",text:answer.text});
    const reply = make("div",null,"content");
    reply.innerHTML = answer.html;
    if (answer.sources?.length) reply.append(button("Sources",()=>sourceList(answer)));
    messages.append(reply);
    $("side-status").textContent = "";
  } catch (error) {
    if (sideChat !== side) return;
    question.remove();
    $("side-question").value = text;
    $("side-status").textContent = error.name === "AbortError" ? "Response stopped." : error.message;
  } finally {
    if (sideChat === side) {
      side.run = null;
      $("side-send").textContent = "Send";
      messages.scrollTop = messages.scrollHeight;
    }
  }
};
$("side-question").onkeydown = event => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing && !sideChat?.run) {
    event.preventDefault();
    $("side-composer").requestSubmit();
  }
};

function draw() {
  renderDraftQuote();
  $("files").replaceChildren(...(current.files||[]).map((f,i)=>{
    const b=button(f.name+" ×",()=>{if(busy)return;current.files.splice(i,1);save();draw();});
    b.disabled=busy; b.title="Remove "+f.name;return b;
  }));
  $("messages").replaceChildren(...current.messages.map((m,i)=>m.role==="assistant"?answer(m,i):question(m)));
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
function attachFiles(attachments) {
  if (busy || uploading) return;
  const draft = [...(current.files || [])];
  for (const entry of attachments) {
    if (!draft.some(file => entry.id ? file.id === entry.id : file === entry)) draft.push(entry);
  }
  if (draft.length > 4) throw Error("A chat message can include up to four files.");
  current.files = draft;
  save(); draw();
  notice("Files ready to send.");
  if (location.hash !== "#chat") location.hash = "chat";
  $("question").focus();
}

function confirmRemoveArtifact(entry) {
  const box = modal("Remove file?");
  const remove = button("Remove file", async () => {
    remove.disabled = true;
    try {
      await api("artifacts/" + entry.id, undefined, "DELETE");
      for (const chat of new Set([...chats, current])) {
        chat.files = chat.files?.filter(file => file.id !== entry.id);
        if (chat.quote) chat.quote.files = chat.quote.files?.filter(file => file.id !== entry.id);
        for (const message of chat.messages) {
          if (message.attachments) message.attachments = message.attachments.map(file =>
            file.id === entry.id ? {name:file.name, removed:true} : file);
        }
      }
      try { localStorage.setItem("wesla.chats", JSON.stringify(chats)); }
      catch { notice("File removed. Browser storage could not be updated."); }
      $("dialog").close();
      draw();
      await loadArtifacts("Removed " + entry.name + ".");
    } catch (error) {
      box.querySelector("[role=status]").textContent = error.message;
      remove.disabled = false;
    }
  });
  const feedback = make("p");
  feedback.setAttribute("role","status");
  box.append(make("p","Remove “" + entry.name + "” and its extracted text from Artifacts? It will no longer be available for download or follow-up questions."),
    button("Cancel", () => $("dialog").close()), remove, feedback);
}

async function loadArtifacts(message) {
  const list = $("artifact-list");
  list.textContent = "Loading files…";
  try {
    const entries = await (await api("artifacts")).json();
    list.replaceChildren(...entries.map(entry => {
      const row = make("div", null, "artifact-row");
      const description = make("div");
      description.append(fileCard(entry),
        make("div", Math.ceil(entry.size / 1024) + " KB · " + (entry.issue ? "Stored file" : "Ready to analyze"), "elapsed"));
      if (entry.issue) description.append(make("small", entry.issue));
      const menu = make("details", null, "artifact-menu");
      menu.name = "artifact-menu";
      const trigger = make("summary","•••");
      trigger.title = "Options for " + entry.name;
      trigger.setAttribute("aria-label",trigger.title);
      const actions = make("div", null, "artifact-options");
      const link = make("a", "Download");
      link.href = "/api/artifacts/" + entry.id + "/download";
      link.onclick = () => { menu.open = false; };
      const attach = button("Attach to chat", () => attachFiles([entry]));
      attach.disabled = Boolean(entry.issue) || busy || uploading;
      const remove = button("Remove", () => { menu.open = false; confirmRemoveArtifact(entry); });
      remove.disabled = busy || uploading;
      actions.append(link, attach, remove);
      menu.append(trigger, actions);
      row.append(description, menu);
      return row;
    }));
    if (message) {
      const feedback = make("p",message);
      feedback.setAttribute("role","status");
      list.prepend(feedback);
    }
    const legacy = chats.some(chat => chat.files?.some(file => !file.id));
    if (!entries.length) list.append(make("p", "Uploaded files will appear here."));
    if (legacy) list.append(make("p", "Some earlier attachments have extracted text only. Upload the originals using the chat composer to make them downloadable."));
  } catch (error) { list.textContent = error.message; }
}

function forgetTurns(indices) {
  if (busy || uploading) return;
  for (const index of indices) current.messages[index].forgotten = true;
  save();
  $("dialog").close();
  loadMemory();
}

$("clear-context").onclick = () => {
  const box = modal("Clear short-term memory?");
  box.append(make("p","Forget all current turns for future answers while keeping the visible transcript."),
    button("Cancel", () => $("dialog").close()),
    button("Clear context", () => forgetTurns(retainedHistory().map(turn => turn.index))));
};

function confirmMemoryRemoval(entry) {
  const box = modal("Remove memory record?");
  const status = make("p");
  status.setAttribute("role","status");
  const remove = button("Remove record", async () => {
    remove.disabled = true;
    try {
      await api("memory/" + entry.id, undefined, "DELETE");
      $("dialog").close();
      await loadMemory();
    } catch (error) {
      status.textContent = error.message;
      remove.disabled = false;
    }
  });
  box.append(make("p","Delete this passage from long-term memory and its local training queue entry?"),
    button("Cancel", () => $("dialog").close()), remove, status);
}

$("export-training").onclick = () => {
  const turns = retainedHistory();
  const examples = [];
  for (let i = 1; i < turns.length; i++) {
    const previous = turns[i-1], answer = turns[i];
    if (previous.role === "user" && answer.role === "assistant" && answer.index === previous.index + 1) {
      examples.push({kind:"learn",memory_id:-i,text:"Question: "+previous.text+"\nAnswer: "+answer.text});
    }
  }
  if (!examples.length) {
    $("training-export-status").textContent = "No complete retained exchanges in this chat.";
    return;
  }
  const file = new Blob([examples.map(example => JSON.stringify(example)).join("\n")+"\n"],
    {type:"application/x-ndjson"});
  const url = URL.createObjectURL(file);
  const link = make("a");
  link.href = url;
  link.download = "wesla-chat-training.jsonl";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url),30000);
  $("training-export-status").textContent = "Exported " + examples.length + " examples for review.";
};

let memoryOffset = 0;
async function loadMemory() {
  $("memory-counts").textContent = "Loading memory…";
  try {
    const history = retainedHistory().map(({role,text,index})=>({role,text,index}));
    const data = await (await api("memory", {history, offset:memoryOffset})).json();
    $("short-memory").replaceChildren(...data.short_term.map(turn => {
      const entry = make("div",null,"memory-entry");
      const remove = button("Forget turn", () => forgetTurns([turn.index]), "action");
      remove.disabled = busy || uploading;
      entry.append(make("p", (turn.role === "user" ? "You" : "WESLA") + " · " + turn.text), remove);
      return entry;
    }));
    $("clear-context").disabled = !history.length || busy || uploading;
    $("export-training").disabled = !history.length || busy;
    if (!data.short_term.length) $("short-memory").textContent = "No retained context in this chat.";
    $("memory-counts").textContent = data.counts.total + " records · " + (data.counts.queued || 0) +
      " queued · " + (data.counts.learned || 0) + " marked learned";
    $("training-counts").textContent = data.pending + " queue entries · " + data.columns + " registered LoRA adapters";
    $("long-memory").replaceChildren(...data.entries.map(entry => {
      const detail = make("details");
      const remove = button("Remove record", () => confirmMemoryRemoval(entry), "action");
      detail.append(make("summary", "#" + entry.id + " · " + entry.status + " · " + entry.knowledge.slice(0,100)),
        make("p",entry.knowledge), make("small", JSON.parse(entry.sources).join(" · ")), remove);
      return detail;
    }));
    $("memory-prev").disabled = memoryOffset === 0;
    $("memory-next").disabled = memoryOffset + 20 >= data.counts.total;
  } catch (error) { $("memory-counts").textContent = error.message; }
}
$("memory-prev").onclick = () => { memoryOffset = Math.max(0, memoryOffset - 20); loadMemory(); };
$("memory-next").onclick = () => { memoryOffset += 20; loadMemory(); };

function showPage() {
  const page = location.hash.slice(1);
  const pages = ["settings", "artifacts", "memory"];
  document.body.classList.toggle("settings-open", pages.includes(page));
  for (const name of pages) $(name).hidden = name !== page;
  document.title = pages.includes(page) ? $(page+"-heading").textContent + " · WESLA" : "Wesla · Chat";
  if (innerWidth <= 750) document.body.classList.add("closed");
  if (pages.includes(page)) { closeSideChat(); $(page+"-heading").focus(); }
  if (page === "artifacts") loadArtifacts();
  if (page === "memory") { memoryOffset = 0; loadMemory(); }
}
addEventListener("hashchange", showPage);
showPage();
$("attach").onclick=()=>{if(!busy&&!uploading)$("fileInput").click();};
$("fileInput").onchange=async e=>{
  const files=[...e.target.files];e.target.value="";
  const chat=current;
  const issues=[];
  uploading=true;$("attach").disabled=$("new").disabled=$("send").disabled=true;
  try {
    for(const file of files){
      if(file.size>15000000){issues.push(file.name + " exceeds 15 MB.");continue;}
      notice("Saving "+file.name+"…");
      const form=new FormData();form.append("file",file);
      const r=await fetch("/api/upload",{method:"POST",body:form});
      const data=await r.json();
      if(!r.ok){issues.push(data.detail);continue;}
      if(data.issue)issues.push(data.name + " saved to Artifacts for download only. " + data.issue);
      else if((chat.files||[]).length<4)(chat.files??=[]).push(data);
      else issues.push(data.name + " saved to Artifacts. This chat already has four files.");
    }
    notice(issues.join(" ") || "Files attached and saved to Artifacts.");
  }catch(e){notice(e.message);}
  finally{uploading=false;$("attach").disabled=$("new").disabled=$("send").disabled=false;if(chat.files?.length&&chat===current)save();draw();}
};
function showActivity(run) {
  const stage = run.stopping ? "Stopping…" : run.stage;
  if ($("activity-stage").textContent !== stage) $("activity-stage").textContent = stage;
  const elapsed = Math.floor((performance.now() - run.start) / 1000);
  $("activity-time").textContent = "· " + elapsed + " s";
}

async function stop(){
  const run=active;if(!run||run.stopping)return;
  run.stopping=true;showActivity(run);
  try{
    const r=await api("stop/"+run.id,{});const result=await r.json();
    if(result.stopped)run.controller.abort();
    else run.stopping=false;
  }catch(e){run.stopping=false;notice(e.message);}
}
function retainedHistory() {
  return current.messages.flatMap((message,index) => message.forgotten ? [] : [{...message,index}]);
}

function referencedFiles(text, history) {
  const question = text.toLowerCase();
  const turns = history.slice().reverse().filter(turn => turn.role === "user" && turn.attachments?.length);
  for (const turn of turns) {
    const named = turn.attachments.filter(file => (file.id || file.chunks) && question.includes(file.name.toLowerCase()));
    if (named.length) return named;
  }
  const refersToFile = /\b(?:this|that|the|these|those|attached|uploaded|previous|same)\s+(?:file|table|spreadsheet|sheet|workbook|document|pdf|attachment|report|dataset)s?\b/.test(question);
  const last = history.at(-1);
  const lastQuestion = history.findLast(turn => turn.role === "user");
  const dependsOnConversation = /\b(?:that|those|these|them|their|it|its|above|same)\b/.test(question)
    || /^(?:(?:ok(?:ay)?|and|so)[, ]+)?(?:what about|how many|what else|what (?:are|were) the names?|(?:list|show|give me) (?:the )?(?:student )?names|can you elaborate|explain more|summari[sz]e|compare them)\b/.test(question);
  const continuesFileAnswer = (last?.route === "files" || lastQuestion?.attachments?.length) && dependsOnConversation;
  if (!refersToFile && !continuesFileAnswer) return [];
  const recent = turns[0]?.attachments.filter(file => file.id || file.chunks) || [];
  const spreadsheets = recent.filter(file => /\.(?:xlsx|csv|tsv)$/i.test(file.name));
  return /\b(?:table|spreadsheet|sheet|workbook)s?\b/.test(question) && spreadsheets.length
    ? spreadsheets : recent;
}

async function send(text) {
  if(busy||uploading)return;
  text=text.trim() || (current.quote ? "Explain this excerpt." : current.files?.length ? "Summarize each attached file." : "");
  if(!text)return;
  busy=true; $("new").disabled=$("model").disabled=$("attach").disabled=true;
  $("send").textContent="■";$("send").setAttribute("aria-label","Stop response");
  const history = retainedHistory();
  const draftFiles = current.files || [];
  const quote = current.quote;
  const attachments = draftFiles.length ? draftFiles : quote?.files?.length ? quote.files : referencedFiles(text, history);
  delete current.quote;
  current.files = [];
  current.messages.push({role:"user",text,attachments,quote:quote?.text});
  $("question").value = "";
  const run={id:crypto.randomUUID(),controller:new AbortController(),start:performance.now(),stage:"Submitting your question"};
  active=run;
  const tick=async()=>{
    if(active!==run)return;
    showActivity(run);
    if(run.polling)return;
    run.polling=true;
    try{
      const r=await api("activity/"+run.id); const data=await r.json();
      if(active!==run)return;
      if(data.stage)run.stage=data.stage;
      showActivity(run);
    }catch{}finally{run.polling=false;}
  };
  notice("");
  $("activity").hidden = false;
  showActivity(run);
  draw(); $("scroll").scrollTop=$("scroll").scrollHeight;
  const timer=setInterval(tick,450);
  try {
    const body={request_id:run.id,text,selected_text:quote?.text,model:current.model,files:attachments,history:history.map(({role,text})=>({role,text}))};
    const r=await fetch("/api/ask",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body),signal:run.controller.signal});
    const data=await r.json();if(!r.ok)throw Error(data.detail||"Request failed.");
    current.messages.push(data); notice(""); save();
  } catch(e) {
    current.messages.pop();
    current.files = draftFiles;
    if (quote) current.quote = quote;
    $("question").value = text;
    notice(run.stopping?"Response stopped.":e.message);
  }
  finally {clearInterval(timer);$("activity").hidden=true;active=null;busy=false; $("send").textContent="↑";$("send").setAttribute("aria-label","Send message");
    $("new").disabled=$("model").disabled=$("attach").disabled=false; draw(); $("scroll").scrollTop=$("scroll").scrollHeight;}
}
$("composer").onsubmit=e=>{e.preventDefault();if(busy)stop();else send($("question").value);};
$("question").onkeydown=e=>{if(e.key==="Enter"&&!e.shiftKey&&!e.isComposing){e.preventDefault();send(e.target.value);}};
$("download").onclick=()=>downloadMenu("Download conversation",{messages:current.messages},"wesla-chat");
$("new").onclick=()=>{closeSideChat();if(location.hash!=="#chat")location.hash="chat";current=fresh();$("question").value="";notice("");draw();if(innerWidth<=750)document.body.classList.add("closed");$("question").focus();};
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
document.addEventListener("pointerdown",e=>{
  if(!picker.contains(e.target))openPicker(false);
  const menu = document.querySelector(".artifact-menu[open]");
  if (menu && !menu.contains(e.target)) menu.open = false;
});
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

