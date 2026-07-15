/**
 * Bullpen Concierge — embeddable chat widget.
 *
 * Usage (one line on any page):
 *   <script src="https://your-host/widget/bullpen-concierge.js"
 *           data-api-base="https://your-host" defer></script>
 *
 * No dependencies, no build step. Talks to the FastAPI backend's
 * /v1/chat/stream SSE endpoint and renders a floating chat bubble.
 */
(function () {
  "use strict";

  var script = document.currentScript;
  var API_BASE = (script && script.dataset.apiBase) || "";
  var GREETING =
    "Hey! I'm the Bullpen Concierge — I can help you set up a wallet, " +
    "fund your account, trade perps or prediction markets, and claim the " +
    "$ANSEM airdrop. What do you need?";
  var QUICK_QUESTIONS = [
    "How do I claim the $ANSEM airdrop?",
    "Who is Ansem?",
    "What is Bullpen?",
    "What is a perp?",
  ];

  var history = []; // {role, content} turns replayed to the stateless backend
  var busy = false;

  /* ---------------------------------------------------------------- css */
  var css =
    ".bpc-btn{position:fixed;bottom:24px;right:24px;width:56px;height:56px;" +
    "border-radius:50%;border:none;background:#16c784;color:#0b0e11;" +
    "font-size:24px;cursor:pointer;z-index:99998;box-shadow:0 4px 20px " +
    "rgba(22,199,132,.45);transition:transform .15s}" +
    ".bpc-btn:hover{transform:scale(1.07)}" +
    ".bpc-panel{position:fixed;bottom:92px;right:24px;width:372px;" +
    "max-width:calc(100vw - 32px);height:540px;max-height:calc(100vh - 120px);" +
    "display:none;flex-direction:column;background:#12161c;color:#e6e8ea;" +
    "border:1px solid #232a33;border-radius:14px;overflow:hidden;" +
    "z-index:99999;font:14px/1.5 -apple-system,BlinkMacSystemFont," +
    "'Segoe UI',Roboto,sans-serif;box-shadow:0 12px 40px rgba(0,0,0,.55)}" +
    ".bpc-panel.bpc-open{display:flex}" +
    ".bpc-head{padding:14px 16px;background:#0b0e11;border-bottom:1px solid " +
    "#232a33;display:flex;align-items:center;gap:10px}" +
    ".bpc-dot{width:8px;height:8px;border-radius:50%;background:#16c784}" +
    ".bpc-title{font-weight:600}" +
    ".bpc-sub{font-size:11px;color:#8a939e;margin-left:auto}" +
    ".bpc-msgs{flex:1;overflow-y:auto;padding:14px;display:flex;" +
    "flex-direction:column;gap:10px}" +
    ".bpc-msg{max-width:85%;padding:9px 12px;border-radius:12px;" +
    "white-space:pre-wrap;word-wrap:break-word}" +
    ".bpc-user{align-self:flex-end;background:#16c784;color:#0b0e11;" +
    "border-bottom-right-radius:4px}" +
    ".bpc-bot{align-self:flex-start;background:#1b2129;" +
    "border-bottom-left-radius:4px}" +
    ".bpc-srcs{align-self:flex-start;display:flex;flex-wrap:wrap;gap:4px;" +
    "margin-top:-4px}" +
    ".bpc-src{font-size:10px;color:#8a939e;background:#1b2129;" +
    "border:1px solid #232a33;border-radius:8px;padding:2px 7px}" +
    ".bpc-quick{display:flex;flex-wrap:wrap;gap:6px;padding:0 14px 10px}" +
    ".bpc-qbtn{font-size:12px;background:transparent;color:#16c784;" +
    "border:1px solid #16c78455;border-radius:14px;padding:5px 10px;" +
    "cursor:pointer}" +
    ".bpc-qbtn:hover{background:#16c78418}" +
    ".bpc-form{display:flex;gap:8px;padding:12px;border-top:1px solid #232a33}" +
    ".bpc-input{flex:1;background:#0b0e11;border:1px solid #232a33;" +
    "border-radius:10px;color:#e6e8ea;padding:9px 12px;outline:none}" +
    ".bpc-input:focus{border-color:#16c784}" +
    ".bpc-send{background:#16c784;border:none;border-radius:10px;" +
    "color:#0b0e11;font-weight:600;padding:0 14px;cursor:pointer}" +
    ".bpc-send:disabled{opacity:.5;cursor:default}" +
    ".bpc-note{font-size:10px;color:#5c6670;text-align:center;padding:0 12px 8px}";

  /* ---------------------------------------------------------------- dom */
  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text) node.textContent = text;
    return node;
  }

  var style = document.createElement("style");
  style.textContent = css;
  document.head.appendChild(style);

  var button = el("button", "bpc-btn", "🐂"); // ox emoji — the Black Bull
  button.setAttribute("aria-label", "Open Bullpen support chat");

  var panel = el("div", "bpc-panel");
  var head = el("div", "bpc-head");
  head.appendChild(el("span", "bpc-dot"));
  head.appendChild(el("span", "bpc-title", "Bullpen Concierge"));
  head.appendChild(el("span", "bpc-sub", "support — not financial advice"));

  var msgs = el("div", "bpc-msgs");

  var quick = el("div", "bpc-quick");
  QUICK_QUESTIONS.forEach(function (q) {
    var b = el("button", "bpc-qbtn", q);
    b.addEventListener("click", function () { send(q); });
    quick.appendChild(b);
  });

  var form = el("form", "bpc-form");
  var input = el("input", "bpc-input");
  input.placeholder = "Ask about wallets, perps, the airdrop…";
  input.maxLength = 2000;
  var sendBtn = el("button", "bpc-send", "Send");
  sendBtn.type = "submit";
  form.appendChild(input);
  form.appendChild(sendBtn);

  var note = el("div", "bpc-note",
    "AI support assistant. Never share your seed phrase — with anyone.");

  panel.appendChild(head);
  panel.appendChild(msgs);
  panel.appendChild(quick);
  panel.appendChild(form);
  panel.appendChild(note);

  document.body.appendChild(button);
  document.body.appendChild(panel);

  addBubble("bot", GREETING);

  button.addEventListener("click", function () {
    panel.classList.toggle("bpc-open");
    if (panel.classList.contains("bpc-open")) input.focus();
  });

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    send(input.value);
  });

  /* ------------------------------------------------------------ helpers */
  function addBubble(kind, text) {
    var bubble = el("div", "bpc-msg " + (kind === "user" ? "bpc-user" : "bpc-bot"));
    bubble.textContent = text;
    msgs.appendChild(bubble);
    msgs.scrollTop = msgs.scrollHeight;
    return bubble;
  }

  function addSources(sources) {
    if (!sources || !sources.length) return;
    var seen = {};
    var wrap = el("div", "bpc-srcs");
    sources.forEach(function (s) {
      var label = (s.metadata && (s.metadata.title || s.metadata.source_id)) || s.id;
      var tag = s.source_type + ": " + label;
      if (seen[tag]) return;
      seen[tag] = true;
      wrap.appendChild(el("span", "bpc-src", tag));
    });
    msgs.appendChild(wrap);
    msgs.scrollTop = msgs.scrollHeight;
  }

  /* ------------------------------------------------------------- stream */
  function send(text) {
    text = (text || "").trim();
    if (!text || busy) return;
    busy = true;
    sendBtn.disabled = true;
    input.value = "";
    quick.style.display = "none";

    addBubble("user", text);
    var botBubble = addBubble("bot", "…");
    var answer = "";
    var pendingSources = null;

    fetch(API_BASE + "/v1/chat/stream", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ message: text, history: history }),
    })
      .then(function (res) {
        if (!res.ok || !res.body) throw new Error("HTTP " + res.status);
        var reader = res.body.getReader();
        var decoder = new TextDecoder();
        var buffer = "";

        function pump() {
          return reader.read().then(function (step) {
            if (step.done) return finish();
            buffer += decoder.decode(step.value, { stream: true });
            var frames = buffer.split("\n\n");
            buffer = frames.pop(); // keep the trailing partial frame
            frames.forEach(handleFrame);
            msgs.scrollTop = msgs.scrollHeight;
            return pump();
          });
        }

        function handleFrame(frame) {
          var event = "message";
          var data = "";
          frame.split("\n").forEach(function (line) {
            if (line.indexOf("event:") === 0) event = line.slice(6).trim();
            else if (line.indexOf("data:") === 0) data += line.slice(5).trim();
          });
          if (!data) return;
          var payload;
          try { payload = JSON.parse(data); } catch (err) { return; }

          if (event === "sources") {
            pendingSources = payload;
          } else if (event === "refusal") {
            // Whole model chain refused: partial text is invalid — replace it.
            answer = payload.text;
            botBubble.textContent = answer;
          } else if (event === "error") {
            answer = "Something went wrong on my end — please try again.";
            botBubble.textContent = answer;
          } else if (payload.text) {
            answer += payload.text;
            botBubble.textContent = answer;
          }
        }

        function finish() {
          addSources(pendingSources);
          history.push({ role: "user", content: text });
          history.push({ role: "assistant", content: answer });
          if (history.length > 20) history = history.slice(-20);
        }

        return pump();
      })
      .catch(function () {
        botBubble.textContent =
          "I couldn't reach support services. Check your connection and try again.";
      })
      .finally(function () {
        busy = false;
        sendBtn.disabled = false;
        input.focus();
      });
  }
})();
