(() => {
  const chatWindow = document.getElementById("chat-window");
  const chatForm = document.getElementById("chat-form");
  const userInput = document.getElementById("user-input");
  const sendBtn = document.getElementById("send-btn");
  const statusEl = document.getElementById("chat-status");
  const rerankerToggle = document.getElementById("reranker-toggle");

  // Persist a lightweight client session ID for multi-turn context
  const SESSION_KEY = "music_rag_session_id";
  let sessionId = window.localStorage.getItem(SESSION_KEY);
  if (!sessionId) {
    sessionId =
      "web-" +
      Date.now().toString(36) +
      "-" +
      Math.random().toString(36).slice(2, 8);
    window.localStorage.setItem(SESSION_KEY, sessionId);
  }

  function appendMessage({ role, content, isHtml = false }) {
    const messageEl = document.createElement("div");
    messageEl.className = `message ${role}`;

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = role === "user" ? "You" : "AI";

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    if (isHtml) {
      bubble.innerHTML = content;
    } else {
      const p = document.createElement("p");
      p.textContent = content;
      bubble.appendChild(p);
    }

    messageEl.appendChild(avatar);
    messageEl.appendChild(bubble);
    chatWindow.appendChild(messageEl);
    chatWindow.scrollTop = chatWindow.scrollHeight;
  }

  async function sendQuestion(question) {
    const useReranker = !!rerankerToggle?.checked;

    try {
      statusEl.textContent = "Thinking...";
      sendBtn.disabled = true;
      userInput.disabled = true;

      const resp = await fetch("/api/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question,
          session_id: sessionId,
          use_reranker: useReranker,
          k: 10,
        }),
      });

      if (!resp.ok) {
        throw new Error(`Request failed with status ${resp.status}`);
      }

      const data = await resp.json();
      if (data.session_id && data.session_id !== sessionId) {
        sessionId = data.session_id;
        window.localStorage.setItem(SESSION_KEY, sessionId);
      }

      appendMessage({
        role: "assistant",
        content: data.answer_html || "No answer returned.",
        isHtml: true,
      });
      statusEl.textContent = "";
    } catch (err) {
      console.error(err);
      statusEl.textContent =
        "Something went wrong talking to the API. Check the backend logs.";
    } finally {
      sendBtn.disabled = false;
      userInput.disabled = false;
      userInput.focus();
    }
  }

  chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const text = userInput.value.trim();
    if (!text) return;

    appendMessage({ role: "user", content: text });
    userInput.value = "";
    sendQuestion(text);
  });

  // Optional: submit with Ctrl+Enter while allowing Shift+Enter for newlines
  userInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      chatForm.requestSubmit();
    }
  });
})();


