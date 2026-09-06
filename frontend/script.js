"use strict";

(() => {
  const HISTORY_KEY = "fakenews.history";
  const HISTORY_LIMIT = 20;
  const MAX_TITLE_LENGTH = 80;

  const FRIENDLY_ERRORS = {
    invalid_url: "We couldn't find an article at this URL. Check the link and try again.",
    http_error: "We couldn't find an article at this URL. Check the link and try again.",
    dns_failure: "We couldn't find an article at this URL. Check the link and try again.",
    not_html: "We couldn't find an article at this URL. Check the link and try again.",
    blocked_domain: "This URL type is not allowed for analysis.",
    blocked_network: "This URL points to a private network and is blocked.",
    timeout: "The article could not be retrieved right now. Please try again.",
    connection_error: "The article could not be retrieved right now. Please try again.",
    too_many_redirects: "The article could not be retrieved right now. Please try again.",
    oversize: "The article could not be retrieved right now. Please try again.",
    not_article: "This page doesn't appear to contain a single news article.",
    extraction_failed: "We retrieved the page, but couldn't identify the article content.",
  };

  const elements = {
    tabButtons: document.querySelectorAll(".tab"),
    panels: document.querySelectorAll(".tab-panel"),
    news: document.getElementById("news"),
    newsUrl: document.getElementById("news-url"),
    analyzeBtn: document.getElementById("analyze-btn"),
    btnLabel: document.querySelector(".btn-label"),
    spinner: document.querySelector(".spinner"),
    clearBtn: document.getElementById("clear-btn"),
    error: document.getElementById("error-message"),
    resultSection: document.getElementById("result-section"),
    sourceBadge: document.getElementById("source-badge"),
    sourceUrl: document.getElementById("source-url"),
    articleTitle: document.getElementById("article-title"),
    verdict: document.getElementById("verdict"),
    verdictHint: document.getElementById("verdict-hint"),
    confidence: document.getElementById("confidence"),
    probReal: document.getElementById("prob-real"),
    probFake: document.getElementById("prob-fake"),
    explanationList: document.getElementById("explanation-list"),
    historySection: document.getElementById("history-section"),
    historyList: document.getElementById("history-list"),
    clearHistoryBtn: document.getElementById("clear-history-btn"),
    liveRegion: document.getElementById("live-region"),
    resultSkeleton: document.getElementById("result-skeleton"),
    resultContent: document.getElementById("result-content"),
    processTime: document.getElementById("process-time"),
    copyResultBtn: document.getElementById("copy-result-btn"),
  };

  let activeTab = "text";

  // ------------------------------------------------------------------ //
  // Tab switching
  // ------------------------------------------------------------------ //
  elements.tabButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const tab = button.dataset.tab;
      activateTab(tab);
    });
  });

  function activateTab(tab) {
    activeTab = tab;
    elements.tabButtons.forEach((b) =>
      b.classList.toggle("active", b.dataset.tab === tab),
    );
    elements.panels.forEach((p) =>
      p.classList.toggle("hidden", p.dataset.panel !== tab),
    );
    clearError();
    hideResult();
  }

  // ------------------------------------------------------------------ //
  // Analysis
  // ------------------------------------------------------------------ //
  elements.analyzeBtn.addEventListener("click", () => analyze());
  elements.clearBtn.addEventListener("click", clearAll);
  elements.clearHistoryBtn.addEventListener("click", clearHistory);
  elements.copyResultBtn.addEventListener("click", copyResult);

  // Submit the URL field with the Enter key.
  elements.newsUrl.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      analyze();
    }
  });

  // Ctrl/Cmd + Enter submits the pasted textarea.
  elements.news.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      analyze();
    }
  });

  function getPayload() {
    if (activeTab === "text") {
      return { news: elements.news.value };
    }
    return { url: elements.newsUrl.value.trim() };
  }

  function validate(payload) {
    if (activeTab === "text") {
      if (!payload.news || !payload.news.trim()) {
        return "Please enter an article to analyze.";
      }
      if (payload.news.trim().length < 10) {
        return "Please enter enough text to analyze (at least 10 characters).";
      }
    } else {
      if (!payload.url) {
        return "Please enter a URL to analyze.";
      }
      try {
        new URL(payload.url);
      } catch (_) {
        return "Please enter a valid URL (e.g. https://example.com).";
      }
    }
    return null;
  }

  async function analyze() {
    const payload = getPayload();
    const validationError = validate(payload);
    if (validationError) {
      showError(validationError);
      return;
    }

    setLoading();
    clearError();
    hideResult();
    showSkeleton();
    const startTime = performance.now();

    const endpoint = activeTab === "text" ? "/predict" : "/predict-url";

    try {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        let message = `Request failed (HTTP ${response.status}).`;
        try {
          const body = await response.json();
          if (body && body.detail) {
            if (typeof body.detail === "string") {
              message = body.detail;
            } else {
              message = body.detail.map((d) => d.msg).join(", ");
            }
          } else if (body && body.category && FRIENDLY_ERRORS[body.category]) {
            message = FRIENDLY_ERRORS[body.category];
          }
        } catch (_) {
          // keep the default message if the response is not JSON
        }
        throw new Error(message);
      }

      const result = await response.json();
      const elapsedMs = Math.round(performance.now() - startTime);
      result._elapsed = (elapsedMs / 1000).toFixed(2);
      result._processing_time_ms = elapsedMs;
      renderResult(result);
      saveHistory(result);
      renderHistory();
    } catch (err) {
      hideResult();
      if (err instanceof TypeError && err.message === "Failed to fetch") {
        showError(
          "Unable to connect to the detector server. Please make sure the backend is running.",
        );
      } else {
        showError(err.message || "Something went wrong. Please try again.");
      }
    } finally {
      clearLoading();
      hideSkeleton();
    }
  }

  function showSkeleton() {
    elements.resultSection.classList.remove("hidden");
    elements.resultSkeleton.classList.remove("hidden");
    elements.resultContent.classList.add("hidden");
  }

  function hideSkeleton() {
    elements.resultSkeleton.classList.add("hidden");
    elements.resultContent.classList.remove("hidden");
  }

  function setLoading() {
    elements.analyzeBtn.disabled = true;
    elements.analyzeBtn.classList.add("loading");
    elements.analyzeBtn.setAttribute("aria-busy", "true");
    elements.btnLabel.textContent = "Analysing…";
    elements.spinner.classList.remove("hidden");
  }

  function clearLoading() {
    elements.analyzeBtn.disabled = false;
    elements.analyzeBtn.classList.remove("loading");
    elements.analyzeBtn.removeAttribute("aria-busy");
    elements.btnLabel.textContent = "Analyse";
    elements.spinner.classList.add("hidden");
  }

  // ------------------------------------------------------------------ //
  // Rendering
  // ------------------------------------------------------------------ //
  function renderResult(result) {
    const label = result.label;
    const sourceType = result.source_type || "text";

    elements.resultSection.classList.remove("hidden");
    elements.sourceBadge.textContent =
      sourceType === "url" ? "URL source" : "Pasted text";
    elements.sourceUrl.classList.toggle("hidden", !result.source);
    if (result.source) {
      elements.sourceUrl.textContent = `Source: ${result.source}`;
    }
    elements.articleTitle.classList.toggle("hidden", !result.page_title);
    if (result.page_title) {
      elements.articleTitle.textContent = result.page_title.slice(0, MAX_TITLE_LENGTH);
      elements.articleTitle.title = result.page_title;
    }

    const verdictEl = elements.verdict;
    verdictEl.textContent =
      label === "uncertain" ? "Uncertain" : label.charAt(0).toUpperCase() + label.slice(1);
    const box = verdictEl.closest(".verdict");
    box.classList.remove("verdict-real", "verdict-fake", "verdict-uncertain");
    box.classList.add(`verdict-${label}`);
    elements.verdictHint.classList.toggle("hidden", label !== "uncertain");

    elements.confidence.textContent = `${result.confidence}%`;
    elements.probReal.textContent = `${result.probability_real}%`;
    elements.probFake.textContent = `${result.probability_fake}%`;
    if (result._elapsed) {
      elements.processTime.textContent = `${result._elapsed}s`;
    }

    renderExplanation(result.explanation);
    elements.liveRegion.textContent =
      `Verdict: ${label}. Confidence ${result.confidence} percent.`;
    window.scrollTo({ top: elements.resultSection.offsetTop - 20, behavior: "smooth" });
  }

  function renderExplanation(explanation) {
    elements.explanationList.innerHTML = "";
    if (!explanation || !explanation.top_influential_words.length) {
      elements.explanationList.innerHTML =
        '<p class="empty-history">No influential words available.</p>';
      return;
    }
    explanation.top_influential_words.forEach((item) => {
      const chip = document.createElement("div");
      chip.className = "word-chip";

      const word = document.createElement("span");
      word.className = "word";
      word.textContent = item.word;

      const impact = document.createElement("span");
      impact.className = "impact";
      impact.textContent = `${item.impact}`;

      const tag = document.createElement("span");
      tag.className = `tag tag-${item.direction === "real" ? "real" : "fake"}`;
      tag.textContent = item.direction === "real" ? "toward real" : "toward fake";

      chip.append(word, impact, tag);
      elements.explanationList.appendChild(chip);
    });
  }

  function hideResult() {
    elements.resultSection.classList.add("hidden");
  }

  // ------------------------------------------------------------------ //
  // History (localStorage)
  // ------------------------------------------------------------------ //
  function loadHistory() {
    try {
      const raw = localStorage.getItem(HISTORY_KEY);
      const parsed = raw ? JSON.parse(raw) : [];
      return Array.isArray(parsed) ? parsed : [];
    } catch (_) {
      return [];
    }
  }

  function saveHistory(result) {
    let history = loadHistory();
    const snippet =
      activeTab === "text"
        ? elements.news.value.trim().replace(/\s+/g, " ").slice(0, 60)
        : elements.newsUrl.value.trim();
    const entry = {
      id: Date.now().toString(36) + Math.random().toString(36).slice(2, 7),
      timestamp: Date.now(),
      title: snippet || "Untitled analysis",
      source_type: result.source_type || (activeTab === "url" ? "url" : "text"),
      label: result.label,
      confidence: result.confidence,
      probability_real: result.probability_real,
      probability_fake: result.probability_fake,
      processing_time_ms: result._processing_time_ms || null,
    };
    history.unshift(entry);
    if (history.length > HISTORY_LIMIT) {
      history = history.slice(0, HISTORY_LIMIT);
    }
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
    } catch (_) {
      // storage full or unavailable — ignore, history is best-effort
    }
  }

  function renderHistory() {
    const history = loadHistory();
    elements.historySection.classList.toggle("hidden", history.length === 0);
    elements.historyList.innerHTML = "";

    if (history.length === 0) {
      const empty = document.createElement("li");
      empty.className = "empty-history";
      empty.textContent = "No previous analyses yet.";
      elements.historyList.appendChild(empty);
      return;
    }

    history.forEach((entry) => {
      const li = document.createElement("li");
      li.className = "history-item";

      const meta = document.createElement("div");
      meta.className = "history-meta";

      const title = document.createElement("div");
      title.className = "history-title";
      title.textContent = entry.title;
      title.title = entry.title;

      const subtitle = document.createElement("div");
      subtitle.className = "history-subtitle";
      const timeInfo =
        entry.processing_time_ms != null
          ? ` · ${entry.processing_time_ms}ms`
          : "";
      subtitle.textContent =
        formatDate(entry.timestamp) +
        " · " +
        (entry.source_type === "url" ? "URL" : "text") +
        timeInfo;

      meta.append(title, subtitle);

      const result = document.createElement("div");
      result.className = "history-result";

      const verdict = document.createElement("span");
      verdict.className = `history-verdict ${entry.label}`;
      verdict.textContent =
        entry.label === "uncertain"
          ? "Uncertain"
          : entry.label.charAt(0).toUpperCase() + entry.label.slice(1);

      const confidence = document.createElement("div");
      confidence.className = "history-confidence";
      confidence.textContent = `${entry.confidence}% confidence`;

      result.append(verdict, confidence);

      li.append(meta, result);
      li.addEventListener("click", () => openHistory(entry));
      elements.historyList.appendChild(li);
    });
  }

  function openHistory(entry) {
    elements.resultSection.classList.remove("hidden");
    const label = entry.label;
    elements.sourceBadge.textContent =
      entry.source_type === "url" ? "URL source" : "Pasted text";
    elements.articleTitle.classList.add("hidden");

    const verdictEl = elements.verdict;
    verdictEl.textContent =
      label === "uncertain" ? "Uncertain" : label.charAt(0).toUpperCase() + label.slice(1);
    const box = verdictEl.closest(".verdict");
    box.classList.remove("verdict-real", "verdict-fake", "verdict-uncertain");
    box.classList.add(`verdict-${label}`);
    elements.verdictHint.classList.toggle("hidden", label !== "uncertain");

    elements.confidence.textContent = `${entry.confidence}%`;
    elements.probReal.textContent = `${entry.probability_real}%`;
    elements.probFake.textContent = `${entry.probability_fake}%`;
    elements.explanationList.innerHTML =
      '<p class="empty-history">Saved history entry (explanation not stored).</p>';
    window.scrollTo({ top: elements.resultSection.offsetTop - 20, behavior: "smooth" });
  }

  function clearHistory() {
    try {
      localStorage.removeItem(HISTORY_KEY);
    } catch (_) {
      // ignore
    }
    renderHistory();
  }

  function formatDate(ts) {
    try {
      return new Date(ts).toLocaleString();
    } catch (_) {
      return "Unknown time";
    }
  }

  // ------------------------------------------------------------------ //
  // Misc helpers
  // ------------------------------------------------------------------ //
  function showError(message) {
    elements.error.textContent = message;
    elements.error.classList.remove("hidden");
  }

  function clearError() {
    elements.error.textContent = "";
    elements.error.classList.add("hidden");
  }

  function clearAll() {
    elements.news.value = "";
    elements.newsUrl.value = "";
    clearError();
    hideResult();
    hideSkeleton();
    elements.news.focus();
  }

  function copyResult() {
    const verdict = elements.verdict.textContent;
    const conf = elements.confidence.textContent;
    const pr = elements.probReal.textContent;
    const pf = elements.probFake.textContent;
    const text = `Verdict: ${verdict} | Confidence: ${conf} | Real: ${pr} | Fake: ${pf}`;
    navigator.clipboard.writeText(text).catch(() => {});
  }

  renderHistory();
})();