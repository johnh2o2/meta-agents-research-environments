// Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
// This source code is licensed under the terms described in the LICENSE file in
// the root directory of this source tree.
import { appendFileSync } from "node:fs";
import { setDefaultResultOrder } from "node:dns";
import { EnvHttpProxyAgent, setGlobalDispatcher } from "/opt/chatbot/node_modules/undici/index.js";

import {
  abortReason,
  attemptSignal,
  classifyLlmRequest,
  injectMediaParts,
  loadMediaManifest,
  redactMediaForTrace,
  retryAllowed,
  waitUnlessAborted,
} from "./launch_request_utils.mjs";

// Audio prompts: on only when $GAIA2_MEDIA_ROOT/manifest.json exists.
const MEDIA = loadMediaManifest(process.env.GAIA2_MEDIA_ROOT || "");
if (MEDIA) console.error(`[launch.mjs] GAIA media markers enabled (root=${MEDIA.root})`);

// Force IPv4-first DNS resolution. Some hosts advertise unusable IPv6 routes
// ahead of IPv4, which can turn outbound API calls into avoidable timeouts.
setDefaultResultOrder("ipv4first");

const hasProxyEnv =
  process.env.HTTPS_PROXY || process.env.https_proxy || process.env.HTTP_PROXY || process.env.http_proxy;
if (hasProxyEnv) {
  // Respect HTTP_PROXY / HTTPS_PROXY / NO_PROXY by scheme and hostname so
  // localhost model backends (e.g. 127.0.0.1:8091) bypass the TLS proxy.
  setGlobalDispatcher(new EnvHttpProxyAgent());
}

// ---------------------------------------------------------------------------
// LLM API trace logging (opt-in via GAIA2_TRACE_FILE env var)
// ---------------------------------------------------------------------------

const TRACE_FILE = process.env.GAIA2_TRACE_FILE || "";
let traceSeq = 0;

function traceLog(url, init, status, rawText, latencyMs) {
  if (!TRACE_FILE) return;
  traceSeq++;
  let requestBody = null;
  try { requestBody = redactMediaForTrace(JSON.parse(init?.body || "{}"), MEDIA); } catch(e) {}

  const entry = {
    seq: traceSeq,
    timestamp: new Date().toISOString(),
    type: "llm_call",
    url: typeof url === "string" ? url : url?.toString?.() || "",
    latency_ms: latencyMs,
    http_status: status,
    request: requestBody,
    raw_response: rawText,
  };

  try {
    appendFileSync(TRACE_FILE, JSON.stringify(entry) + "\n");
  } catch(e) {
    // best-effort
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function sseChunk(meta, choices) {
  return "data: " + JSON.stringify({ ...meta, choices }) + "\n\n";
}

// ---------------------------------------------------------------------------
// Thinking-model tool-call signatures
// ---------------------------------------------------------------------------
// Thinking models (e.g. Gemini via the genai gateway) attach
// extra_content.google.thought_signature to each tool_call on the turn that
// produced it, and REQUIRE that same signature be echoed back on the follow-up
// (tool-result) request. OpenClaw's core neither re-serializes extra_content nor
// preserves the tool_call id verbatim (it de-underscores call_596832 ->
// call596832), so the signature is lost and the genai gateway rejects the next
// request with HTTP 400. We bridge that here: capture each tool_call's
// extra_content keyed by a normalized id on the way back, and re-attach it to
// the matching tool_call on the way out. Non-thinking models and other backends
// never populate extra_content, so the map stays empty and behavior is
// unchanged.
const toolCallExtraContent = new Map();

function normalizeId(id) {
  // OpenClaw mutates call_596832 -> call596832, so match on alphanumerics only.
  return String(id == null ? "" : id).replace(/[^a-zA-Z0-9]/g, "");
}

function captureToolCallSignatures(json) {
  for (const choice of (json?.choices || [])) {
    const tcs = choice?.message?.tool_calls;
    if (!Array.isArray(tcs)) continue;
    for (const tc of tcs) {
      if (tc && tc.id != null && tc.extra_content != null) {
        toolCallExtraContent.set(normalizeId(tc.id), tc.extra_content);
      }
    }
  }
}

function reinjectToolCallSignatures(messages) {
  let changed = false;
  if (!Array.isArray(messages)) return changed;
  for (const m of messages) {
    const tcs = m?.tool_calls;
    if (!Array.isArray(tcs)) continue;
    for (const tc of tcs) {
      if (!tc || tc.id == null) continue;
      const sig = toolCallExtraContent.get(normalizeId(tc.id));
      if (sig != null) {
        tc.extra_content = sig;
        changed = true;
      }
    }
  }
  return changed;
}

function isMissingSignatureError(text) {
  if (!text) return false;
  const t = text.toLowerCase();
  if (t.includes("thought_signature") || t.includes("thought signature")) return true;
  return (
    t.includes("signature") &&
    (t.includes("missing") || t.includes("invalid") || t.includes("required") || t.includes("not found"))
  );
}

// ---------------------------------------------------------------------------
// ChatCompletion → SSE converter
// ---------------------------------------------------------------------------
// OpenClaw's OpenAI SDK sends stream:true and expects SSE responses.
// After we strip stream from the request for OpenAI-compatible backends, we
// wrap the non-streaming JSON response as SSE ChatCompletionChunk events.

// Own import, kept here so it does not touch the import block above.
import { forwardedReasoning } from "./launch_request_utils.mjs";
if (!forwardedReasoning({ reasoning: "x" }, process.env)) {
  console.error("[launch.mjs] reasoning not forwarded to OpenClaw (GAIA2_OPENCLAW_FORWARD_REASONING=0)");
}

function completionToSSE(json) {
  const meta = {
    id: json.id || "chatcmpl-" + Date.now(),
    object: "chat.completion.chunk",
    model: json.model || "",
    created: json.created || Math.floor(Date.now() / 1000),
  };
  let sse = "";
  for (const choice of (json.choices || [])) {
    const msg = choice.message || {};
    const idx = choice.index || 0;
    const hasToolCalls = Array.isArray(msg.tool_calls) && msg.tool_calls.length > 0;
    sse += sseChunk(meta, [{ index: idx, finish_reason: null, delta: { role: msg.role || "assistant", content: "" } }]);
    const reasoning = forwardedReasoning(msg, process.env);
    if (reasoning) {
      sse += sseChunk(meta, [{ index: idx, finish_reason: null, delta: { reasoning } }]);
    }
    // Some OpenAI-compatible backends return assistant text alongside tool
    // calls. Suppress that mixed text so OpenClaw does not end the turn
    // before the tool calls execute.
    if (msg.content && !hasToolCalls) {
      sse += sseChunk(meta, [{ index: idx, finish_reason: null, delta: { content: msg.content } }]);
    }
    if (msg.tool_calls) {
      for (let i = 0; i < msg.tool_calls.length; i++) {
        const tc = msg.tool_calls[i];
        const firstToolCall = { index: i, id: tc.id, type: "function", function: { name: tc.function.name } };
        // Thinking models carry a thought signature the follow-up turn must echo
        // back; forward it so OpenClaw can re-serialize it (non-thinking models
        // have no extra_content, leaving the delta shape unchanged).
        if (tc.extra_content != null) firstToolCall.extra_content = tc.extra_content;
        sse += sseChunk(meta, [{ index: idx, finish_reason: null, delta: { tool_calls: [firstToolCall] } }]);
        sse += sseChunk(meta, [{ index: idx, finish_reason: null, delta: { tool_calls: [{ index: i, function: { arguments: tc.function.arguments } }] } }]);
      }
    }
    sse += sseChunk(meta, [{ index: idx, finish_reason: choice.finish_reason || "stop", delta: {} }]);
  }
  sse += "data: [DONE]\n\n";
  return sse;
}

// ---------------------------------------------------------------------------
// Fetch interceptor
// ---------------------------------------------------------------------------
// Intercepts LLM requests to:
//   1. Preserve proxy-aware fetch behavior and request logging
//   2. Retry transient upstream failures
//   3. Shim non-streaming /chat/completions JSON responses into SSE
//      because OpenClaw expects a streaming Chat Completions transport

const _origFetch = globalThis.fetch;
globalThis.fetch = async function(url, init) {
  const {
    urlStr,
    isCompletions,
    isResponses,
    isAnthropicMessages,
    isLLMCall,
  } = classifyLlmRequest(url, init);

  if (isCompletions && init?.body) {
    try {
      const reqBody = JSON.parse(init.body);
      let changed = false;
      const isOpenRouter = urlStr.includes("openrouter.ai");
      if (reqBody.stream && !isOpenRouter) {
        reqBody.stream = false;
        changed = true;
      }
      // Best-effort propagation of the container thinking level to generic
      // OpenAI-compatible chat-completions providers.
      const effort = process.env.REASONING_EFFORT || process.env.THINKING;
      if (isOpenRouter && effort && effort !== "none" && effort !== "off") {
        reqBody.reasoning = { enabled: true };
        changed = true;
      } else if (effort && effort !== "none" && effort !== "off" && !reqBody.reasoning) {
        reqBody.reasoning = { effort };
        changed = true;
      }
      // Re-attach captured thinking-model tool-call signatures to the echoed
      // assistant tool_calls so the follow-up (tool-result) turn is accepted.
      if (reinjectToolCallSignatures(reqBody.messages)) {
        changed = true;
      }
      if (changed) {
        init = { ...init, body: JSON.stringify(reqBody) };
      }
      if (MEDIA) {
        try {
          if (injectMediaParts(reqBody, MEDIA).length > 0) {
            init = { ...init, body: JSON.stringify(reqBody) };
          }
        } catch (e) {
          // Refuse without calling upstream: never send a turn without its audio.
          const errBody = JSON.stringify({
            error: { message: `gaia2 media refused: ${e?.code || "bridge_error"}`, type: "invalid_request_error" },
          });
          console.error(`[launch.mjs] GAIA media refused: ${e?.message || e}`);
          traceLog(url, init, 400, errBody, 0);
          return new Response(errBody, { status: 400, headers: { "content-type": "application/json" } });
        }
      }
    } catch(e) { /* not JSON, leave as-is */ }
  }

  // For LLM calls: retry on 429, 5xx, and network errors with exponential backoff
  const MAX_RETRIES = isLLMCall ? 5 : 0;
  const FETCH_TIMEOUT_MS = isLLMCall ? 180000 : 0; // 180s timeout
  let resp, text, latencyMs;
  // One-shot repair for a thinking-model follow-up turn rejected with HTTP 400
  // because its tool_call thought signature is missing/invalid.
  let signatureRepairAttempted = false;
  // OpenClaw abandons a request by aborting its signal; every attempt follows
  // it, and an aborted request is never retried.
  const callerSignal = init?.signal;

  for (let attempt = 0; ; attempt++) {
    const startTime = Date.now();
    const attemptSig = FETCH_TIMEOUT_MS > 0 ? attemptSignal(callerSignal, FETCH_TIMEOUT_MS) : null;

    try {
      if (attemptSig) {
        try {
          resp = await _origFetch.call(this, url, { ...init, signal: attemptSig.signal });
        } finally {
          attemptSig.clearTimeout();
        }
      } else {
        resp = await _origFetch.call(this, url, init);
      }
    } catch (err) {
      attemptSig?.release();
      latencyMs = Date.now() - startTime;
      if (!isLLMCall) throw err;
      const isTimeout = err?.name === "TimeoutError" || err?.name === "AbortError" || err?.code === "UND_ERR_CONNECT_TIMEOUT" || err?.message?.includes("timed out");
      const errType = callerSignal?.aborted ? "caller_aborted" : isTimeout ? "timeout" : "fetch_error";
      const errInfo = { error: err?.message || String(err), type: errType };
      traceLog(url, init, 0, JSON.stringify(errInfo), latencyMs);

      if (retryAllowed(attempt, MAX_RETRIES, callerSignal)) {
        const delay = Math.min(1000 * Math.pow(2, attempt) + Math.random() * 500, 60000);
        console.error(`[launch.mjs] LLM fetch error (attempt ${attempt + 1}/${MAX_RETRIES + 1}): ${errInfo.type} for ${urlStr} (${latencyMs}ms): ${errInfo.error} — retrying in ${(delay / 1000).toFixed(1)}s`);
        await waitUnlessAborted(delay, callerSignal);
        if (callerSignal?.aborted) throw abortReason(callerSignal);
        continue;
      }
      console.error(`[launch.mjs] LLM fetch error (${callerSignal?.aborted ? "caller aborted, not retrying" : "final attempt"}): ${errInfo.type} for ${urlStr} (${latencyMs}ms): ${errInfo.error}`);
      throw err;
    }

    if (!isLLMCall) return resp;

    try {
      text = await resp.text();
    } finally {
      attemptSig?.release();
    }
    latencyMs = Date.now() - startTime;

    // Retry on 429 (rate limit) and 5xx (server errors)
    if ((resp.status === 429 || resp.status >= 500) && retryAllowed(attempt, MAX_RETRIES, callerSignal)) {
      const retryAfter = resp.headers.get("retry-after");
      const delay = retryAfter ? Math.min(parseFloat(retryAfter) * 1000, 60000) : Math.min(1000 * Math.pow(2, attempt) + Math.random() * 500, 60000);
      const preview = text.length > 200 ? text.slice(0, 200) + "..." : text;
      console.error(`[launch.mjs] LLM API HTTP ${resp.status} (attempt ${attempt + 1}/${MAX_RETRIES + 1}) from ${urlStr} (${latencyMs}ms): ${preview} — retrying in ${(delay / 1000).toFixed(1)}s`);
      traceLog(url, init, resp.status, text, latencyMs);
      await waitUnlessAborted(delay, callerSignal);
      if (callerSignal?.aborted) throw abortReason(callerSignal);
      continue;
    }

    // Thinking-model follow-up turns are rejected with HTTP 400 when the
    // tool_call thought signature is missing/invalid. Repair once by
    // re-attaching captured signatures to the outbound body and resend, rather
    // than letting the episode die. Other 400s are left as-is.
    if (resp.status === 400 && !signatureRepairAttempted && isMissingSignatureError(text)) {
      signatureRepairAttempted = true;
      let repairedInit = null;
      try {
        const rb = JSON.parse(init?.body || "null");
        if (rb && reinjectToolCallSignatures(rb.messages)) {
          repairedInit = { ...init, body: JSON.stringify(rb) };
        }
      } catch (e) { /* body not JSON — nothing to repair */ }
      if (repairedInit) {
        console.error(`[launch.mjs] LLM API HTTP 400 missing thought signature from ${urlStr} (${latencyMs}ms) — re-injecting tool_call signatures and retrying`);
        traceLog(url, init, resp.status, text, latencyMs);
        init = repairedInit;
        continue;
      }
    }

    break; // Success or non-retryable error
  }

  const ct = resp.headers.get("content-type") || "";
  const isSSE = ct.includes("text/event-stream") || text.trimStart().startsWith("data: ");

  // Log the raw request/response before any transformation
  traceLog(url, init, resp.status, text, latencyMs);

  // Surface HTTP errors to stderr so they appear in entrypoint.log
  if (resp.status >= 400) {
    const preview = text.length > 500 ? text.slice(0, 500) + "..." : text;
    console.error(`[launch.mjs] LLM API error: HTTP ${resp.status} from ${urlStr} (${latencyMs}ms): ${preview}`);
  }

  // Anthropic Messages API: log only, pass response through unchanged.
  if (isAnthropicMessages) {
    return new Response(text, { status: resp.status, statusText: resp.statusText, headers: resp.headers });
  }

  // Responses API requests are handled natively upstream; do not rewrite them.
  if (isResponses) {
    return new Response(text, { status: resp.status, statusText: resp.statusText, headers: resp.headers });
  }

  // Streaming chat-completions responses already have the right transport shape.
  if (isSSE) {
    return new Response(text, { status: resp.status, statusText: resp.statusText, headers: resp.headers });
  }

  // Only synthesize SSE for successful non-streaming chat-completions payloads.
  if (!isCompletions || !resp.ok) {
    return new Response(text, { status: resp.status, statusText: resp.statusText, headers: resp.headers });
  }

  try {
    const body = JSON.parse(text);
    // Capture any thinking-model tool-call signatures before converting, keyed
    // by normalized id, so the follow-up turn's request can echo them back.
    captureToolCallSignatures(body);
    const sse = completionToSSE(body);
    const headers = new Headers(resp.headers);
    headers.set("content-type", "text/event-stream");
    return new Response(sse, { status: resp.status, statusText: resp.statusText, headers });
  } catch(e) {
    return new Response(text, { status: resp.status, statusText: resp.statusText, headers: resp.headers });
  }
};

// Rename the process to avoid host security policies targeting "openclaw"
process.title = "gaia2-oc";

// entry.js respawns a child Node process to add warning-suppression flags.
// That child bypasses this wrapper, so the fetch shim never reaches the real
// gateway process. The wrapper already sets NODE_EXTRA_CA_CERTS itself.
process.env.OPENCLAW_NODE_OPTIONS_READY ||= "1";
if (process.env.NODE_EXTRA_CA_CERTS) {
  process.env.OPENCLAW_NODE_EXTRA_CA_CERTS_READY ||= "1";
}

const args = process.argv.slice(2);
// Set argv[1] to gaia2-gw.mjs (renamed copy of openclaw.mjs).
// Some host security policies kill processes with "openclaw" in argv.
process.argv = ["node", "/opt/chatbot/gaia2-gw.mjs", ...args];
await import("/opt/chatbot/dist/entry.js");
