// Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
// This source code is licensed under the terms described in the LICENSE file in
// the root directory of this source tree.
import { createHash } from "node:crypto";
import { existsSync, readFileSync, realpathSync } from "node:fs";
import path from "node:path";

export function isGoogleGenerativeAiCall(method, urlStr) {
  return (
    method === "POST" &&
    urlStr.includes("/models/") &&
    /:(generateContent|streamGenerateContent)(?:[?#]|$)/.test(urlStr)
  );
}

// OpenClaw's openai-completions provider turns a forwarded `delta.reasoning`
// into a thinking block, re-sends it on every later request and counts it in
// its tool-result context guard, even when the served chat template never
// renders past reasoning. GAIA2_OPENCLAW_FORWARD_REASONING=0 (or false/off/no)
// keeps it away from OpenClaw; the raw response and trace still carry it.
export function forwardedReasoning(msg, env) {
  const flag = String(env?.GAIA2_OPENCLAW_FORWARD_REASONING ?? "").trim().toLowerCase();
  if (["0", "false", "off", "no"].includes(flag)) return "";
  return typeof msg?.reasoning === "string" ? msg.reasoning : "";
}

export function classifyLlmRequest(url, init) {
  const urlStr = typeof url === "string" ? url : url?.toString?.() || "";
  const method = init?.method?.toUpperCase() || "";
  const isCompletions =
    method === "POST" && urlStr.includes("/chat/completions");
  const isResponses = method === "POST" && urlStr.includes("/responses");
  const isAnthropicMessages =
    method === "POST" && urlStr.includes("/v1/messages");
  const isGoogleGenerativeAi = isGoogleGenerativeAiCall(method, urlStr);

  return {
    urlStr,
    isCompletions,
    isResponses,
    isAnthropicMessages,
    isGoogleGenerativeAi,
    isLLMCall:
      isCompletions ||
      isResponses ||
      isAnthropicMessages ||
      isGoogleGenerativeAi,
  };
}

// ---------------------------------------------------------------------------
// GAIA media markers -> chat-completions input_audio parts
// ---------------------------------------------------------------------------
// The adapter sends a media turn as text plus one [[gaia2-media:<asset_id>]]
// marker per asset. On each outbound chat-completions request the markers in
// user messages are replaced by input_audio parts read from
// $GAIA2_MEDIA_ROOT/manifest.json. Refusal codes stay digit-free so OpenClaw's
// retry heuristics (which match e.g. "500") never retry a refused request.

const MEDIA_MARKER_RE = /\[\[gaia2-media:([A-Za-z0-9._-]{1,128})\]\]/g;

export class MediaRefError extends Error {
  constructor(code, detail) {
    super(`${code}: ${detail}`);
    this.code = code;
  }
}

// Returns null (bridge off) when the root holds no manifest.json.
export function loadMediaManifest(root) {
  if (!root || !existsSync(path.join(root, "manifest.json"))) return null;
  const manifest = JSON.parse(readFileSync(path.join(root, "manifest.json"), "utf8"));
  return { root: realpathSync(root), assets: manifest.assets || {}, cache: new Map(), shaToId: new Map() };
}

function resolveAsset(media, assetId) {
  const hit = media.cache.get(assetId);
  if (hit) return hit;
  const entry = Object.hasOwn(media.assets, assetId) ? media.assets[assetId] : null;
  if (!entry || typeof entry.path !== "string") throw new MediaRefError("unknown_asset", assetId);
  if (entry.mime_type !== "audio/wav") throw new MediaRefError("unsupported_mime", assetId);
  let file;
  try {
    file = realpathSync(path.resolve(media.root, entry.path));
  } catch {
    throw new MediaRefError("unreadable_asset", assetId);
  }
  if (!file.startsWith(media.root + path.sep)) throw new MediaRefError("outside_media_root", assetId);
  const bytes = readFileSync(file);
  const sha256 = createHash("sha256").update(bytes).digest("hex");
  if (sha256 !== entry.sha256) throw new MediaRefError("sha_mismatch", assetId);
  const resolved = {
    part: { type: "input_audio", input_audio: { data: bytes.toString("base64"), format: "wav" } },
    audit: { asset_id: assetId, sha256 },
  };
  media.cache.set(assetId, resolved);
  media.shaToId.set(sha256, assetId);
  return resolved;
}

function splitOnMarkers(text, media, delivered) {
  const parts = [];
  let last = 0;
  for (const m of text.matchAll(MEDIA_MARKER_RE)) {
    const before = text.slice(last, m.index);
    if (before.trim()) parts.push({ type: "text", text: before });
    const { part, audit } = resolveAsset(media, m[1]);
    parts.push(part);
    delivered.push(audit);
    last = m.index + m[0].length;
  }
  if (last === 0) return null;
  const after = text.slice(last);
  if (after.trim()) parts.push({ type: "text", text: after });
  return parts;
}

// Rewrites marker-bearing user messages of a parsed request body in place and
// returns [{asset_id, sha256}] per delivered asset ([] -> body untouched).
// Throws MediaRefError when a marker cannot be resolved.
export function injectMediaParts(reqBody, media) {
  const delivered = [];
  for (const msg of Array.isArray(reqBody?.messages) ? reqBody.messages : []) {
    if (msg?.role !== "user") continue;
    if (typeof msg.content === "string") {
      const parts = splitOnMarkers(msg.content, media, delivered);
      if (parts) msg.content = parts;
    } else if (Array.isArray(msg.content)) {
      const out = [];
      let found = false;
      for (const item of msg.content) {
        const parts = item?.type === "text" && typeof item.text === "string"
          ? splitOnMarkers(item.text, media, delivered) : null;
        found ||= parts !== null;
        out.push(...(parts ?? [item]));
      }
      if (found) msg.content = out;
    }
  }
  return delivered;
}

// Trace copy of a request body: each input_audio part becomes
// {asset_id, sha256} of the bytes actually sent, never the base64 itself.
export function redactMediaForTrace(body, media) {
  if (!media || !Array.isArray(body?.messages)) return body;
  for (const msg of body.messages) {
    if (!Array.isArray(msg?.content)) continue;
    msg.content = msg.content.map((p) => {
      if (p?.type !== "input_audio" || typeof p.input_audio?.data !== "string") return p;
      const sha256 = createHash("sha256").update(Buffer.from(p.input_audio.data, "base64")).digest("hex");
      return { type: "input_audio", input_audio: { asset_id: media.shaToId.get(sha256) ?? null, sha256 } };
    });
  }
  return body;
}
