// Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
// This source code is licensed under the terms described in the LICENSE file in
// the root directory of this source tree.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

import {
  classifyLlmRequest,
  injectMediaParts,
  isGoogleGenerativeAiCall,
  loadMediaManifest,
  MediaRefError,
  redactMediaForTrace,
} from "./launch_request_utils.mjs";

assert.equal(
  isGoogleGenerativeAiCall(
    "POST",
    "https://provider.example/v1beta/models/gemini-3.1-pro:generateContent"
  ),
  true
);
assert.equal(
  isGoogleGenerativeAiCall(
    "POST",
    "https://provider.example/v1beta/models/gemini-3.1-pro:streamGenerateContent?alt=sse"
  ),
  true
);
assert.equal(
  isGoogleGenerativeAiCall(
    "POST",
    "https://provider.example/v1beta/models/gemini-3.1-pro:countTokens"
  ),
  false
);
assert.equal(
  isGoogleGenerativeAiCall(
    "GET",
    "https://provider.example/v1beta/models/gemini-3.1-pro:generateContent"
  ),
  false
);

const googleRequest = classifyLlmRequest(
  "https://provider.example/v1/models/gemini-3.1-pro:streamGenerateContent?alt=sse",
  { method: "POST" }
);
assert.equal(googleRequest.isGoogleGenerativeAi, true);
assert.equal(googleRequest.isLLMCall, true);

const openaiResponsesRequest = classifyLlmRequest(
  "https://provider.example/v1/responses",
  { method: "POST" }
);
assert.equal(openaiResponsesRequest.isResponses, true);
assert.equal(openaiResponsesRequest.isLLMCall, true);

const anthropicRequest = classifyLlmRequest(
  "https://provider.example/v1/messages",
  { method: "POST" }
);
assert.equal(anthropicRequest.isAnthropicMessages, true);
assert.equal(anthropicRequest.isLLMCall, true);

const completionsRequest = classifyLlmRequest(
  "https://provider.example/v1/chat/completions",
  { method: "POST" }
);
assert.equal(completionsRequest.isCompletions, true);
assert.equal(completionsRequest.isLLMCall, true);

const nonLlmRequest = classifyLlmRequest(
  "https://provider.example/v1/files",
  { method: "POST" }
);
assert.equal(nonLlmRequest.isLLMCall, false);

// ---------------------------------------------------------------------------
// GAIA media markers
// ---------------------------------------------------------------------------

const tmp = mkdtempSync(path.join(tmpdir(), "gaia2-media-test-"));
try {
  const root = path.join(tmp, "media");
  mkdirSync(path.join(root, "assets"), { recursive: true });
  const wav = Buffer.concat([Buffer.from("RIFF\0\0\0\0WAVEfmt "), Buffer.alloc(32, 7)]);
  const sha = createHash("sha256").update(wav).digest("hex");
  const id = "a_" + sha.slice(0, 32);
  writeFileSync(path.join(root, "assets", `${id}.wav`), wav);
  writeFileSync(path.join(tmp, "outside.wav"), wav);
  writeFileSync(
    path.join(root, "manifest.json"),
    JSON.stringify({
      version: 1,
      assets: {
        [id]: { path: `assets/${id}.wav`, mime_type: "audio/wav", sha256: sha, bytes: wav.length },
        a_badsha: { path: `assets/${id}.wav`, mime_type: "audio/wav", sha256: "0".repeat(64), bytes: 1 },
        a_escape: { path: "../outside.wav", mime_type: "audio/wav", sha256: sha, bytes: wav.length },
      },
    })
  );

  // No manifest -> bridge off.
  assert.equal(loadMediaManifest(tmp), null);
  assert.equal(loadMediaManifest(""), null);
  const media = loadMediaManifest(root);
  assert.ok(media);

  // No markers -> body untouched (same object, same bytes).
  const plain = {
    model: "m",
    messages: [
      { role: "system", content: `sys [[gaia2-media:${id}]]` },
      { role: "user", content: "hello" },
      { role: "assistant", content: `echo [[gaia2-media:${id}]]` },
    ],
  };
  const plainBytes = JSON.stringify(plain);
  assert.deepEqual(injectMediaParts(plain, media), []);
  assert.equal(JSON.stringify(plain), plainBytes);

  // String content -> text + input_audio parts; markers outside user turns ignored.
  const body = {
    messages: [
      { role: "system", content: "sys" },
      { role: "user", content: `[User request provided as audio]\n[[gaia2-media:${id}]]` },
    ],
  };
  assert.deepEqual(injectMediaParts(body, media), [{ asset_id: id, sha256: sha }]);
  assert.equal(body.messages[0].content, "sys");
  assert.deepEqual(body.messages[1].content, [
    { type: "text", text: "[User request provided as audio]\n" },
    { type: "input_audio", input_audio: { data: wav.toString("base64"), format: "wav" } },
  ]);

  // Array content keeps non-text parts and other text.
  const arr = {
    messages: [{ role: "user", content: [{ type: "image_url", image_url: { url: "x" } }, { type: "text", text: `[[gaia2-media:${id}]]` }] }],
  };
  assert.equal(injectMediaParts(arr, media).length, 1);
  assert.deepEqual(arr.messages[0].content.map((p) => p.type), ["image_url", "input_audio"]);

  // Trace copy carries {asset_id, sha256}, never base64.
  const traced = redactMediaForTrace(JSON.parse(JSON.stringify(body)), media);
  assert.deepEqual(traced.messages[1].content[1], { type: "input_audio", input_audio: { asset_id: id, sha256: sha } });
  assert.ok(!JSON.stringify(traced).includes(wav.toString("base64")));
  assert.equal(redactMediaForTrace(plain, null), plain);

  // Refusals: unknown asset, sha mismatch, path escaping the root.
  for (const [assetId, code] of [["a_unknown", "unknown_asset"], ["a_badsha", "sha_mismatch"], ["a_escape", "outside_media_root"]]) {
    assert.throws(
      () => injectMediaParts({ messages: [{ role: "user", content: `[[gaia2-media:${assetId}]]` }] }, media),
      (e) => e instanceof MediaRefError && e.code === code && !/\d/.test(e.code)
    );
  }
} finally {
  rmSync(tmp, { recursive: true, force: true });
}
