// Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
// This source code is licensed under the terms described in the LICENSE file in
// the root directory of this source tree.
//
// The fetch shim in launch.mjs must follow the caller's abort signal: an
// abandoned LLM request stops, and is not retried. Runs launch.mjs itself with
// the image-only modules (/opt/chatbot/...) stubbed out by a resolve hook.
import assert from "node:assert/strict";
import { register } from "node:module";

import {
  attemptSignal,
  retryAllowed,
  waitUnlessAborted,
} from "./launch_request_utils.mjs";

// -- helpers ---------------------------------------------------------------

{
  const caller = new AbortController();
  const a = attemptSignal(caller.signal, 60000);
  assert.equal(a.signal.aborted, false);
  caller.abort(new Error("caller gave up"));
  assert.equal(a.signal.aborted, true);
  assert.equal(a.signal.reason.message, "caller gave up");
  a.release();
}
{
  const caller = new AbortController();
  caller.abort();
  assert.equal(attemptSignal(caller.signal, 60000).signal.aborted, true);
}
{
  const a = attemptSignal(undefined, 20);
  await new Promise((r) => setTimeout(r, 60));
  assert.equal(a.signal.aborted, true, "per-attempt timeout still fires");
}
{
  const caller = new AbortController();
  const a = attemptSignal(caller.signal, 60000);
  a.release();
  caller.abort();
  assert.equal(a.signal.aborted, false, "a released attempt no longer follows the caller");
}
assert.equal(retryAllowed(0, 5, undefined), true);
assert.equal(retryAllowed(5, 5, undefined), false);
{
  const caller = new AbortController();
  assert.equal(retryAllowed(0, 5, caller.signal), true);
  caller.abort();
  assert.equal(retryAllowed(0, 5, caller.signal), false);
}
{
  const caller = new AbortController();
  const t0 = Date.now();
  setTimeout(() => caller.abort(), 20);
  await waitUnlessAborted(10000, caller.signal);
  assert.ok(Date.now() - t0 < 2000, "backoff ends when the caller aborts");
}

// -- launch.mjs end to end -------------------------------------------------

const STUBS = {
  "/opt/chatbot/node_modules/undici/index.js":
    "data:text/javascript,export class EnvHttpProxyAgent {} export function setGlobalDispatcher() {}",
  "/opt/chatbot/dist/entry.js": "data:text/javascript,",
};
const hooks = `
const STUBS = ${JSON.stringify(STUBS)};
export async function resolve(specifier, context, next) {
  if (specifier in STUBS) return { url: STUBS[specifier], shortCircuit: true };
  return next(specifier, context);
}`;
register("data:text/javascript," + encodeURIComponent(hooks));

for (const key of ["HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "GAIA2_TRACE_FILE", "GAIA2_MEDIA_ROOT"]) {
  delete process.env[key];
}
const savedArgv = process.argv;
const calls = [];
let upstream = null;
globalThis.fetch = async (url, init) => {
  calls.push(init?.signal);
  return upstream(url, init);
};
await import("./launch.mjs");
process.argv = savedArgv;

const URL_ = "http://127.0.0.1:1/v1/chat/completions";
const body = JSON.stringify({ model: "m", messages: [{ role: "user", content: "hi" }] });
const hang = (_url, init) =>
  new Promise((_resolve, reject) => {
    init.signal.addEventListener("abort", () => reject(init.signal.reason), { once: true });
  });
const ok = () =>
  new Response(JSON.stringify({ choices: [{ index: 0, message: { role: "assistant", content: "done" }, finish_reason: "stop" }] }), {
    status: 200,
    headers: { "content-type": "application/json" },
  });

async function within(promise, ms, what) {
  let timer;
  const guard = new Promise((_r, reject) => {
    timer = setTimeout(() => reject(new Error(`${what}: still pending after ${ms} ms`)), ms);
  });
  try {
    return await Promise.race([promise, guard]);
  } finally {
    clearTimeout(timer);
  }
}

// 1. Caller aborts while the attempt is in flight: the upstream request is
//    aborted with the caller's reason and nothing is retried.
{
  calls.length = 0;
  upstream = hang;
  const caller = new AbortController();
  const pending = globalThis.fetch(URL_, { method: "POST", body, signal: caller.signal });
  setTimeout(() => caller.abort(new Error("openclaw abandoned the turn")), 30);
  await assert.rejects(within(pending, 3000, "aborted in-flight request"), /openclaw abandoned the turn/);
  await new Promise((r) => setTimeout(r, 1500)); // a retry would have started by now
  assert.equal(calls.length, 1, "no retry after the caller aborted");
  assert.equal(calls[0].aborted, true, "the upstream request itself was aborted");
}

// 2. Caller aborts during the retry backoff after a 503: no second attempt.
{
  calls.length = 0;
  upstream = () => new Response("busy", { status: 503 });
  const caller = new AbortController();
  const pending = globalThis.fetch(URL_, { method: "POST", body, signal: caller.signal });
  setTimeout(() => caller.abort(), 200);
  await assert.rejects(within(pending, 3000, "abort during backoff"), (e) => e?.name === "AbortError");
  await new Promise((r) => setTimeout(r, 1500));
  assert.equal(calls.length, 1, "no retry after the caller aborted during backoff");
}

// 3. Without an abort the retry policy is unchanged: a 503 is retried.
{
  calls.length = 0;
  let n = 0;
  upstream = () => (n++ === 0 ? new Response("busy", { status: 503 }) : ok());
  const caller = new AbortController();
  const resp = await within(
    globalThis.fetch(URL_, { method: "POST", body, signal: caller.signal }),
    10000,
    "retried request"
  );
  assert.equal(resp.status, 200);
  assert.match(await resp.text(), /"content":"done"/);
  assert.equal(calls.length, 2);
  assert.notEqual(calls[0], caller.signal, "each attempt gets its own timeout-bearing signal");
}

console.log("test_launch_fetch_abort: ok");
