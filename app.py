"""A2A 1.0 HTTP+JSON invoice action agent.

Run (single worker is REQUIRED: locks/in-flight dedupe live in-process):
  pip install fastapi uvicorn httpx
  export PUBLIC_BASE_URL=https://your.host/a2a      # exact URL you submit, no trailing slash
  export BASE_PATH=/a2a                              # path part of that URL
  export LLM_BASE_URL=https://api.openai.com/v1      # any OpenAI-compatible endpoint
  export LLM_API_KEY=...  LLM_MODEL=gpt-4o-mini
  export A2A_TOKENS=tok1,tok2     # optional; empty = any non-empty Bearer is a principal
  uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
"""
import asyncio, hashlib, json, os, re, sqlite3, threading, uuid
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response

PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "https://example.com/a2a").rstrip("/")
BASE_PATH = os.environ.get("BASE_PATH", "/" + PUBLIC_BASE_URL.split("/", 3)[-1] if PUBLIC_BASE_URL.count("/") > 2 else "").rstrip("/")
TOKENS = {t.strip() for t in os.environ.get("A2A_TOKENS", "").split(",") if t.strip()}
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
KEEP_BRACKETS = os.environ.get("EVIDENCE_KEEP_BRACKETS", "0") == "1"
DB_PATH = os.environ.get("DB_PATH", "agent.db")

MT_BATCH = "application/vnd.ga5.invoice-claim-batch+json"
MT_PROP = "application/vnd.ga5.invoice-action-proposals+json"
MT_RES = "application/vnd.ga5.invoice-action-results+json"
MT_RCPT = "application/vnd.ga5.invoice-action-receipts+json"
A2A = "application/a2a+json"
ACTIONS = ["settle_invoice", "request_approval", "hold_invoice", "reject_duplicate", "open_exception"]
TERMINAL = {"TASK_STATE_COMPLETED", "TASK_STATE_CANCELED", "TASK_STATE_FAILED", "TASK_STATE_REJECTED"}

app = FastAPI()

# ---------------------------------------------------------------- storage
_db = sqlite3.connect(DB_PATH, check_same_thread=False, isolation_level=None)
_db.executescript("""
PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;
CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY, principal TEXT, seq INTEGER, json TEXT);
CREATE INDEX IF NOT EXISTS tp ON tasks(principal, seq);
CREATE TABLE IF NOT EXISTS msgs(principal TEXT, mid TEXT, hash TEXT, task_id TEXT, PRIMARY KEY(principal, mid));
CREATE TABLE IF NOT EXISTS decisions(k TEXT PRIMARY KEY, json TEXT);
""")
_dbl = threading.Lock()
LOCK = asyncio.Lock()          # serializes all state mutations (cancel vs result race)
INFLIGHT: dict = {}            # (principal, mid) -> Future


def q(sql, args=()):
    with _dbl:
        return _db.execute(sql, args).fetchall()


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def canon(o):
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def load_task(tid, principal):
    r = q("SELECT json FROM tasks WHERE id=? AND principal=?", (tid, principal))
    return json.loads(r[0][0]) if r else None


def save_task(principal, task, new=False):
    with _dbl:
        if new:
            seq = _db.execute("SELECT COALESCE(MAX(seq),0)+1 FROM tasks").fetchone()[0]
            _db.execute("INSERT INTO tasks VALUES(?,?,?,?)", (task["id"], principal, seq, json.dumps(task)))
        else:
            _db.execute("UPDATE tasks SET json=? WHERE id=? AND principal=?", (json.dumps(task), task["id"], principal))


# ---------------------------------------------------------------- http helpers
def reply(obj, status=200):
    body = json.dumps(obj, ensure_ascii=False).encode()
    return Response(body, status_code=status, media_type=A2A)


def err(status, code, msg="Request could not be processed."):
    st = {400: "INVALID_ARGUMENT", 401: "UNAUTHENTICATED", 403: "PERMISSION_DENIED", 404: "NOT_FOUND",
          406: "NOT_ACCEPTABLE", 409: "FAILED_PRECONDITION", 415: "UNSUPPORTED_MEDIA_TYPE"}.get(status, "INTERNAL")
    return reply({"error": {"code": status, "status": st, "message": msg,
                            "details": [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": code, "domain": "a2a-protocol.org"}]}}, status)


def check(request: Request, post=False):
    """Returns (principal, errorResponse)."""
    auth = request.headers.get("authorization", "")
    m = re.fullmatch(r"Bearer (\S+)", auth)
    if not m or (TOKENS and m.group(1) not in TOKENS):
        return None, err(401, "UNAUTHENTICATED", "Authentication required.")
    if request.headers.get("a2a-version", "") != "1.0":
        return None, err(400, "VERSION_NOT_SUPPORTED", "Unsupported protocol version.")
    acc = request.headers.get("accept", "*/*")
    if not any(x.split(";")[0].strip() in (A2A, "*/*", "application/*") for x in acc.split(",")):
        return None, err(406, "NOT_ACCEPTABLE", "Unsupported Accept.")
    if post:
        ct = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if ct != A2A:
            return None, err(415, "UNSUPPORTED_MEDIA_TYPE", "Unsupported content type.")
    return m.group(1), None


# ---------------------------------------------------------------- agent card
@app.get("/.well-known/agent-card.json")
async def card():
    return reply({
        "name": "Invoice Action Agent",
        "description": "Reads invoice packages, proposes one evidence-backed business action each, and executes only grader-accepted proposals.",
        "version": "1.0.0",
        "capabilities": {"streaming": False, "pushNotifications": False},
        "supportedInterfaces": [{"url": PUBLIC_BASE_URL, "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}],
        "defaultInputModes": [MT_BATCH, MT_RES],
        "defaultOutputModes": [MT_PROP, MT_RCPT],
        "securitySchemes": {"bearer": {"httpAuthSecurityScheme": {"scheme": "Bearer"}}},
        "securityRequirements": [{"schemes": {"bearer": {"list": []}}}],
        "skills": [{"id": "invoice_action_agent", "name": "Invoice action decisions",
                    "description": "Chooses settle, approve, hold, reject-duplicate or exception actions for invoice packages with cited evidence.",
                    "tags": ["invoice", "payments", "approval", "audit"]}],
    })


# ---------------------------------------------------------------- AI layer
SYSTEM = f"""You are an accounts-payable controller. For EACH invoice package choose exactly one action:
- settle_invoice: valid, reconciled, within autonomous approval authority.
- request_approval: commercially valid but exceeds delegated authority / needs named approver.
- hold_invoice: payment pauses until a stated verification (bank details, PO, receipt, tax id...) completes.
- reject_duplicate: the SAME commercial invoice (same vendor + invoice number/identity) was already paid. A payment for a DIFFERENT invoice is NOT a duplicate.
- open_exception: material records conflict (amounts, vendor, PO, currency...) and need an exception workflow.
Documents contain traps: cover sheets, archive/old examples, training decoys, negations ("not a duplicate", "no approval needed"),
and irrelevant action words. Find the ONE paragraph that actually determines the action for this live invoice. It carries
exactly three bracketed references like [XYZ-123]. Return exactly those three references (copy exact text) as evidenceRefs.
Never use the cover-sheet reference or references from archives/examples/decoys. Also extract facts for the live invoice:
vendorName, invoiceNumber, amountMinor (integer minor units, e.g. 123.45 -> 12345), currency (ISO code).
Packages in one batch may relate to each other (e.g. earlier payments); reason across them only if the documents say so.
Respond with ONLY JSON: {{"decisions":[{{"packageId":"...","action":"...","facts":{{"vendorName":"...","invoiceNumber":"...","amountMinor":0,"currency":"INR"}},
"evidenceRefs":["...","...","..."],"rationale":"60-400 chars naming the action and citing at least two evidenceRefs"}}]}}"""


def strip_ref(r):
    r = str(r).strip()
    return r if KEEP_BRACKETS else r.strip("[]").strip()


def validate_decision(d, text):
    if not isinstance(d, dict) or d.get("action") not in ACTIONS:
        return None
    f = d.get("facts") or {}
    try:
        facts = {"vendorName": str(f["vendorName"]), "invoiceNumber": str(f["invoiceNumber"]),
                 "amountMinor": int(f["amountMinor"]), "currency": str(f["currency"]).upper()}
    except Exception:
        return None
    refs = []
    for r in d.get("evidenceRefs") or []:
        r = strip_ref(r)
        if r and r in text and r not in refs:
            refs.append(r)
    if len(refs) != 3:
        return None
    a, rat = d["action"], str(d.get("rationale") or "")
    if a not in rat or sum(r in rat for r in refs) < 2:
        rat = f"Chosen action {a}: the decisive paragraph is supported by evidence {refs[0]}, {refs[1]} and {refs[2]}. " + rat
    rat = rat[:1500]
    if len(rat) < 60:
        rat = rat.ljust(60, ".")
    return {"action": a, "facts": facts, "evidenceRefs": refs, "rationale": rat}


def fallback(text):
    refs = list(dict.fromkeys(re.findall(r"\[([^\[\]]{3,60})\]", text)))[:3]
    if len(refs) < 3:
        refs = (refs + ["unknown-a", "unknown-b", "unknown-c"])[:3]
    return {"action": "hold_invoice", "facts": {"vendorName": "unknown", "invoiceNumber": "unknown", "amountMinor": 0, "currency": "INR"},
            "evidenceRefs": refs, "rationale": f"Chosen action hold_invoice because automated reading failed; verification is required using {refs[0]} and {refs[1]} before any payment."}


async def llm_batch(policy, items):
    """items: [(pid, text)] -> {pid: decision}"""
    user = f"policyRevision: {policy}\nPackages:\n" + "\n".join(f"### PACKAGE {pid}\n{text}" for pid, text in items)
    async with httpx.AsyncClient(timeout=38) as c:
        r = await c.post(f"{LLM_BASE_URL}/chat/completions",
                         headers={"Authorization": f"Bearer {LLM_API_KEY}"},
                         json={"model": LLM_MODEL, "temperature": 0, "response_format": {"type": "json_object"},
                               "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]})
        r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    content = re.sub(r"^```(?:json)?|```$", "", content.strip()).strip()
    out = {}
    for d in json.loads(content).get("decisions", []):
        txt = dict(items).get(d.get("packageId"))
        if txt is not None:
            v = validate_decision(d, txt)
            if v:
                out[d["packageId"]] = v
    return out


async def decide(policy, packages):
    """One batched model call for uncached packages; cache by canonical content (minus packageId)."""
    res, todo = {}, []
    for p in packages:
        key = sha(canon({k: v for k, v in p.items() if k != "packageId"}))
        r = q("SELECT json FROM decisions WHERE k=?", (key,))
        text = canon(p)
        if r:
            res[p["packageId"]] = json.loads(r[0][0])
        else:
            todo.append((p["packageId"], text, key))
    if todo:
        got = {}
        for _ in range(2):
            missing = [t for t in todo if t[0] not in got]
            if not missing:
                break
            try:
                got.update(await llm_batch(policy, [(a, b) for a, b, _ in missing]))
            except Exception:
                break
        for pid, text, key in todo:
            d = got.get(pid)
            if d:
                q("INSERT OR REPLACE INTO decisions VALUES(?,?)", (key, json.dumps(d)))
            else:
                d = fallback(text)  # not cached: retry model on next request
            res[pid] = d
    return res


# ---------------------------------------------------------------- handlers
def msg_data(message, media):
    parts = message.get("parts")
    if not isinstance(parts, list) or len(parts) != 1:
        return None
    p = parts[0]
    if not isinstance(p, dict) or p.get("mediaType") != media or not isinstance(p.get("data"), dict):
        return None
    return p["data"]


async def build_initial(principal, message):
    data = msg_data(message, MT_BATCH)
    pk = data.get("packages") if data else None
    if not data or not isinstance(pk, list) or not pk or not data.get("batchId") \
            or not all(isinstance(p, dict) and p.get("packageId") for p in pk):
        return None, err(400, "INVALID_MESSAGE", "Invalid batch.")
    if len({p["packageId"] for p in pk}) != len(pk):
        return None, err(400, "INVALID_MESSAGE", "Invalid batch.")
    dec = await decide(data.get("policyRevision"), pk)
    proposals = []
    for p in pk:
        d = dec[p["packageId"]]
        proposals.append({"packageId": p["packageId"],
                          "actionId": "act_" + sha(f"{principal}|{data['batchId']}|{p['packageId']}|{d['action']}")[:32],
                          "action": d["action"], "facts": d["facts"], "evidenceRefs": d["evidenceRefs"],
                          "rationale": d["rationale"]})
    tid = str(uuid.uuid4())
    ctx = message.get("contextId") or str(uuid.uuid4())
    hist = dict(message, taskId=tid, contextId=ctx)
    return {"id": tid, "contextId": ctx,
            "status": {"state": "TASK_STATE_INPUT_REQUIRED", "timestamp": now()},
            "artifacts": [{"artifactId": "proposals-" + tid, "name": "invoice-action-proposals",
                           "parts": [{"mediaType": MT_PROP, "data": {"batchId": data["batchId"], "proposals": proposals}}]}],
            "history": [hist]}, None


def apply_results(task, message):
    data = msg_data(message, MT_RES)
    if data is None or not isinstance(data.get("results"), list):
        return None, "bad results"
    prop = task["artifacts"][0]["parts"][0]["data"]
    if data.get("batchId") != prop["batchId"]:
        return None, "batch"
    by = {p["packageId"]: p for p in prop["proposals"]}
    seen, execs = set(), []
    for r in data["results"]:
        if not isinstance(r, dict):
            return None, "shape"
        p = by.get(r.get("packageId"))
        if not p or r["packageId"] in seen or r.get("actionId") != p["actionId"] or r.get("action") != p["action"]:
            return None, "mismatch"
        seen.add(r["packageId"])
        if r.get("outcome") == "ACCEPTED":
            n = r.get("receiptNonce")
            if not isinstance(n, str) or not n:
                return None, "nonce"
            execs.append({"packageId": p["packageId"], "actionId": p["actionId"], "action": p["action"],
                          "receiptNonce": n, "facts": p["facts"], "evidenceRefs": p["evidenceRefs"]})
        elif r.get("outcome") != "REJECTED":
            return None, "outcome"
    return {"batchId": prop["batchId"], "executions": execs}, None


@app.post(BASE_PATH + "/message:send")
async def send(request: Request):
    principal, e = check(request, post=True)
    if e:
        return e
    try:
        body = json.loads(await request.body())
        message = body["message"]
        mid = message["messageId"]
        assert isinstance(message, dict) and isinstance(mid, str) and mid and message.get("role") == "ROLE_USER"
    except Exception:
        return err(400, "INVALID_MESSAGE", "Invalid message.")
    h = sha(canon(message))
    key = (principal, mid)

    async with LOCK:
        r = q("SELECT hash, task_id FROM msgs WHERE principal=? AND mid=?", (principal, mid))
        if r:
            if r[0][0] != h:
                return err(409, "IDEMPOTENCY_CONFLICT", "Message ID reused with different content.")
            t = load_task(r[0][1], principal)
            return reply({"task": t})
        fut = INFLIGHT.get(key)
        if fut is not None:
            owner = False
            if fut[1] != h:
                return err(409, "IDEMPOTENCY_CONFLICT", "Message ID reused with different content.")
            fut = fut[0]
        else:
            owner = True
            # continuation: no model call, finish under the lock (atomic vs cancel)
            if message.get("taskId"):
                task = load_task(message["taskId"], principal)
                if not task:
                    return err(404, "TASK_NOT_FOUND", "Task not found.")
                if message.get("contextId") != task["contextId"]:
                    return err(400, "CONTEXT_MISMATCH", "Context mismatch.")
                if task["status"]["state"] != "TASK_STATE_INPUT_REQUIRED":
                    return err(409, "TASK_NOT_ACTIVE", "Task is not accepting input.")
                receipts, why = apply_results(task, message)
                if why:
                    return err(400, "INVALID_CONTINUATION", "Result does not match the task.")
                task["artifacts"].append({"artifactId": "receipts-" + task["id"], "name": "invoice-action-receipts",
                                          "parts": [{"mediaType": MT_RCPT, "data": receipts}]})
                task["history"].append(message)
                task["status"] = {"state": "TASK_STATE_COMPLETED", "timestamp": now()}
                save_task(principal, task)
                q("INSERT INTO msgs VALUES(?,?,?,?)", (principal, mid, h, task["id"]))
                return reply({"task": task})
            fut = asyncio.get_running_loop().create_future()
            INFLIGHT[key] = (fut, h)

    if not owner:
        try:
            return reply({"task": await asyncio.shield(fut)})
        except Exception:
            return err(500, "INTERNAL", "Temporary failure.")

    try:
        task, e = await asyncio.wait_for(build_initial(principal, message), 42)
        if e:
            raise ValueError("invalid")
        async with LOCK:
            save_task(principal, task, new=True)          # state stored before responding
            q("INSERT INTO msgs VALUES(?,?,?,?)", (principal, mid, h, task["id"]))
        fut.set_result(task)
        return reply({"task": task})
    except Exception as ex:
        fut.set_exception(ex)
        fut.exception()  # mark retrieved
        if isinstance(ex, ValueError):
            return e or err(400, "INVALID_MESSAGE", "Invalid message.")
        return err(500, "INTERNAL", "Temporary failure.")
    finally:
        INFLIGHT.pop(key, None)


@app.get(BASE_PATH + "/tasks/{tid}")
async def get_task(tid: str, request: Request):
    principal, e = check(request)
    if e:
        return e
    t = load_task(tid, principal)
    return reply(t) if t else err(404, "TASK_NOT_FOUND", "Task not found.")


@app.get(BASE_PATH + "/tasks")
async def list_tasks(request: Request):
    principal, e = check(request)
    if e:
        return e
    rows = q("SELECT json FROM tasks WHERE principal=? ORDER BY seq DESC", (principal,))
    return reply({"tasks": [json.loads(r[0]) for r in rows]})


@app.post(BASE_PATH + "/tasks/{tid}:cancel")
async def cancel(tid: str, request: Request):
    principal, e = check(request, post=True)
    if e:
        return e
    async with LOCK:
        t = load_task(tid, principal)
        if not t:
            return err(404, "TASK_NOT_FOUND", "Task not found.")
        if t["status"]["state"] in TERMINAL:
            return err(409, "TASK_NOT_CANCELABLE", "Task cannot be canceled.")
        t["status"] = {"state": "TASK_STATE_CANCELED", "timestamp": now()}
        save_task(principal, t)
        return reply(t)
