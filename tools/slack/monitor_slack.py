#!/usr/bin/env python3
"""
lynx HPC Cluster Monitor — Slack Notifier
Cron: 0 0,6,12,18 * * *  (6時間ごと)
"""

import subprocess, json, urllib.request, os, sys, traceback, time
from datetime import datetime
from collections import defaultdict
import os

WEBHOOK_URL = os.environ["SLACK_WEBHOOK_URL"]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(SCRIPT_DIR, "cpp_log")
PARTITION_ORDER = ["40g", "80g", "h100", "gh200", "vis"]
GPU_PARTITIONS = {"40g", "80g", "h100", "gh200"}

os.environ["PATH"] = os.environ.get("PATH", "") + ":/opt/slurm/22.05.2/bin"


# ─── Shell helpers ─────────────────────────────────────────────────────────────


def run(cmd, timeout=30):
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return r.stdout.strip()
    except Exception:
        return ""


# ─── Data collection ───────────────────────────────────────────────────────────


def collect_nodes():
    """Returns dict[partition] = {alloc, idle, mixed, drain, down, total, bad:[]}"""
    out = run("sinfo -h -N -o '%P|%n|%t' 2>/dev/null")
    stats = {}
    for line in out.splitlines():
        cols = line.strip().split("|")
        if len(cols) < 3:
            continue
        part, node, state = cols[0].rstrip("*"), cols[1], cols[2].lower()
        if part not in stats:
            stats[part] = dict(alloc=0, idle=0, mix=0, drain=0, down=0, total=0, bad=[])
        s = stats[part]
        s["total"] += 1
        for key in ("alloc", "idle", "mix", "drain", "down"):
            if key in state:
                s[key] += 1
                if key in ("drain", "down"):
                    s["bad"].append((node, key.upper()))
                break
    return stats


def collect_jobs():
    """Returns (running[], pending[]) — tries squeue --json, falls back to text."""
    running, pending = [], []

    raw = run("squeue --json 2>/dev/null")
    if raw:
        try:
            jobs_raw = json.loads(raw).get("jobs", [])
            for j in jobs_raw:
                states = j.get("job_state", [])
                state = (
                    states[0] if isinstance(states, list) and states else str(states)
                )
                rt = j.get("run_time")
                if isinstance(rt, dict):
                    secs = rt.get("number", 0) or 0
                elif rt is not None:
                    secs = int(rt)
                else:
                    # SLURM 22.05: run_time is null — derive from start_time
                    start = j.get("start_time", 0) or 0
                    secs = max(0, int(time.time()) - start) if start > 0 else 0
                d, rem = divmod(secs, 86400)
                h, rem = divmod(rem, 3600)
                m, s = divmod(rem, 60)
                time_str = (
                    f"{d}-{h:02d}:{m:02d}:{s:02d}"
                    if d
                    else f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
                )
                nc = j.get("node_count", {})
                nodes = str(nc.get("number", 1) if isinstance(nc, dict) else nc)
                reason_raw = j.get("state_reason", "")
                reason = (
                    (reason_raw if isinstance(reason_raw, str) else str(reason_raw))
                    .replace("None", "")
                    .strip()
                )
                job = dict(
                    id=str(j.get("job_id", "")),
                    user=j.get("user_name", "?"),
                    name=(j.get("name", "") or "")[:22],
                    state=state,
                    time=time_str,
                    nodes=nodes,
                    reason=reason,
                )
                if state == "RUNNING":
                    running.append(job)
                elif state == "PENDING":
                    pending.append(job)
            return running, pending
        except (json.JSONDecodeError, KeyError):
            pass

    # テキスト fallback
    out = run("squeue -h -o '%i|%u|%.22j|%T|%M|%D|%R' 2>/dev/null")
    for line in out.splitlines():
        p = line.strip().split("|", 6)
        if len(p) < 6:
            continue
        state = p[3].strip()
        job = dict(
            id=p[0].strip(),
            user=p[1].strip(),
            name=p[2].strip()[:22],
            state=state,
            time=p[4].strip(),
            nodes=p[5].strip(),
            reason=(p[6].strip() if len(p) > 6 else "").replace("None", "").strip(),
        )
        if state == "RUNNING":
            running.append(job)
        elif state == "PENDING":
            pending.append(job)
    return running, pending


def collect_storage():
    out = run("df -h /data3 2>/dev/null | tail -1")
    if not out:
        return None
    p = out.split()
    if len(p) < 5:
        return None
    try:
        return dict(total=p[1], used=p[2], avail=p[3], pct=int(p[4].rstrip("%")))
    except ValueError:
        return None


def collect_gpus():
    """SSH to active GPU nodes (up to 6) and query nvidia-smi."""
    active, seen = [], set()
    out = run("sinfo -h -N -o '%P|%n|%t' 2>/dev/null")
    for line in out.splitlines():
        p = line.strip().split("|")
        if len(p) < 3:
            continue
        part, node, state = p[0].rstrip("*"), p[1], p[2].lower()
        if (
            part in GPU_PARTITIONS
            and node not in seen
            and ("alloc" in state or "mix" in state)
        ):
            active.append(node)
            seen.add(node)

    results = []
    for node in active:
        cmd = (
            f"ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no"
            f" -o UserKnownHostsFile=/dev/null -o BatchMode=yes {node}"
            f" 'nvidia-smi --query-gpu=index,temperature.gpu,memory.used,memory.total,utilization.gpu"
            f" --format=csv,noheader,nounits' 2>/dev/null"
        )
        out = run(cmd, timeout=15)
        for line in out.splitlines():
            cols = [c.strip() for c in line.split(",")]
            if len(cols) >= 5:
                try:
                    results.append(
                        dict(
                            node=node,
                            gpu=int(cols[0]),
                            temp=int(cols[1]),
                            mem_used=int(cols[2]),
                            mem_total=int(cols[3]),
                            util=int(cols[4]),
                        )
                    )
                except ValueError:
                    pass
    return results


# ─── Formatting helpers ────────────────────────────────────────────────────────


def pbar(used, total, width=10):
    n = round(used / total * width) if total > 0 else 0
    return "█" * n + "░" * (width - n)


def disk_icon(pct):
    return "🔴" if pct >= 90 else "🟡" if pct >= 70 else "🟢"


def temp_icon(t):
    return "🔥" if t >= 85 else "🟡" if t >= 75 else "✅"


# ─── Slack payload (Block Kit) ─────────────────────────────────────────────────


def build_payload(nodes, running, pending, storage, gpus):
    now = datetime.now().strftime("%Y-%m-%d %H:%M JST")
    total_bad = sum(s["drain"] + s["down"] for s in nodes.values())
    disk_pct = storage["pct"] if storage else 0
    color = (
        "#E01E5A"
        if (total_bad > 2 or disk_pct >= 90)
        else "#ECB22E" if (total_bad > 0 or disk_pct >= 70) else "#2EB886"
    )
    blocks = []

    # ── Header ──────────────────────────────────────────────────────────────
    blocks += [
        {"type": "header", "text": {"type": "plain_text", "text": "🖥️  lynx Monitor"}},
        {"type": "context", "elements": [{"type": "mrkdwn", "text": f"📅 {now}"}]},
        {"type": "divider"},
    ]

    # ── Node Health ──────────────────────────────────────────────────────────
    lines, bad_lines = [], []
    for p in PARTITION_ORDER:
        if p not in nodes:
            continue
        s = nodes[p]
        healthy = s["alloc"] + s["idle"] + s["mix"]
        flag = "⚠️" if s["bad"] else "✅"
        lines.append(
            f"`{p:<6}` │ `{pbar(healthy, s['total'])}` │ {healthy}/{s['total']}  {flag}"
        )
        for node, reason in s["bad"]:
            icon = "🔴" if reason == "DOWN" else "⚠️"
            bad_lines.append(f"  {icon} `{node}` — {reason}")
    body = "\n".join(lines) or "_データ取得失敗_"
    if bad_lines:
        body += "\n\n*要注意ノード:*\n" + "\n".join(bad_lines)
    blocks.append(
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*📊 Node Health*\n{body}"},
        }
    )
    blocks.append({"type": "divider"})

    # ── Job Queue ────────────────────────────────────────────────────────────
    jlines = [f"*▶ 実行中: {len(running)} 件*   *⏳ 待機中: {len(pending)} 件*"]
    if running:
        jlines += ["", "*実行中ジョブ*"]
        for j in running[:12]:
            jlines.append(
                f"  `{j['id']:>7}` `{j['user']:<10}` `{j['name']:<22}` ⏱ `{j['time']}` ({j['nodes']} nodes)"
            )
        if len(running) > 12:
            jlines.append(f"  _...他 {len(running)-12} 件_")
    if pending:
        jlines += ["", "*待機中ジョブ*"]
        for j in pending[:8]:
            reason = j["reason"] or "Resources"
            jlines.append(
                f"  `{j['id']:>7}` `{j['user']:<10}` `{j['name']:<22}` _{reason}_"
            )
        if len(pending) > 8:
            jlines.append(f"  _...他 {len(pending)-8} 件_")
    if not running and not pending:
        jlines.append("\n_実行中・待機中のジョブなし_")
    blocks.append(
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": "*⚡ Job Queue*\n" + "\n".join(jlines)},
        }
    )
    blocks.append({"type": "divider"})

    # ── Storage ──────────────────────────────────────────────────────────────
    if storage:
        st = (
            f"`{pbar(storage['pct'], 100, 12)}` {storage['pct']}%  {disk_icon(storage['pct'])}\n"
            f"使用量: *{storage['used']}* / {storage['total']}　　空き: *{storage['avail']}*"
        )
    else:
        st = "_取得失敗_"
    blocks.append(
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*💾 Storage: /data3*\n{st}"},
        }
    )

    # ── GPU Status ───────────────────────────────────────────────────────────
    if gpus:
        blocks.append({"type": "divider"})
        glines, prev = [], None
        for g in gpus:
            if g["node"] != prev:
                glines.append(f"\n*{g['node']}*")
                prev = g["node"]
            mp = round(g["mem_used"] / g["mem_total"] * 100)
            glines.append(
                f"  GPU{g['gpu']}  {temp_icon(g['temp'])} `{g['temp']}°C`  "
                f"`{pbar(g['mem_used'], g['mem_total'], 8)}` {mp}%  util `{g['util']}%`"
            )
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*🔥 GPU Status (アクティブノード)*" + "\n".join(glines),
                },
            }
        )

    blocks.append({"type": "divider"})
    return {"attachments": [{"color": color, "blocks": blocks}]}


# ─── Slack POST ────────────────────────────────────────────────────────────────


def post_slack(payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        WEBHOOK_URL, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.status


def post_error(msg):
    payload = {"text": f"🚨 *lynx monitor エラー*\n```{msg[:2000]}```"}
    try:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            WEBHOOK_URL, data=data, headers={"Content-Type": "application/json"}
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass


# ─── Entry point ───────────────────────────────────────────────────────────────


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    log = lambda m: print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)

    try:
        log("node stats 収集中...")
        nodes = collect_nodes()
        log(f"  partitions={list(nodes.keys())}")

        log("job queue 収集中...")
        running, pending = collect_jobs()
        log(f"  running={len(running)}, pending={len(pending)}")

        log("storage 確認中...")
        storage = collect_storage()
        log(f"  storage={storage}")

        log("GPU status 確認中 (SSH)...")
        gpus = collect_gpus()
        log(f"  gpu_entries={len(gpus)}")

        log("Slack に送信中...")
        payload = build_payload(nodes, running, pending, storage, gpus)
        status = post_slack(payload)
        log(f"  完了 (HTTP {status}) ✓")

    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        post_error(tb)
        sys.exit(1)


if __name__ == "__main__":
    main()
