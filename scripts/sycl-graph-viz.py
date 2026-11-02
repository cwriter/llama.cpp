#!/usr/bin/env python3
"""Data-dependency view of a graph the SYCL backend executed, with its fusions and fusion candidates.

Dump the graphs first (one file per device split and graph, see ggml/src/ggml-sycl/graph-dump.hpp):

    GGML_SYCL_GRAPH_DUMP=/path/dir GGML_SYCL_GRAPH_DUMP_TOKENS=1 llama-cli ... -n 4

then

    scripts/sycl-graph-viz.py /path/dir/graph-dev0-2-t1.json -o out          # out.html + out.txt
    scripts/sycl-graph-viz.py graph.json -o out --layers 12,13               # two layers and their neighbours

A box is one kernel launch as the backend runs it: the node that submitted work plus the nodes its fused
kernel computed (skipped over, or absorbed and computed elsewhere). Views are folded into the edges.
Edges are read-after-write dependencies through tensor storage, so an in-place write (SET_ROWS, CPY into a
view) orders the readers of that storage after it.

Candidates, counted per launch saved:
  vertical    A -> B where B is the only launch reading A (A's output dies in B): an epilogue/prologue fusion
  horizontal  launches reading the same activation with no path between them: one kernel can do both
              (the shared-expert GLU and the router, the q/k/v projections, ...)
The text report aggregates them over layers (names with the layer suffix stripped), largest count first.
Levels are longest-path depths in the launch DAG: launches on the same level do not depend on each other.
"""

import argparse
import collections
import html
import json
import re
import sys

VIEW_OPS = {"RESHAPE", "VIEW", "PERMUTE", "TRANSPOSE", "NONE"}
ELEMENTWISE = {"ADD", "ADD1", "SUB", "MUL", "DIV", "SCALE", "SQR", "SQRT", "LOG", "SIN", "COS", "CLAMP",
               "CPY", "CONT", "DUP", "ADD_ID", "FILL"}

LAYER_RE = re.compile(r"-(\d+)$")


def op_class(op):
    if op.startswith("MUL_MAT") or op in ("OUT_PROD",):
        return "matmul"
    if op in ("FLASH_ATTN_EXT", "SOFT_MAX", "ROPE", "LIGHTNING_INDEXER") or "ATTN" in op:
        return "attn"
    if op in ("RMS_NORM", "NORM", "L2_NORM", "GROUP_NORM"):
        return "norm"
    if op in ("ARGSORT", "TOP_K", "ARGMAX", "GET_ROWS", "SET_ROWS", "GET_ROWS_BACK"):
        return "index"
    if op in ("GATED_DELTA_NET", "SSM_CONV", "SSM_SCAN", "RWKV_WKV6", "RWKV_WKV7"):
        return "recurrent"
    if op in ("SUM_ROWS", "SUM", "MEAN", "CUMSUM", "CONCAT", "REPEAT"):
        return "reduce"
    return "elementwise"


COLORS = {"matmul": "#4878d0", "attn": "#d65f5f", "norm": "#6acc64", "index": "#ee854a",
          "recurrent": "#956cb4", "reduce": "#8c613c", "elementwise": "#82868c"}


def short_kernel(name):
    # a SYCL kernel is named by its lambda or functor type: keep the enclosing function or the functor
    n = name.replace("(anonymous namespace)::", "")
    if "::{lambda" in n:
        n = n[:n.index("::{lambda")]
    depth, out = 0, []
    for ch in n:
        if ch in "<(":
            depth += 1
        elif ch in ">)":
            depth -= 1
        elif depth == 0:
            out.append(ch)
    n = "".join(out).strip()
    return n.split(" ")[-1] if n else name[:60]


def strip_layer(name):
    # layer suffixes, cache_r_l16, node_2398: any number, so the same step of every layer aggregates
    return re.sub(r"\d+", "N", name)


def layer_of(name):
    m = LAYER_RE.search(name)
    return int(m.group(1)) if m else None


class Graph:
    def __init__(self, path):
        with open(path) as f:
            d = json.load(f)
        self.meta = {k: d[k] for k in ("device", "seq", "tokens", "n_nodes")}
        self.traced = d.get("traced", False)
        self.nodes = d["nodes"]
        self.ext = d["ext"]
        self.build()

    # storage a tensor reference lives in: ("n", i) or ("e", j)
    def storage(self, ref):
        t = self.nodes[ref["node"]] if "node" in ref else self.ext[ref["ext"]]
        vs = t["view_src"]
        if vs is not None:
            return ("n", vs["node"]) if "node" in vs else ("e", vs["ext"])
        return ("n", ref["node"]) if "node" in ref else ("e", ref["ext"])

    def build(self):
        nodes = self.nodes
        n = len(nodes)

        # kind of each node and the launch that computed it
        kind = [None] * n
        group = [None] * n
        for t in nodes:
            i = t["i"]
            if t["op"] in VIEW_OPS:
                kind[i] = "view"
            elif t["jumped_by"] >= 0 and nodes[t["jumped_by"]]["submitted"]:
                kind[i], group[i] = "fused", t["jumped_by"]
            elif t["submitted"] and t["jumped_by"] < 0:
                kind[i], group[i] = "launch", i
            else:
                kind[i] = "absorbed"
        self.kind = kind

        # read-after-write deps through storage, in execution order
        last_writer = {}
        readers = collections.defaultdict(list)  # storage -> readers since its last write
        deps = [[] for _ in range(n)]            # (producer node, tensor ref, storage)
        war = [[] for _ in range(n)]             # nodes that must stay before this one
        for t in nodes:
            i = t["i"]
            for ref in t["src"]:
                if ref is None:
                    continue
                s = self.storage(ref)
                w = last_writer.get(s)
                if w is not None:
                    deps[i].append((w, ref))
                if kind[i] != "view":
                    readers[s].append(i)
            if kind[i] != "view":
                s = ("n", i) if t["view_src"] is None else self.storage({"node": i})
                if s != ("n", i):
                    war[i].extend(r for r in readers[s] if r != i)
                last_writer[s] = i
                readers[s] = []
        # a view's deps belong to whoever reads the view
        for i in range(n):
            if kind[i] == "view":
                continue
            expanded, seen = [], set()
            stack = list(deps[i])
            while stack:
                w, ref = stack.pop()
                if kind[w] == "view":
                    stack.extend(deps[w])
                elif w not in seen:
                    seen.add(w)
                    expanded.append((w, ref))
            deps[i] = expanded
        self.deps = deps

        consumers = [[] for _ in range(n)]
        for i in range(n):
            for w, _ in deps[i]:
                consumers[w].append(i)

        # absorbed nodes belong to the first launch downstream that reads them, else the one upstream
        def attribute(i):
            seen, queue = {i}, collections.deque(consumers[i])
            while queue:
                c = queue.popleft()
                if c in seen:
                    continue
                seen.add(c)
                if group[c] is not None and kind[c] != "absorbed":
                    return group[c]
                queue.extend(consumers[c])
            seen, queue = {i}, collections.deque(w for w, _ in deps[i])
            while queue:
                p = queue.popleft()
                if p in seen:
                    continue
                seen.add(p)
                if group[p] is not None and kind[p] != "absorbed":
                    return group[p]
                queue.extend(w for w, _ in deps[p])
            return None

        for i in range(n):
            if kind[i] == "absorbed":
                group[i] = attribute(i)
        # absorbed with no launch near it at all (no-ops, inputs computed nowhere): its own pseudo launch
        for i in range(n):
            if kind[i] == "absorbed" and group[i] is None:
                group[i] = i
        self.group = group

        # launch groups in execution order
        members = collections.defaultdict(list)
        for i in range(n):
            if group[i] is not None:
                members[group[i]].append(i)
        self.groups = sorted(members)
        self.members = members
        gid = {g: k for k, g in enumerate(self.groups)}
        self.gid = gid

        G = len(self.groups)
        edges = {}  # (a, b) -> list of (tensor name, bytes)
        order = set()
        for i in range(n):
            if group[i] is None:
                continue
            b = gid[group[i]]
            for w, ref in deps[i]:
                if group[w] is None:
                    continue
                a = gid[group[w]]
                if a != b:
                    t = nodes[w]
                    edges.setdefault((a, b), [])
                    if (t["name"], t["bytes"]) not in edges[(a, b)]:
                        edges[(a, b)].append((t["name"], t["bytes"]))
            for r in war[i]:
                if group[r] is not None and gid[group[r]] != b:
                    order.add((gid[group[r]], b))
        self.edges = edges
        self.order = order - set(edges)

        self.succ = [[] for _ in range(G)]
        self.pred = [[] for _ in range(G)]
        for a, b in list(edges) + list(self.order):
            if a < b:  # execution order is a topological order
                self.succ[a].append(b)
                self.pred[b].append(a)

        # ASAP level, ALAP level
        level = [0] * G
        for b in range(G):
            for a in self.pred[b]:
                level[b] = max(level[b], level[a] + 1)
        depth = max(level) if G else 0
        alap = [depth] * G
        for a in reversed(range(G)):
            for b in self.succ[a]:
                alap[a] = min(alap[a], alap[b] - 1)
        self.level, self.alap, self.depth = level, alap, depth

        # reachability bitsets for independence tests
        reach = [0] * G
        for a in reversed(range(G)):
            r = 0
            for b in self.succ[a]:
                r |= (1 << b) | reach[b]
            reach[a] = r
        self.reach = reach

    def independent(self, a, b):
        return not (self.reach[a] >> b) & 1 and not (self.reach[b] >> a) & 1

    def ginfo(self, k):
        g = self.groups[k]
        lead = self.nodes[g]
        mem = [self.nodes[i] for i in self.members[g]]
        return lead, mem

    def label(self, k):
        lead, mem = self.ginfo(k)
        return lead["name"], lead["op"]

    def kernels(self, k):
        return [w for m in self.ginfo(k)[1] for w in m.get("work", [])]

    def gsig(self, k):
        lead, mem = self.ginfo(k)
        return "%s[%s]" % (strip_layer(lead["name"]), "+".join(m["op"] for m in mem))

    def candidates(self):
        nodes = self.nodes
        G = len(self.groups)
        vertical = []
        for a in range(G):
            outs = [b for b in self.succ[a] if (a, b) in self.edges]
            if len(outs) != 1:
                continue
            b = outs[0]
            # the output must not leave the graph (another split, the logits)
            lead, mem = self.ginfo(a)
            if any(m["flags"] & 2 for m in mem):
                continue
            vertical.append((a, b))
        horizontal = []
        by_input = collections.defaultdict(set)
        for k in range(G):
            lead, mem = self.ginfo(k)
            for m in mem:
                for ref in m["src"]:
                    if ref is None:
                        continue
                    s = self.storage(ref)
                    t = nodes[s[1]] if s[0] == "n" else self.ext[s[1]]
                    if t["usage"] == "weights":
                        continue
                    by_input[s].add(k)
        for s, readers in by_input.items():
            readers = sorted(readers)
            # greedy: clusters of pairwise independent readers, in execution order
            clusters = []
            for r in readers:
                for c in clusters:
                    if all(self.independent(r, x) for x in c):
                        c.append(r)
                        break
                else:
                    clusters.append([r])
            for c in clusters:
                if len(c) >= 2:
                    t = nodes[s[1]] if s[0] == "n" else self.ext[s[1]]
                    horizontal.append((t["name"], t["bytes"], c))
        return vertical, horizontal


def report(g, vertical, horizontal, out):
    nodes = g.nodes
    kinds = collections.Counter(g.kind)
    G = len(g.groups)
    p = lambda *a: print(*a, file=out)
    m = g.meta
    p("graph: device %d, seq %d, %d tokens, %d nodes" % (m["device"], m["seq"], m["tokens"], m["n_nodes"]))
    work = [w for k in range(G) for w in g.kernels(k)]
    p("launches %d (submitting nodes %d), fused into them %d, absorbed %d, views %d" % (
        G, kinds["launch"], kinds["fused"], kinds["absorbed"], kinds["view"]))
    if g.traced:
        p("submitted: %d kernels, %d copies, %d fills" % (
            sum(1 for w in work if w not in ("copy", "fill")), work.count("copy"), work.count("fill")))
    else:
        p("NOTE: dump without Level Zero tracing: every visited node counts as a launch, absorbed nodes are not known")
    width = collections.Counter(g.level)
    p("dependency depth %d levels for %d launches: mean width %.2f, max %d" % (
        g.depth + 1, G, G / max(1, g.depth + 1), max(width.values()) if width else 0))
    hist = collections.Counter(width.values())
    p("levels by width: " + ", ".join("%dx%d" % (w, hist[w]) for w in sorted(hist)))

    p("\nlaunches by op (leader op [fused ops]):")
    ops = collections.Counter("+".join(mm["op"] for mm in g.ginfo(k)[1]) for k in range(G))
    for o, c in ops.most_common(40):
        p("  %5d  %s" % (c, o))

    if g.traced:
        p("\nkernels by name:")
        for o, c in collections.Counter(short_kernel(w) for w in work).most_common(40):
            p("  %5d  %s" % (c, o))

    p("\nfusions the backend did (nodes per launch > 1), by pattern:")
    fused = collections.Counter(g.gsig(k) for k in range(G) if len(g.ginfo(k)[1]) > 1)
    for o, c in fused.most_common(60):
        p("  %5d  %s" % (c, o))

    p("\nvertical candidates (B is the only launch reading A), by pattern:")
    vc = collections.Counter()
    vbytes = {}
    for a, b in vertical:
        key = "%s  ->  %s" % (g.gsig(a), g.gsig(b))
        vc[key] += 1
        vbytes[key] = sum(x[1] for x in g.edges[(a, b)])
    for o, c in vc.most_common(60):
        p("  %5d  %s   (%s B)" % (c, o, vbytes[o]))

    p("\nhorizontal candidates (independent launches reading the same activation), by pattern:")
    hc = collections.Counter()
    for name, nbytes, c in horizontal:
        key = "%s: %s" % (strip_layer(name), " | ".join(g.gsig(k) for k in c))
        hc[key] += 1
    for o, c in hc.most_common(60):
        p("  %5d  %s" % (c, o))


def layout_levels(g, keep):
    # y = ASAP level, x = barycentre ordering within the level
    rows = collections.defaultdict(list)
    for k in keep:
        rows[g.level[k]].append(k)
    pos = {}
    for lv in sorted(rows):
        for x, k in enumerate(rows[lv]):
            pos[k] = x
    for _ in range(6):
        for lv in sorted(rows):
            def bc(k):
                ps = [pos[a] for a in g.pred[k] if a in pos]
                return sum(ps) / len(ps) if ps else pos[k]
            rows[lv].sort(key=bc)
            for x, k in enumerate(rows[lv]):
                pos[k] = x
    levels = sorted(rows)
    yidx = {lv: y for y, lv in enumerate(levels)}
    return {k: (pos[k], yidx[g.level[k]]) for k in keep}


def layout_order(g, keep):
    # y = launch order, x = lane: a launch continues the lane of a producer that has no other reader left
    lanes = []
    remaining = {k: len([b for b in g.succ[k] if b in keep]) for k in keep}
    out = {}
    for y, k in enumerate(sorted(keep)):
        preds = [a for a in g.pred[k] if a in keep]
        for a in preds:
            remaining[a] -= 1
        lane = None
        for a in preds:
            if a in out and lanes[out[a][0]] == a:
                lane = out[a][0]
                break
        if lane is None:
            for x, occ in enumerate(lanes):
                if occ is None or remaining.get(occ, 0) <= 0:
                    lane = x
                    break
        if lane is None:
            lanes.append(None)
            lane = len(lanes) - 1
        lanes[lane] = k
        out[k] = (lane, y)
    return out


def write_html(g, vertical, horizontal, keep, path, title):
    nodes = g.nodes
    vset = set(vertical)
    data_nodes = []
    for k in sorted(keep):
        lead, mem = g.ginfo(k)
        cls = op_class(lead["op"].split("(")[0])
        data_nodes.append({
            "k": k, "name": lead["name"], "op": lead["op"], "cls": cls, "color": COLORS[cls],
            "level": g.level[k], "slack": g.alap[k] - g.level[k],
            "members": [{"name": m["name"], "op": m["op"], "type": m["type"], "ne": m["ne"], "kind": g.kind[m["i"]],
                         "bytes": m["bytes"]} for m in mem],
            "kernels": [short_kernel(w) for w in g.kernels(k)],
            "weights": sorted({(g.ext[r["ext"]]["name"], g.ext[r["ext"]]["type"], g.ext[r["ext"]]["bytes"])
                               for m in mem for r in m["src"] if r and "ext" in r and g.ext[r["ext"]]["usage"] == "weights"}),
            "absorbed": sum(1 for m in mem if g.kind[m["i"]] == "absorbed"),
        })
    data_edges = []
    for (a, b), ts in g.edges.items():
        if a in keep and b in keep:
            data_edges.append({"a": a, "b": b, "t": ts, "v": (a, b) in vset})
    for a, b in g.order:
        if a in keep and b in keep:
            data_edges.append({"a": a, "b": b, "t": [], "order": True})
    data_h = []
    for name, nbytes, c in horizontal:
        c = [k for k in c if k in keep]
        if len(c) >= 2:
            data_h.append({"input": name, "bytes": nbytes, "ks": c})
    lay = {"levels": layout_levels(g, keep), "order": layout_order(g, keep)}
    payload = {
        "meta": g.meta, "title": title, "nodes": data_nodes, "edges": data_edges, "horizontal": data_h,
        "layout": {name: {str(k): v for k, v in l.items()} for name, l in lay.items()}, "colors": COLORS,
    }
    with open(path, "w") as f:
        f.write(HTML.replace("__TITLE__", html.escape(title)).replace("__DATA__", json.dumps(payload).replace("</", "<\\/")))


HTML = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
body { margin: 0; font: 12px system-ui, sans-serif; color: #222; background: #fafafa; }
#bar { position: fixed; top: 0; left: 0; right: 0; padding: 6px 10px; background: #fff; border-bottom: 1px solid #ddd; z-index: 2; }
#bar label { margin-right: 12px; }
#svg { position: absolute; top: 40px; left: 0; right: 360px; bottom: 0; }
#side { position: fixed; top: 40px; right: 0; width: 350px; bottom: 0; overflow: auto; padding: 5px; background: #fff;
        border-left: 1px solid #ddd; font: 11px ui-monospace, monospace; white-space: pre-wrap; }
.node rect { stroke: #333; stroke-width: 0.6; rx: 3; }
.node.multi rect { stroke-width: 2.2; }
.node text { font-size: 9px; pointer-events: none; fill: #fff; }
.edge { fill: none; stroke: #999; stroke-width: 0.8; }
.edge.v { stroke: #e08000; stroke-width: 2.2; }
.edge.order { stroke: #bbb; stroke-dasharray: 2 3; }
.hz { fill: none; stroke: #9b30d0; stroke-width: 1.6; stroke-dasharray: 5 3; }
.dim { opacity: 0.12; }
.hl rect { stroke: #000; stroke-width: 3; }
.legend span { display: inline-block; padding: 1px 5px; margin-right: 4px; color: #fff; border-radius: 3px; }
</style></head><body>
<div id="bar"><b>__TITLE__</b> &nbsp;
<label>layout <select id="lay"><option value="levels">dependency levels</option><option value="order">execution order</option></select></label>
<label><input type="checkbox" id="cv" checked> vertical candidates</label>
<label><input type="checkbox" id="ch" checked> horizontal candidates</label>
<label><input type="checkbox" id="co"> order edges</label>
<label>find <input id="q" size="18" placeholder="regex on names/ops"></label>
<span class="legend" id="legend"></span></div>
<svg id="svg"></svg>
<div id="side">click a launch: its tensors, fused nodes, inputs.
thick outline: several nodes in one launch (fused).
orange edge: vertical candidate (only reader).
purple dashed: horizontal candidate (same input, independent).
dotted: write-after-read order.
wheel: zoom, drag: pan.</div>
<script>
const D = __DATA__;
const NS = "http://www.w3.org/2000/svg";
const svg = document.getElementById("svg");
const side = document.getElementById("side");
const W = 150, H = 26, GX = 175, GY = 48;
const byK = {}; D.nodes.forEach(n => byK[n.k] = n);
const pred = {}, succ = {};
D.edges.forEach(e => { (succ[e.a] = succ[e.a] || []).push(e.b); (pred[e.b] = pred[e.b] || []).push(e.a); });
document.getElementById("legend").innerHTML = Object.entries(D.colors).map(([k, c]) => `<span style="background:${c}">${k}</span>`).join("");
let vb = null, selected = null;
function el(t, a, p) { const e = document.createElementNS(NS, t); for (const k in a) e.setAttribute(k, a[k]); if (p) p.appendChild(e); return e; }
function fmtB(b) { return b >= 1 << 20 ? (b / (1 << 20)).toFixed(1) + " MiB" : b >= 1024 ? (b / 1024).toFixed(1) + " KiB" : b + " B"; }
function draw() {
  svg.innerHTML = "";
  const L = D.layout[document.getElementById("lay").value];
  const P = k => { const p = L[k]; return [p[0] * GX + 10, p[1] * GY + 10]; };
  const root = el("g", {}, svg);
  const ge = el("g", {}, root), gh = el("g", {}, root), gn = el("g", {}, root);
  const showV = document.getElementById("cv").checked, showH = document.getElementById("ch").checked, showO = document.getElementById("co").checked;
  D.edges.forEach(e => {
    if (e.order && !showO) return;
    const [x1, y1] = P(e.a), [x2, y2] = P(e.b);
    const a = [x1 + W / 2, y1 + H], b = [x2 + W / 2, y2], my = (a[1] + b[1]) / 2;
    const p = el("path", { d: `M${a[0]},${a[1]} C${a[0]},${my} ${b[0]},${my} ${b[0]},${b[1]}`,
      class: "edge" + (e.v && showV ? " v" : "") + (e.order ? " order" : "") }, ge);
    p.dataset.a = e.a; p.dataset.b = e.b;
    if (e.t.length) el("title", {}, p).textContent = e.t.map(t => t[0] + " " + fmtB(t[1])).join("\n");
  });
  if (showH) D.horizontal.forEach(h => {
    for (let i = 1; i < h.ks.length; i++) {
      const [x1, y1] = P(h.ks[i - 1]), [x2, y2] = P(h.ks[i]);
      const p = el("path", { d: `M${x1 + W},${y1 + H / 2} Q${(x1 + x2) / 2 + W},${Math.min(y1, y2) - 20} ${x2 + W},${y2 + H / 2}`, class: "hz" }, gh);
      p.dataset.a = h.ks[i - 1]; p.dataset.b = h.ks[i];
      el("title", {}, p).textContent = "same input: " + h.input + " (" + fmtB(h.bytes) + ")";
    }
  });
  D.nodes.forEach(n => {
    const [x, y] = P(n.k);
    const g = el("g", { class: "node" + (n.members.length > 1 ? " multi" : ""), transform: `translate(${x},${y})` }, gn);
    g.dataset.k = n.k;
    el("rect", { width: W, height: H, fill: n.color, "fill-opacity": n.absorbed === n.members.length ? 0.35 : 1 }, g);
    const t1 = el("text", { x: 4, y: 11 }, g); t1.textContent = n.name.slice(0, 28);
    const t2 = el("text", { x: 4, y: 22 }, g);
    t2.textContent = n.members.map(m => m.op).join("+").slice(0, 30) + (n.members.length > 1 ? ` (${n.members.length})` : "");
    el("title", {}, g).textContent = `${n.name}\n${n.members.map(m => m.op).join(" + ")}\n${n.kernels.join("\n")}\nlevel ${n.level}, slack ${n.slack}`;
    g.onclick = ev => { ev.stopPropagation(); select(n.k); };
  });
  if (!vb) { const bb = root.getBBox(); vb = [bb.x - 10, bb.y - 10, Math.min(bb.width + 20, 1600), Math.min(bb.height + 20, 1200)]; }
  svg.setAttribute("viewBox", vb.join(" "));
  highlight();
}
function closure(k, adj) { const s = new Set([k]), st = [k]; while (st.length) { const x = st.pop(); (adj[x] || []).forEach(y => { if (!s.has(y)) { s.add(y); st.push(y); } }); } return s; }
function highlight() {
  const q = document.getElementById("q").value;
  let re = null; try { re = q ? new RegExp(q) : null; } catch (e) {}
  let keep = null;
  if (selected !== null) { const a = closure(selected, pred), b = closure(selected, succ); keep = new Set([...a, ...b]); }
  svg.querySelectorAll(".node").forEach(g => {
    const k = +g.dataset.k, n = byK[k];
    const hit = re && (re.test(n.name) || n.members.some(m => re.test(m.name) || re.test(m.op)));
    g.classList.toggle("dim", (keep && !keep.has(k)) || (re && !hit));
    g.classList.toggle("hl", k === selected || !!hit);
  });
  svg.querySelectorAll("path").forEach(p => {
    const a = +p.dataset.a, b = +p.dataset.b;
    p.classList.toggle("dim", !!keep && !(keep.has(a) && keep.has(b)));
  });
}
function select(k) {
  selected = k; highlight();
  const n = byK[k];
  const ins = D.edges.filter(e => e.b === k && !e.order), outs = D.edges.filter(e => e.a === k && !e.order);
  const hs = D.horizontal.filter(h => h.ks.includes(k));
  let s = `${n.name}\n${n.op}   level ${n.level}, slack ${n.slack}\n\nnodes in this launch:\n`;
  n.members.forEach(m => s += `  ${m.kind.padEnd(8)} ${m.op} ${m.name} ${m.type} [${m.ne.join(",")}] ${fmtB(m.bytes)}\n`);
  if (n.kernels.length) { s += "\nsubmitted:\n"; n.kernels.forEach(x => s += `  ${x}\n`); }
  if (n.weights.length) { s += "\nweights:\n"; n.weights.forEach(w => s += `  ${w[0]} ${w[1]} ${fmtB(w[2])}\n`); }
  s += "\ninputs from:\n"; ins.forEach(e => s += `  ${byK[e.a].name} (${byK[e.a].op})${e.v ? "  [vertical candidate]" : ""}\n` + e.t.map(t => `      ${t[0]} ${fmtB(t[1])}\n`).join(""));
  s += "\noutputs to:\n"; outs.forEach(e => s += `  ${byK[e.b].name} (${byK[e.b].op})${e.v ? "  [vertical candidate]" : ""}\n`);
  if (hs.length) { s += "\nhorizontal candidates:\n"; hs.forEach(h => s += `  input ${h.input} (${fmtB(h.bytes)}): ` + h.ks.map(x => byK[x].name).join(", ") + "\n"); }
  side.textContent = s;
}
svg.onclick = () => { selected = null; highlight(); };
svg.onwheel = ev => { ev.preventDefault(); const f = ev.deltaY > 0 ? 1.15 : 1 / 1.15; const r = svg.getBoundingClientRect();
  const mx = vb[0] + (ev.clientX - r.left) / r.width * vb[2], my = vb[1] + (ev.clientY - r.top) / r.height * vb[3];
  vb = [mx - (mx - vb[0]) * f, my - (my - vb[1]) * f, vb[2] * f, vb[3] * f]; svg.setAttribute("viewBox", vb.join(" ")); };
let drag = null;
svg.onmousedown = ev => drag = [ev.clientX, ev.clientY, vb[0], vb[1]];
window.onmouseup = () => drag = null;
window.onmousemove = ev => { if (!drag) return; const r = svg.getBoundingClientRect();
  vb[0] = drag[2] - (ev.clientX - drag[0]) / r.width * vb[2]; vb[1] = drag[3] - (ev.clientY - drag[1]) / r.height * vb[3]; svg.setAttribute("viewBox", vb.join(" ")); };
["lay", "cv", "ch", "co"].forEach(id => document.getElementById(id).onchange = () => { if (id === "lay") vb = null; draw(); });
document.getElementById("q").oninput = highlight;
draw();
</script></body></html>
"""


def write_dot(g, vertical, horizontal, keep, path):
    vset = set(vertical)
    with open(path, "w") as f:
        f.write("digraph G {\n  rankdir=TB; node [shape=box, style=filled, fontname=Helvetica, fontsize=9, fontcolor=white];\n")
        for k in sorted(keep):
            lead, mem = g.ginfo(k)
            cls = op_class(lead["op"].split("(")[0])
            label = "%s\\n%s" % (lead["name"], "+".join(m["op"] for m in mem))
            f.write('  g%d [label="%s", fillcolor="%s"%s];\n' % (k, label.replace('"', "'"), COLORS[cls],
                                                               ", penwidth=3" if len(mem) > 1 else ""))
        for (a, b) in g.edges:
            if a in keep and b in keep:
                f.write("  g%d -> g%d%s;\n" % (a, b, ' [color="#e08000", penwidth=2.5]' if (a, b) in vset else ""))
        for name, nbytes, c in horizontal:
            c = [k for k in c if k in keep]
            for x, y in zip(c, c[1:]):
                f.write('  g%d -> g%d [dir=none, style=dashed, color="#9b30d0", constraint=false];\n' % (x, y))
        f.write("}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump", help="graph-dev*.json written with GGML_SYCL_GRAPH_DUMP")
    ap.add_argument("-o", "--out", help="output prefix: <out>.html, <out>.txt (default: the dump's name)")
    ap.add_argument("--layers", help="only launches of these layers (e.g. 3,4 or 3-5) plus their direct neighbours")
    ap.add_argument("--dot", action="store_true", help="also write <out>.dot for graphviz")
    args = ap.parse_args()

    g = Graph(args.dump)
    vertical, horizontal = g.candidates()
    out = args.out or re.sub(r"\.json$", "", args.dump)

    keep = set(range(len(g.groups)))
    if args.layers:
        want = set()
        for part in args.layers.split(","):
            lo, _, hi = part.partition("-")
            want.update(range(int(lo), int(hi or lo) + 1))
        core = {k for k in keep if any(layer_of(m["name"]) in want for m in g.ginfo(k)[1])}
        keep = set(core)
        for k in core:
            keep.update(g.pred[k])
            keep.update(g.succ[k])

    with open(out + ".txt", "w") as f:
        report(g, vertical, horizontal, f)
    report(g, vertical, horizontal, sys.stdout)
    title = "dev%d seq%d t%d%s" % (g.meta["device"], g.meta["seq"], g.meta["tokens"],
                                    " layers " + args.layers if args.layers else "")
    write_html(g, vertical, horizontal, keep, out + ".html", title)
    if args.dot:
        write_dot(g, vertical, horizontal, keep, out + ".dot")
    print("\nwrote %s.html (%d launches), %s.txt" % (out, len(keep), out), file=sys.stderr)


if __name__ == "__main__":
    main()
