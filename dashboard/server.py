"""Servidor HTTP do dashboard (stdlib, sem dependencias novas).

Serve duas coisas:
  GET /            -> a pagina HTML (auto-atualiza via fetch ao /api/summary)
  GET /api/summary -> JSON com todo o estado (KPIs, posicoes, trades, etc.)

A pagina e estatica; todo o dado vem do endpoint JSON, recarregado a cada N
segundos no navegador. Assim o painel sempre reflete o SQLite mais recente sem
reiniciar o servidor.
"""

from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from dashboard.broker_view import fetch_live_view
from dashboard.queries import build_dashboard_data

logger = logging.getLogger("dashboard.server")


def _collect(db_path: Path | str, use_broker: bool) -> dict:
    account = positions = None
    status = "off"
    if use_broker:
        account, positions, status = fetch_live_view()
    return build_dashboard_data(
        db_path, account=account, live_positions=positions, broker_status=status
    )


def make_handler(db_path: Path | str, use_broker: bool):
    class Handler(BaseHTTPRequestHandler):
        # silencia o log padrao ruidoso do http.server
        def log_message(self, fmt, *args):  # noqa: A003
            logger.debug("%s - %s", self.address_string(), fmt % args)

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send(200, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/api/summary":
                try:
                    data = _collect(db_path, use_broker)
                    body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
                    self._send(200, body, "application/json; charset=utf-8")
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Falha ao montar /api/summary")
                    body = json.dumps({"error": str(exc)}).encode("utf-8")
                    self._send(500, body, "application/json; charset=utf-8")
            else:
                self._send(404, b"not found", "text/plain; charset=utf-8")

    return Handler


def serve(
    db_path: Path | str = "data/trading.sqlite",
    *,
    host: str = "127.0.0.1",
    port: int = 8787,
    use_broker: bool = True,
) -> None:
    handler = make_handler(db_path, use_broker)
    httpd = ThreadingHTTPServer((host, port), handler)
    logger.info(
        "Dashboard em http://%s:%d  (broker=%s, db=%s)",
        host, port, "on" if use_broker else "off", db_path,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Encerrando dashboard.")
    finally:
        httpd.server_close()


# --------------------------------------------------------------------------- #
# Pagina (HTML + CSS + JS embutidos; sem assets externos)
# --------------------------------------------------------------------------- #

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NEXUS · Trading Console</title>
<style>
  :root{
    --bg:#05070d; --bg2:#0a0f1c; --panel:rgba(18,26,44,.55); --panel-solid:#0e1525;
    --border:rgba(120,160,255,.14); --border-hi:rgba(120,180,255,.45);
    --text:#e8f0ff; --muted:#7d8db5; --dim:#566089;
    --cyan:#22d3ee; --blue:#5b8cff; --violet:#a855f7; --pink:#ec4899;
    --green:#2ee6a6; --red:#ff5d73; --amber:#ffb547;
    --glow-cyan:0 0 24px rgba(34,211,238,.35);
  }
  *{box-sizing:border-box;}
  html,body{margin:0;height:100%;}
  body{
    background:var(--bg); color:var(--text); overflow-x:hidden;
    font:14px/1.55 "Inter",-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
    -webkit-font-smoothing:antialiased;
  }
  /* fundo animado: orbs + grade em perspectiva */
  .bg{position:fixed; inset:0; z-index:-2; overflow:hidden; background:
      radial-gradient(900px 600px at 12% -5%, rgba(91,140,255,.16), transparent 60%),
      radial-gradient(800px 600px at 100% 0%, rgba(168,85,247,.14), transparent 55%),
      radial-gradient(700px 500px at 50% 110%, rgba(34,211,238,.10), transparent 60%),
      var(--bg);}
  .bg::before{content:""; position:absolute; inset:-40% 0 0; background-image:
      linear-gradient(rgba(91,140,255,.07) 1px,transparent 1px),
      linear-gradient(90deg,rgba(91,140,255,.07) 1px,transparent 1px);
    background-size:46px 46px; transform:perspective(420px) rotateX(62deg);
    transform-origin:top center; mask-image:linear-gradient(to bottom,#000,transparent 70%);
    animation:drift 22s linear infinite;}
  @keyframes drift{from{background-position:0 0,0 0}to{background-position:0 46px,46px 0}}
  .orb{position:fixed; z-index:-1; border-radius:50%; filter:blur(70px); opacity:.5; pointer-events:none;}
  .orb.a{width:380px;height:380px;background:#1d3aff;top:-120px;left:-80px;animation:float1 18s ease-in-out infinite;}
  .orb.b{width:320px;height:320px;background:#a855f7;bottom:-120px;right:-60px;animation:float2 21s ease-in-out infinite;}
  @keyframes float1{50%{transform:translate(60px,40px) scale(1.1)}}
  @keyframes float2{50%{transform:translate(-50px,-30px) scale(1.08)}}

  header{position:sticky; top:0; z-index:20; display:flex; align-items:center; gap:18px;
    padding:14px 26px; backdrop-filter:blur(14px);
    background:linear-gradient(180deg,rgba(5,7,13,.92),rgba(5,7,13,.55));
    border-bottom:1px solid var(--border);}
  .brand{display:flex; align-items:center; gap:12px;}
  .logo{width:34px;height:34px;border-radius:9px;position:relative;
    background:conic-gradient(from 180deg,var(--cyan),var(--blue),var(--violet),var(--cyan));
    box-shadow:var(--glow-cyan); animation:spin 8s linear infinite;}
  .logo::after{content:"◆";position:absolute;inset:0;display:grid;place-items:center;
    color:#03060f;font-size:14px;font-weight:900;}
  @keyframes spin{to{transform:rotate(360deg)}}
  .brand h1{font-size:15px;margin:0;font-weight:700;letter-spacing:.14em;}
  .brand small{display:block;font-size:9px;letter-spacing:.34em;color:var(--muted);font-weight:600;}
  .spacer{flex:1;}
  .chip{display:inline-flex;align-items:center;gap:7px;font-size:11px;font-weight:600;
    padding:6px 12px;border-radius:999px;border:1px solid var(--border);color:var(--muted);
    letter-spacing:.04em;text-transform:uppercase;}
  .dot{width:8px;height:8px;border-radius:50%;background:var(--muted);}
  .chip.live{color:var(--green);border-color:rgba(46,230,166,.5);box-shadow:0 0 16px rgba(46,230,166,.25) inset;}
  .chip.live .dot{background:var(--green);box-shadow:0 0 0 0 rgba(46,230,166,.7);animation:pulse 1.8s infinite;}
  .chip.err{color:var(--red);border-color:rgba(255,93,115,.5);} .chip.err .dot{background:var(--red);}
  @keyframes pulse{0%{box-shadow:0 0 0 0 rgba(46,230,166,.6)}70%{box-shadow:0 0 0 8px rgba(46,230,166,0)}100%{box-shadow:0 0 0 0 rgba(46,230,166,0)}}
  .ring{width:30px;height:30px;transform:rotate(-90deg);cursor:pointer;}
  .ring circle{fill:none;stroke-width:3;}
  .ring .bgc{stroke:rgba(120,160,255,.16);} .ring .fgc{stroke:var(--cyan);stroke-linecap:round;
    transition:stroke-dashoffset .3s linear;filter:drop-shadow(0 0 4px var(--cyan));}
  .btn{cursor:pointer;font:inherit;font-size:12px;font-weight:600;color:var(--text);
    background:rgba(120,160,255,.08);border:1px solid var(--border);border-radius:9px;
    padding:7px 12px;transition:.2s;letter-spacing:.03em;}
  .btn:hover{border-color:var(--border-hi);background:rgba(120,160,255,.16);box-shadow:0 0 14px rgba(91,140,255,.2);}
  #updated{font-size:11px;color:var(--dim);min-width:90px;text-align:right;}

  main{max-width:1320px;margin:0 auto;padding:24px 26px 60px;}
  .tabs{display:flex;gap:6px;margin-bottom:22px;flex-wrap:wrap;}
  .tab{cursor:pointer;font-size:12px;font-weight:600;letter-spacing:.05em;text-transform:uppercase;
    color:var(--muted);padding:9px 16px;border-radius:10px;border:1px solid transparent;transition:.2s;}
  .tab:hover{color:var(--text);}
  .tab.on{color:var(--text);border-color:var(--border-hi);
    background:linear-gradient(180deg,rgba(91,140,255,.18),rgba(91,140,255,.04));
    box-shadow:0 0 18px rgba(91,140,255,.18);}
  .view{display:none;animation:fade .4s ease;}
  .view.on{display:block;}
  @keyframes fade{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}

  .grid{display:grid;gap:16px;}
  .kpis{grid-template-columns:repeat(auto-fit,minmax(186px,1fr));margin-bottom:22px;}
  .cols2{grid-template-columns:1.4fr 1fr;}
  @media(max-width:920px){.cols2{grid-template-columns:1fr;}}

  .card{position:relative;border-radius:16px;padding:18px 20px;overflow:hidden;
    background:var(--panel);border:1px solid var(--border);backdrop-filter:blur(12px);
    transition:transform .25s, border-color .25s, box-shadow .25s;}
  .card::before{content:"";position:absolute;inset:0;border-radius:16px;padding:1px;
    background:linear-gradient(140deg,rgba(120,180,255,.5),transparent 40%);
    -webkit-mask:linear-gradient(#000 0 0) content-box,linear-gradient(#000 0 0);
    -webkit-mask-composite:xor;mask-composite:exclude;opacity:.6;pointer-events:none;}
  .card:hover{transform:translateY(-3px);border-color:var(--border-hi);
    box-shadow:0 14px 40px -20px rgba(91,140,255,.5);}
  .kpi .label{font-size:10px;letter-spacing:.16em;text-transform:uppercase;color:var(--muted);font-weight:700;}
  .kpi .value{font-size:27px;font-weight:750;margin-top:8px;letter-spacing:-.01em;
    font-variant-numeric:tabular-nums;}
  .kpi .sub{font-size:11px;color:var(--dim);margin-top:5px;}
  .kpi .spark{position:absolute;right:14px;bottom:10px;opacity:.5;}
  .pos{color:var(--green);} .neg{color:var(--red);} .zero{color:var(--muted);}
  .glowtext.pos{text-shadow:0 0 18px rgba(46,230,166,.45);}
  .glowtext.neg{text-shadow:0 0 18px rgba(255,93,115,.4);}

  .panel-title{display:flex;align-items:center;gap:10px;margin:0 0 14px;font-size:12px;
    letter-spacing:.12em;text-transform:uppercase;color:var(--muted);font-weight:700;}
  .panel-title .ico{width:6px;height:18px;border-radius:3px;
    background:linear-gradient(var(--cyan),var(--blue));box-shadow:var(--glow-cyan);}
  section{margin-bottom:22px;}

  table{width:100%;border-collapse:collapse;font-size:13px;}
  th,td{text-align:left;padding:9px 12px;white-space:nowrap;}
  thead th{font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);
    font-weight:700;border-bottom:1px solid var(--border);cursor:pointer;user-select:none;position:relative;}
  thead th:hover{color:var(--text);}
  th.sorted::after{content:"▾";position:absolute;margin-left:5px;color:var(--cyan);}
  th.sorted.asc::after{content:"▴";}
  tbody td{border-bottom:1px solid rgba(120,160,255,.06);font-variant-numeric:tabular-nums;}
  tbody tr{transition:background .15s;}
  tbody tr:hover{background:rgba(91,140,255,.07);}
  td.num,th.num{text-align:right;}
  .tag{font-size:10px;font-weight:600;padding:2px 9px;border-radius:6px;letter-spacing:.03em;
    border:1px solid var(--border);color:var(--muted);background:rgba(120,160,255,.05);}
  .tag.trend_up,.tag.win{color:var(--green);border-color:rgba(46,230,166,.4);}
  .tag.trend_down,.tag.loss{color:var(--red);border-color:rgba(255,93,115,.4);}
  .tag.high_vol{color:var(--amber);border-color:rgba(255,181,71,.4);}
  .side-buy{color:var(--green);font-weight:700;} .side-sell{color:var(--red);font-weight:700;}
  .empty{color:var(--dim);padding:26px 12px;text-align:center;font-style:italic;}
  .mini{height:6px;border-radius:4px;background:rgba(120,160,255,.1);overflow:hidden;min-width:70px;display:inline-block;vertical-align:middle;}
  .mini>span{display:block;height:100%;border-radius:4px;}
  code.ev{color:var(--cyan);font-family:"SF Mono",ui-monospace,monospace;font-size:12px;}
  .search{font:inherit;font-size:12px;color:var(--text);background:rgba(5,7,13,.6);
    border:1px solid var(--border);border-radius:9px;padding:7px 12px;width:200px;outline:none;transition:.2s;}
  .search:focus{border-color:var(--border-hi);box-shadow:0 0 14px rgba(91,140,255,.2);}
  .toolbar{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:12px;flex-wrap:wrap;}
  .banner{border:1px solid rgba(255,93,115,.5);background:rgba(255,93,115,.08);color:var(--red);
    border-radius:12px;padding:12px 16px;margin-bottom:18px;font-size:13px;}
  /* chart */
  .chartwrap{position:relative;height:260px;}
  svg.chart{width:100%;height:100%;overflow:visible;}
  .chart .area{transition:opacity .3s;}
  .chart-tip{position:absolute;pointer-events:none;background:var(--panel-solid);border:1px solid var(--border-hi);
    border-radius:8px;padding:6px 10px;font-size:11px;opacity:0;transition:opacity .12s;white-space:nowrap;
    box-shadow:0 8px 30px -8px rgba(0,0,0,.7);transform:translate(-50%,-130%);}
  .scanline{stroke:var(--cyan);stroke-width:1;opacity:0;stroke-dasharray:3 3;}
  .legend{display:flex;gap:18px;font-size:11px;color:var(--muted);margin-top:8px;}
  .legend i{width:18px;height:3px;border-radius:2px;display:inline-block;vertical-align:middle;margin-right:6px;}
</style>
</head>
<body>
<div class="bg"></div><div class="orb a"></div><div class="orb b"></div>

<header>
  <div class="brand">
    <div class="logo"></div>
    <div><h1>NEXUS</h1><small>TRADING CONSOLE</small></div>
  </div>
  <span id="broker" class="chip off"><span class="dot"></span><span id="brokerLabel">broker</span></span>
  <div class="spacer"></div>
  <button class="btn" id="pauseBtn">⏸ Pausar</button>
  <button class="btn" id="refreshBtn">⟳ Atualizar</button>
  <svg class="ring" id="ring" viewBox="0 0 36 36" title="próxima atualização">
    <circle class="bgc" cx="18" cy="18" r="15"></circle>
    <circle class="fgc" cx="18" cy="18" r="15" stroke-dasharray="94.2" stroke-dashoffset="0"></circle>
  </svg>
  <span id="updated">—</span>
</header>

<main>
  <div id="err"></div>
  <div class="tabs" id="tabs">
    <div class="tab on" data-v="overview">◎ Visão Geral</div>
    <div class="tab" data-v="track">▲ Track Record</div>
    <div class="tab" data-v="trades">⇅ Trades & Ordens</div>
    <div class="tab" data-v="audit">◇ Auditoria</div>
  </div>

  <!-- OVERVIEW -->
  <div class="view on" data-view="overview">
    <div class="grid kpis" id="kpis"></div>
    <div class="grid cols2">
      <section><div class="card">
        <h3 class="panel-title"><span class="ico"></span>P&L Realizado Acumulado</h3>
        <div class="chartwrap"><svg class="chart" id="pnlChart"></svg><div class="chart-tip" id="tip"></div></div>
        <div class="legend"><span><i style="background:linear-gradient(90deg,var(--cyan),var(--blue))"></i>Curva de P&L</span></div>
      </div></section>
      <section><div class="card">
        <h3 class="panel-title"><span class="ico"></span>Posições Abertas</h3>
        <div id="positions"></div>
      </div></section>
    </div>
    <div class="grid cols2">
      <section><div class="card">
        <h3 class="panel-title"><span class="ico"></span>🏆 Maiores Ganhos</h3>
        <div id="winners"></div>
      </div></section>
      <section><div class="card">
        <h3 class="panel-title"><span class="ico"></span>🩸 Maiores Perdas</h3>
        <div id="losers"></div>
      </div></section>
    </div>
    <div class="grid cols2">
      <section><div class="card">
        <h3 class="panel-title"><span class="ico"></span>Por Estratégia</h3><div id="byStrategy"></div>
      </div></section>
      <section><div class="card">
        <h3 class="panel-title"><span class="ico"></span>Por Regime</h3><div id="byRegime"></div>
      </div></section>
    </div>
  </div>

  <!-- TRACK RECORD -->
  <div class="view" data-view="track">
    <div class="grid kpis" id="trKpis"></div>
    <section><div class="card">
      <h3 class="panel-title"><span class="ico"></span>Curva de Equity vs Benchmark (base 100, since-inception)</h3>
      <div class="chartwrap"><svg class="chart" id="navChart"></svg><div class="chart-tip" id="navTip"></div></div>
      <div class="legend">
        <span><i style="background:var(--cyan)"></i>Estratégia (NAV)</span>
        <span><i style="background:var(--amber)"></i>SPY (buy&hold)</span>
        <span><i style="background:var(--violet)"></i>60/40 (SPY/IEF)</span>
      </div>
    </div></section>
    <section><div class="card">
      <p style="color:var(--dim);font-size:11px;margin:0;line-height:1.6">
        Rentabilidade passada não representa garantia de resultados futuros. Track record de
        simulação/paper trading — não houve gestão de recursos de terceiros. Série append-only com
        cadeia de hash; integridade verificável via <code class="ev">reporting.verify_chain</code>.
      </p>
    </div></section>
  </div>

  <!-- TRADES -->
  <div class="view" data-view="trades">
    <section><div class="card">
      <div class="toolbar">
        <h3 class="panel-title" style="margin:0"><span class="ico"></span>Ordens em Aberto</h3>
      </div>
      <div id="openOrders"></div>
    </div></section>
    <section><div class="card">
      <div class="toolbar">
        <h3 class="panel-title" style="margin:0"><span class="ico"></span>Trades Recentes</h3>
        <input class="search" id="tradeSearch" placeholder="🔍 filtrar símbolo / estratégia…">
      </div>
      <div id="trades"></div>
    </div></section>
  </div>

  <!-- AUDIT -->
  <div class="view" data-view="audit">
    <section><div class="card">
      <div class="toolbar">
        <h3 class="panel-title" style="margin:0"><span class="ico"></span>Trilha de Auditoria</h3>
        <input class="search" id="auditSearch" placeholder="🔍 filtrar evento / ator…">
      </div>
      <div id="audit"></div>
    </div></section>
  </div>
</main>

<script>
const REFRESH_MS = 10000;
let DATA = null, paused = false, tickStart = Date.now();
const sortState = {}; // por tabela
const filters = {trades:"", audit:""};

const fmtUSD = v => (v==null) ? "—" :
  (v<0?"-":"")+"$"+Math.abs(v).toLocaleString("en-US",{minimumFractionDigits:2,maximumFractionDigits:2});
const fmtNum = v => (v==null) ? "—" : Number(v).toLocaleString("en-US",{maximumFractionDigits:4});
const fmtPct = v => (v==null) ? "—" : (v*100).toFixed(2)+"%";
const cls = v => (v==null||v===0) ? "zero" : (v>0?"pos":"neg");
const esc = s => (s==null?"":String(s)).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;"}[c]));
const tshort = ts => { if(!ts) return "—"; const d=new Date(ts);
  return isNaN(d)? esc(ts):d.toLocaleString("pt-BR",{day:"2-digit",month:"2-digit",hour:"2-digit",minute:"2-digit"}); };

/* ---------- count-up animado nos KPIs ---------- */
function animNum(el, to, fmt){
  const from = Number(el.dataset.v||0); el.dataset.v = to;
  if(from===to){ el.textContent = fmt(to); return; }
  const t0 = performance.now(), dur = 600;
  function step(t){ const k = Math.min(1,(t-t0)/dur); const e = 1-Math.pow(1-k,3);
    el.textContent = fmt(from+(to-from)*e); if(k<1) requestAnimationFrame(step); }
  requestAnimationFrame(step);
}

function kpiCard(id,label,sub){
  return `<div class="card kpi"><div class="label">${label}</div>
    <div class="value glowtext" id="kpi-${id}">—</div><div class="sub" id="sub-${id}">${sub||""}</div></div>`;
}
function buildKpis(){
  document.getElementById("kpis").innerHTML = [
    kpiCard("equity","Equity"), kpiCard("upnl","P&L Não Realizado"),
    kpiCard("rpnl","P&L Realizado"), kpiCard("wr","Win Rate"),
    kpiCard("pf","Profit Factor"), kpiCard("bw","Melhor / Pior"),
  ].join("");
}
function setKpi(id,val,fmt,sub,signed){
  const el=document.getElementById("kpi-"+id); if(!el) return;
  el.className="value glowtext "+(signed?cls(val):"");
  if(typeof val==="number") animNum(el,val,fmt); else el.textContent=val;
  if(sub!=null) document.getElementById("sub-"+id).innerHTML=sub;
}
function renderKpis(k){
  setKpi("equity",k.equity==null?"—":k.equity,fmtUSD,k.cash!=null?("cash "+fmtUSD(k.cash)):"snapshot local");
  setKpi("upnl",k.unrealized_pnl??0,fmtUSD,(k.open_positions||0)+" posições abertas",true);
  setKpi("rpnl",k.total_pnl??0,fmtUSD,(k.n_closed||0)+" trades fechados",true);
  setKpi("wr",k.win_rate==null?"—":fmtPct(k.win_rate),null,`${k.n_wins||0}W · ${k.n_losses||0}L`);
  setKpi("pf",k.profit_factor==null?"—":fmtNum(k.profit_factor),null,"ganho méd. "+fmtUSD(k.avg_win));
  setKpi("bw",k.best==null?"—":fmtUSD(k.best),null,"pior "+fmtUSD(k.worst),true);
}

/* ---------- tabela genérica com ordenação ---------- */
function renderTable(host, cols, rows, rowFn, empty, tableId){
  if(!rows||!rows.length){ host.innerHTML=`<div class="empty">${empty||"Sem dados ainda."}</div>`; return; }
  const st = tableId? sortState[tableId] : null;
  if(st && st.col!=null){
    const c = cols[st.col];
    rows = rows.slice().sort((a,b)=>{
      let x=a[c.key], y=b[c.key];
      if(typeof x==="number"||typeof y==="number"){x=x??-Infinity;y=y??-Infinity;return st.dir*(x-y);}
      return st.dir*String(x??"").localeCompare(String(y??""));
    });
  }
  const head = cols.map((c,i)=>{
    const on = st&&st.col===i; const asc = on&&st.dir>0;
    return `<th class="${c.num?'num ':''}${on?'sorted '+(asc?'asc':''):''}" data-col="${i}">${c.label}</th>`;
  }).join("");
  host.innerHTML = `<table data-tid="${tableId||''}"><thead><tr>${head}</tr></thead><tbody>${rows.map(rowFn).join("")}</tbody></table>`;
  if(tableId) host.querySelectorAll("th").forEach(th=>th.onclick=()=>{
    const col=+th.dataset.col, cur=sortState[tableId];
    sortState[tableId] = (cur&&cur.col===col)? {col,dir:-cur.dir} : {col,dir:-1};
    render(); // re-render só re-desenha
  });
}

function renderPositions(rows){
  renderTable(document.getElementById("positions"),
    [{label:"Símbolo",key:"symbol"},{label:"Qtd",key:"qty",num:1},{label:"Médio",key:"avg_entry_price",num:1},
     {label:"Atual",key:"current_price",num:1},{label:"Valor",key:"market_value",num:1},
     {label:"P&L",key:"unrealized_pnl",num:1},{label:"%",key:"unrealized_pnl_pct",num:1}],
    rows, p=>`<tr><td><b>${esc(p.symbol)}</b></td><td class="num">${fmtNum(p.qty)}</td>
      <td class="num">${fmtUSD(p.avg_entry_price)}</td><td class="num">${fmtUSD(p.current_price)}</td>
      <td class="num">${fmtUSD(p.market_value)}</td>
      <td class="num ${cls(p.unrealized_pnl)}">${fmtUSD(p.unrealized_pnl)}</td>
      <td class="num ${cls(p.unrealized_pnl_pct)}">${fmtPct(p.unrealized_pnl_pct)}</td></tr>`,
    "Nenhuma posição aberta.","pos");
}
function renderTop(id,rows){
  renderTable(document.getElementById(id),
    [{label:"Símbolo",key:"symbol"},{label:"Estratégia",key:"strategy"},{label:"Regime",key:"regime"},
     {label:"P&L",key:"realized_pnl",num:1},{label:"Retorno",key:"return_pct",num:1},{label:"Fechado",key:"closed_at"}],
    rows, t=>`<tr><td><b>${esc(t.symbol)}</b></td><td>${esc(t.strategy)}</td>
      <td><span class="tag ${esc(t.regime)}">${esc(t.regime)}</span></td>
      <td class="num ${cls(t.realized_pnl)}">${fmtUSD(t.realized_pnl)}</td>
      <td class="num ${cls(t.return_pct)}">${fmtPct(t.return_pct)}</td>
      <td class="zero">${tshort(t.closed_at)}</td></tr>`,
    "Sem trades fechados ainda.",id);
}
function renderBreakdown(id,rows){
  const max=Math.max(1,...rows.map(r=>Math.abs(r.pnl)));
  renderTable(document.getElementById(id),
    [{label:"Nome",key:"key"},{label:"Trades",key:"n",num:1},{label:"Win",key:"win_rate",num:1},
     {label:"P&L",key:"pnl",num:1},{label:""}],
    rows, r=>`<tr><td><b>${esc(r.key)}</b></td><td class="num">${r.n}</td>
      <td class="num">${fmtPct(r.win_rate)}</td>
      <td class="num ${cls(r.pnl)}">${fmtUSD(r.pnl)}</td>
      <td><span class="mini"><span style="width:${Math.abs(r.pnl)/max*100}%;background:${r.pnl<0?'var(--red)':'linear-gradient(90deg,var(--cyan),var(--green))'}"></span></span></td></tr>`,
    "Sem dados agregados ainda.",id);
}
function renderOrders(rows){
  renderTable(document.getElementById("openOrders"),
    [{label:"Símbolo",key:"symbol"},{label:"Lado",key:"side"},{label:"Qtd",key:"qty",num:1},
     {label:"Tipo",key:"order_type"},{label:"Limite/Stop",key:"limit_price",num:1},
     {label:"Estratégia",key:"strategy"},{label:"Status",key:"status"},{label:"Atualizado",key:"updated_at"}],
    rows, o=>`<tr><td><b>${esc(o.symbol)}</b></td>
      <td class="side-${esc(o.side)}">${esc((o.side||'').toUpperCase())}</td><td class="num">${fmtNum(o.qty)}</td>
      <td>${esc(o.order_type)}</td>
      <td class="num">${o.limit_price?esc(o.limit_price):(o.stop_price?('stop '+esc(o.stop_price)):'—')}</td>
      <td>${esc(o.strategy)}</td><td><span class="tag">${esc(o.status)}</span></td>
      <td class="zero">${tshort(o.updated_at)}</td></tr>`,
    "Nenhuma ordem em aberto.","ord");
}
function renderTrades(rows){
  const f=filters.trades.toLowerCase();
  if(f) rows=rows.filter(t=>(t.symbol+" "+t.strategy).toLowerCase().includes(f));
  renderTable(document.getElementById("trades"),
    [{label:"Quando",key:"ts"},{label:"Símbolo",key:"symbol"},{label:"Lado",key:"side"},
     {label:"Qtd",key:"qty",num:1},{label:"Preço",key:"price",num:1},
     {label:"Estratégia",key:"strategy"},{label:"Status",key:"status"}],
    rows, t=>`<tr><td class="zero">${tshort(t.ts)}</td><td><b>${esc(t.symbol)}</b></td>
      <td class="side-${esc(t.side)}">${esc((t.side||'').toUpperCase())}</td><td class="num">${fmtNum(t.qty)}</td>
      <td class="num">${fmtUSD(t.price)}</td><td>${esc(t.strategy)}</td>
      <td><span class="tag">${esc(t.status)}</span></td></tr>`,
    "Nenhum trade executado ainda.","trd");
}
function renderAudit(rows){
  const f=filters.audit.toLowerCase();
  if(f) rows=rows.filter(a=>(a.actor+" "+a.event+" "+(a.symbol||"")).toLowerCase().includes(f));
  renderTable(document.getElementById("audit"),
    [{label:"Quando",key:"ts"},{label:"Ator",key:"actor"},{label:"Evento",key:"event"},
     {label:"Símbolo",key:"symbol"},{label:"Detalhe"}],
    rows, a=>`<tr><td class="zero">${tshort(a.ts)}</td><td><span class="tag">${esc(a.actor)}</span></td>
      <td><code class="ev">${esc(a.event)}</code></td><td>${esc(a.symbol||'—')}</td>
      <td class="zero">${esc(typeof a.payload==='object'?JSON.stringify(a.payload):a.payload).slice(0,80)}</td></tr>`,
    "Sem eventos de auditoria.","aud");
}

/* ---------- gráfico SVG de P&L acumulado ---------- */
function renderChart(series){
  const svg=document.getElementById("pnlChart"); const tip=document.getElementById("tip");
  const W=svg.clientWidth||600, H=svg.clientHeight||240, pad={l:8,r:8,t:14,b:18};
  if(!series||series.length<1){
    svg.innerHTML=`<text x="50%" y="50%" text-anchor="middle" fill="var(--dim)" font-style="italic" font-size="13">A curva aparece quando os primeiros trades fecharem.</text>`;
    return;
  }
  const pts = series.length===1 ? [series[0],series[0]] : series;
  const ys = pts.map(p=>p.cum), min=Math.min(0,...ys), max=Math.max(0,...ys);
  const span=(max-min)||1;
  const X=i=>pad.l+(i/(pts.length-1))*(W-pad.l-pad.r);
  const Y=v=>pad.t+(1-(v-min)/span)*(H-pad.t-pad.b);
  const line=pts.map((p,i)=>`${i?'L':'M'}${X(i).toFixed(1)},${Y(p.cum).toFixed(1)}`).join(" ");
  const area=`${line} L${X(pts.length-1).toFixed(1)},${Y(min).toFixed(1)} L${X(0).toFixed(1)},${Y(min).toFixed(1)} Z`;
  const zeroY=Y(0).toFixed(1);
  const last=series[series.length-1].cum, up=last>=0;
  const stroke=up?"var(--green)":"var(--red)";
  svg.innerHTML=`
    <defs>
      <linearGradient id="g" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stop-color="${up?'rgba(46,230,166,.45)':'rgba(255,93,115,.4)'}"/>
        <stop offset="1" stop-color="rgba(10,15,28,0)"/>
      </linearGradient>
      <filter id="glow"><feGaussianBlur stdDeviation="3" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
    </defs>
    <line x1="${pad.l}" x2="${W-pad.r}" y1="${zeroY}" y2="${zeroY}" stroke="rgba(120,160,255,.18)" stroke-dasharray="4 4"/>
    <path class="area" d="${area}" fill="url(#g)"/>
    <path d="${line}" fill="none" stroke="${stroke}" stroke-width="2.4" filter="url(#glow)" stroke-linejoin="round"/>
    <line class="scanline" id="scan" x1="0" x2="0" y1="${pad.t}" y2="${H-pad.b}"/>
    <circle id="hover" r="4.5" fill="${stroke}" stroke="#03060f" stroke-width="2" opacity="0"/>`;
  // hit area p/ tooltip
  const scan=svg.querySelector("#scan"), hov=svg.querySelector("#hover");
  svg.onmousemove=e=>{
    const r=svg.getBoundingClientRect(); const mx=e.clientX-r.left;
    let i=Math.round(((mx-pad.l)/(W-pad.l-pad.r))*(pts.length-1));
    i=Math.max(0,Math.min(pts.length-1,i)); const p=pts[i];
    const x=X(i),y=Y(p.cum);
    scan.setAttribute("x1",x);scan.setAttribute("x2",x);scan.style.opacity=.6;
    hov.setAttribute("cx",x);hov.setAttribute("cy",y);hov.style.opacity=1;
    tip.style.opacity=1;tip.style.left=x+"px";tip.style.top=y+"px";
    tip.innerHTML=`<b>${esc(p.symbol||'')}</b> · ${tshort(p.t)}<br>acum <b class="${cls(p.cum)}">${fmtUSD(p.cum)}</b> · trade ${fmtUSD(p.pnl)}`;
  };
  svg.onmouseleave=()=>{scan.style.opacity=0;hov.style.opacity=0;tip.style.opacity=0;};
}

/* ---------- track record: KPIs since-inception ---------- */
function renderTrackKpis(tr){
  const host=document.getElementById("trKpis"); if(!host) return;
  if(!tr || !tr.n_days || tr.n_days<2){
    host.innerHTML=`<div class="empty" style="grid-column:1/-1">A curva de track record aparece quando houver ≥2 snapshots EOD em nav_history.</div>`;
    return;
  }
  const card=(label,val,sub,signed)=>`<div class="card kpi"><div class="label">${label}</div>
    <div class="value ${signed?cls(typeof val==='number'?val:0):''}">${val}</div><div class="sub">${sub||""}</div></div>`;
  host.innerHTML=[
    card("CAGR",fmtPct(tr.cagr),`since ${esc(tr.since||'—')}`,true),
    card("Sharpe",fmtNum(tr.sharpe),`vol ${fmtPct(tr.vol)} a.a.`),
    card("Sortino",fmtNum(tr.sortino),`${tr.n_days} dias de NAV`),
    card("MaxDD",fmtPct(tr.max_drawdown),`Calmar ${fmtNum(tr.calmar)}`,true),
    card("Excess vs SPY",fmtPct(tr.excess_cagr),`β ${fmtNum(tr.beta_vs_spy)}`,true),
    card("Meses +",fmtPct(tr.pct_positive_months),"% meses positivos"),
  ].join("");
}

/* ---------- track record: curva multi-linha (estratégia vs benchmarks) ---------- */
function renderNavChart(series){
  const svg=document.getElementById("navChart"), tip=document.getElementById("navTip");
  if(!svg) return;
  const W=svg.clientWidth||600, H=svg.clientHeight||240, pad={l:8,r:8,t:14,b:18};
  if(!series||series.length<2){
    svg.innerHTML=`<text x="50%" y="50%" text-anchor="middle" fill="var(--dim)" font-style="italic" font-size="13">A curva aparece quando houver ≥2 snapshots EOD.</text>`;
    return;
  }
  const lines=[{k:"strat",c:"var(--cyan)"},{k:"spy",c:"var(--amber)"},{k:"sf",c:"var(--violet)"}];
  const vals=[]; series.forEach(p=>lines.forEach(l=>{ if(p[l.k]!=null) vals.push(p[l.k]); }));
  if(!vals.length){ svg.innerHTML=""; return; }
  const min=Math.min(...vals), max=Math.max(...vals), span=(max-min)||1;
  const X=i=>pad.l+(i/(series.length-1))*(W-pad.l-pad.r);
  const Y=v=>pad.t+(1-(v-min)/span)*(H-pad.t-pad.b);
  const path=l=>{ let d="",pen=false;
    series.forEach((p,i)=>{ const v=p[l.k]; if(v==null){pen=false;return;}
      d+=(pen?"L":"M")+X(i).toFixed(1)+","+Y(v).toFixed(1)+" "; pen=true; }); return d; };
  const base100=Y(100).toFixed(1);
  svg.innerHTML=`
    <line x1="${pad.l}" x2="${W-pad.r}" y1="${base100}" y2="${base100}" stroke="rgba(120,160,255,.18)" stroke-dasharray="4 4"/>
    ${lines.map(l=>`<path d="${path(l)}" fill="none" stroke="${l.c}" stroke-width="2.2" stroke-linejoin="round"/>`).join("")}
    <line class="scanline" id="navScan" x1="0" x2="0" y1="${pad.t}" y2="${H-pad.b}"/>`;
  const scan=svg.querySelector("#navScan");
  svg.onmousemove=e=>{
    const r=svg.getBoundingClientRect(); const mx=e.clientX-r.left;
    let i=Math.round(((mx-pad.l)/(W-pad.l-pad.r))*(series.length-1));
    i=Math.max(0,Math.min(series.length-1,i)); const p=series[i];
    scan.setAttribute("x1",X(i));scan.setAttribute("x2",X(i));scan.style.opacity=.6;
    tip.style.opacity=1;tip.style.left=X(i)+"px";tip.style.top=pad.t+"px";
    tip.innerHTML=`${tshort(p.t)}<br>estrat <b>${p.strat!=null?p.strat.toFixed(1):'—'}</b> · SPY ${p.spy!=null?p.spy.toFixed(1):'—'} · 60/40 ${p.sf!=null?p.sf.toFixed(1):'—'}`;
  };
  svg.onmouseleave=()=>{scan.style.opacity=0;tip.style.opacity=0;};
}

/* ---------- render mestre (sem refetch) ---------- */
function render(){
  if(!DATA) return; const d=DATA;
  renderKpis(d.kpis||{});
  renderChart(d.pnl_curve||[]);
  renderPositions(d.positions||[]);
  renderTop("winners",(d.top_trades||{}).winners||[]);
  renderTop("losers",(d.top_trades||{}).losers||[]);
  renderBreakdown("byStrategy",d.by_strategy||[]);
  renderBreakdown("byRegime",d.by_regime||[]);
  renderOrders(d.open_orders||[]);
  renderTrades(d.recent_trades||[]);
  renderAudit(d.audit||[]);
  renderTrackKpis(d.track_record||{});
  renderNavChart(d.nav_curve||[]);
}

async function load(){
  try{
    const r=await fetch("/api/summary",{cache:"no-store"}); DATA=await r.json();
    document.getElementById("err").innerHTML = DATA.error?`<div class="banner">⚠ ${esc(DATA.error)}</div>`:"";
    const st=DATA.broker_status||"off", chip=document.getElementById("broker");
    chip.className="chip "+(st==="live"?"live":(st.startsWith("error")?"err":"off"));
    document.getElementById("brokerLabel").textContent= st==="live"?"AO VIVO":(st.startsWith("error")?"OFFLINE":"SÓ DB");
    render();
    document.getElementById("updated").textContent="↻ "+new Date().toLocaleTimeString("pt-BR");
    tickStart=Date.now();
  }catch(e){ document.getElementById("updated").textContent="falha: "+e; }
}

/* ---------- tabs ---------- */
document.getElementById("tabs").onclick=e=>{
  const t=e.target.closest(".tab"); if(!t) return;
  document.querySelectorAll(".tab").forEach(x=>x.classList.toggle("on",x===t));
  document.querySelectorAll(".view").forEach(v=>v.classList.toggle("on",v.dataset.view===t.dataset.v));
  if(t.dataset.v==="overview"||t.dataset.v==="track") render();
};
/* ---------- controles ---------- */
document.getElementById("refreshBtn").onclick=load;
document.getElementById("pauseBtn").onclick=function(){ paused=!paused;
  this.textContent=paused?"▶ Retomar":"⏸ Pausar"; if(!paused){load();} };
document.getElementById("tradeSearch").oninput=e=>{filters.trades=e.target.value;renderTrades((DATA||{}).recent_trades||[]);};
document.getElementById("auditSearch").oninput=e=>{filters.audit=e.target.value;renderAudit((DATA||{}).audit||[]);};

/* ---------- anel de countdown + loop ---------- */
const ringFg=document.querySelector(".ring .fgc"), C=2*Math.PI*15;
ringFg.setAttribute("stroke-dasharray",C.toFixed(1));
document.getElementById("ring").onclick=load;
function tickRing(){
  if(!paused){
    const k=Math.min(1,(Date.now()-tickStart)/REFRESH_MS);
    ringFg.setAttribute("stroke-dashoffset",(C*k).toFixed(1));
    if(k>=1) load();
  } else ringFg.setAttribute("stroke-dashoffset",C);
  requestAnimationFrame(tickRing);
}

buildKpis(); load(); tickRing();
</script>
</body>
</html>"""
