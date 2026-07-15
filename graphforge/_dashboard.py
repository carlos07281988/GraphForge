"""Auto-generated Agent Web Dashboard.

Provides a single-page HTML dashboard for any CompiledGraph,
with graph topology visualization and interactive execution.
"""
from __future__ import annotations
from typing import Any

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>GraphForge Dashboard</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,system-ui,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
     background:#0f0f1e;color:#e0e0e0;display:flex;height:100vh}
#sidebar{width:380px;background:#1a1a30;padding:20px;overflow-y:auto;border-right:1px solid #2a2a45}
#main{flex:1;padding:20px;overflow-y:auto}
h1{font-size:18px;color:#6366f1;margin-bottom:16px;letter-spacing:.5px}
h2{font-size:14px;color:#a0a0c0;margin:12px 0 8px;text-transform:uppercase;letter-spacing:1px}
label{display:block;font-size:12px;color:#8888aa;margin:8px 0 4px}
textarea,input[type=text]{width:100%;background:#0f0f1e;border:1px solid #2a2a45;border-radius:6px;
  color:#e0e0e0;padding:8px 12px;font-size:13px;font-family:monospace;resize:vertical}
textarea:focus,input:focus{outline:none;border-color:#6366f1}
button{background:#6366f1;color:#fff;border:none;border-radius:6px;padding:8px 20px;
  font-size:13px;cursor:pointer;margin-top:12px;transition:background .2s}
button:hover{background:#5355d1}
button:disabled{background:#3a3a60;cursor:not-allowed}
#graph-viz{background:#0f0f1e;border:1px solid #2a2a45;border-radius:8px;padding:16px;min-height:200px}
#result{background:#0f0f1e;border:1px solid #2a2a45;border-radius:8px;padding:16px;margin-top:12px;
  white-space:pre-wrap;font-family:monospace;font-size:12px;display:none;max-height:400px;overflow-y:auto}
.status-badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;margin:2px}
.status-ok{background:#065f4620;color:#34d399;border:1px solid #065f46}
.status-info{background:#1e3a5f20;color:#60a5fa;border:1px solid #1e3a5f}
.node-list{list-style:none;margin:8px 0}
.node-list li{padding:4px 0;font-size:12px;color:#a0a0c0}
.node-list li::before{content:"⬤ ";color:#6366f1;margin-right:4px}
</style></head>
<body>
<div id="sidebar">
<h1>⚡ GraphForge</h1>
<div id="graph-info"></div>
<h2>Nodes</h2>
<ul id="node-list" class="node-list"></ul>
<h2>Input State</h2>
<label>State JSON</label>
<textarea id="state-input" rows="8">{"messages": [{"role": "user", "content": "Hello"}]}</textarea>
<label>Config</label>
<input id="config-input" type="text" value='{"thread_id": "dashboard"}'/>
<button id="run-btn" onclick="runGraph()">▶ Execute</button>
<div id="status"></div>
</div>
<div id="main">
<h2>Graph Topology</h2>
<div id="graph-viz"><pre id="mermaid-src" style="font-size:11px;color:#666"></pre></div>
<h2>Execution Result</h2>
<div id="result"></div>
</div>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script>
let graphData = null;
async function loadGraph(){try{
  const r=await fetch('/info');graphData=await r.json();
  document.getElementById('graph-info').innerHTML=
    '<span class="status-badge status-ok">'+graphData.nodes+' nodes</span> '+
    '<span class="status-badge status-info">'+graphData.edges+' edges</span>';
  const ul=document.getElementById('node-list');
  graphData.node_names.forEach(n=>{
    const li=document.createElement('li');li.textContent=n;ul.appendChild(li)
  });
  renderGraph();
}catch(e){document.getElementById('graph-viz').innerHTML='<span style="color:#f87171">Failed to load graph</span>'}}

function renderGraph(){if(!graphData)return;
  let mermaidDef='graph TD;\\n';
  graphData.edges.forEach(e=>{mermaidDef+=`  ${e[0]}-->${e[1]||'__end__'};\\n`});
  document.getElementById('mermaid-src').textContent=mermaidDef;
  if(typeof mermaid!=='undefined'){mermaid.initialize({theme:'dark',themeVariables:{background:'#0f0f1e',primaryColor:'#6366f1'}});
  try{mermaid.run({nodes:[document.getElementById('mermaid-src')]})}catch(e){}}}

async function runGraph(){const btn=document.getElementById('run-btn'),status=document.getElementById('status');
  btn.disabled=true;status.innerHTML='<span class="status-badge status-info">Running...</span>';
  const resultDiv=document.getElementById('result');resultDiv.style.display='none';
  try{
    const state=JSON.parse(document.getElementById('state-input').value);
    const config=JSON.parse(document.getElementById('config-input').value);
    const r=await fetch('/invoke',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({state,config})});
    const data=await r.json();
    resultDiv.textContent=JSON.stringify(data,null,2);resultDiv.style.display='block';
    status.innerHTML='<span class="status-badge status-ok">Completed</span>';
  }catch(e){
    resultDiv.textContent='Error: '+e.message;resultDiv.style.display='block';
    status.innerHTML='<span class="status-badge" style="color:#f87171">Failed</span>';
  }finally{btn.disabled=false}}
loadGraph();
</script></body></html>"""

def get_dashboard_html() -> str:
    """Return the dashboard HTML page."""
    return DASHBOARD_HTML


__all__ = ["get_dashboard_html"]
