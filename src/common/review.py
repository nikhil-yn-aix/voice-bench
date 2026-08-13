import csv
import html
import json
import os
import shutil
from pathlib import Path

from src.common.config import DATA, OUTPUTS, ROOT


def escape(value):
    return html.escape(str(value))


def number(value, digits=3):
    return "not measured" if value is None else f"{value:.{digits}f}"


def megabytes(value):
    return None if value is None else value / 1024**2


def metric(model, key):
    value = model.get("warm_metrics", {}).get(key, {})
    return value.get("mean") if isinstance(value, dict) else value


def shell(title, body, parent=""):
    prefix = "../" if parent else ""
    style = """
:root{color-scheme:light;--ink:#17212b;--muted:#52606d;--line:#cbd2d9;
--paper:#fff;--soft:#f4f6f8;--link:#075ea8;--good:#087f5b;--warn:#c92a2a;
--accent:#d8f3e8;--shadow:0 8px 24px rgba(23,33,43,.07)}
*{box-sizing:border-box}
body{margin:0;background:var(--soft);color:var(--ink);
font:16px/1.55 Inter,Segoe UI,Arial,sans-serif}
header{background:#17212b;color:white;border-bottom:4px solid #20c997;
padding:28px max(24px,calc((100vw - 1180px)/2)) 22px}
header p{color:#d9e2ec;margin:4px 0 0}
nav{display:flex;flex-wrap:wrap;gap:20px;margin-top:18px}
nav a{color:#9bd0ff;font-weight:700}
main{max-width:1180px;margin:30px auto;padding:0 24px 56px}
h1{font-size:32px;line-height:1.2;margin:0}h2{font-size:22px;margin:34px 0 12px}
h3{font-size:17px;margin:0 0 8px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(235px,1fr));gap:20px}
.card{background:white;border:1px solid var(--line);border-top:4px solid var(--ink);
box-shadow:var(--shadow);padding:16px}.card p{margin:6px 0}
.stat{font-size:24px;font-weight:700}.muted{color:var(--muted)}
.pill{display:inline-block;border:1px solid var(--line);padding:2px 6px;
margin:2px 5px 2px 0;font:13px ui-monospace,SFMono-Regular,Consolas,monospace}
.pill.good{border-color:var(--good);color:var(--good)}
.pill.warn{border-color:var(--warn);color:var(--warn)}a{color:var(--link)}
.table-wrap{overflow:auto;border-top:2px solid var(--ink);border-bottom:1px solid var(--line)}
table{border-collapse:collapse;width:100%}
th,td{padding:11px 10px;text-align:left;vertical-align:top;border-bottom:1px solid var(--line)}
th{font-size:13px;color:var(--muted);background:#e9eef2;white-space:nowrap}
tbody tr:nth-child(even){background:#f8fafb}
tr:last-child td{border-bottom:0}audio{width:min(100%,360px);height:40px}
.sample{margin:22px 0}.output{background:white;border-left:4px solid var(--line);
padding:9px 13px;margin:12px 0}
.plot figure{margin:0}.plot img{width:100%;height:auto;display:block}
.plot figcaption{color:var(--muted);
font-size:14px;padding:10px 4px 2px}.plot{padding:10px;text-decoration:none;
transition:transform .15s ease,box-shadow .15s ease}.plot:hover{box-shadow:0 12px 30px
rgba(23,33,43,.13);transform:translateY(-2px)}
.plot-grid{grid-template-columns:repeat(2,minmax(0,1fr))}
.links{display:flex;flex-wrap:wrap;gap:16px}.rank{font-size:14px}.rank strong{font-size:18px}
.reference{font:14px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace;
background:var(--soft);padding:12px}
.rating{border:0;border-top:1px solid var(--line);margin:16px 0 0;padding:12px 0 0}
.rating legend{font-weight:700;padding-right:10px}.scale{display:flex;gap:5px;margin-top:7px}
.scale label{border:1px solid var(--line);cursor:pointer;min-width:38px;padding:7px;
text-align:center}.scale input{position:absolute;opacity:0}
.scale label:has(input:checked){background:var(--ink);color:white}
textarea{border:1px solid var(--line);font:inherit;min-height:72px;padding:9px;width:100%}
.toolbar{align-items:center;background:var(--accent);border:1px solid #96d8c1;display:flex;
flex-wrap:wrap;gap:12px;margin:18px 0;padding:14px}.toolbar button{background:var(--good);
border:0;color:white;cursor:pointer;font:700 15px Arial;padding:10px 15px}
.toolbar button:hover{filter:brightness(.92)}.toolbar button.secondary{background:white;
border:1px solid var(--ink);color:var(--ink)}
.sentence{background:white;border:1px solid var(--line);border-top:5px solid var(--ink);
box-shadow:var(--shadow);margin:36px 0;padding:4px 20px 20px}
.comparison{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:24px}
.result{background:white;border:4px solid var(--good);box-shadow:var(--shadow);
margin:24px 0;padding:20px}.result h2{color:var(--good);margin-top:0}
.submit-error:empty{display:none}
.tabs{background:var(--soft);border-bottom:1px solid var(--line);display:flex;flex-wrap:wrap;
margin:24px 0;position:sticky;top:0;z-index:5}
.tabs button{background:transparent;border:0;border-bottom:4px solid transparent;cursor:pointer;
font:700 16px Arial;padding:12px 16px}.tabs button.active{border-color:var(--good)}
.panel{display:none}.panel.active{display:block}
.frame{background:white;border:1px solid var(--line);
height:75vh;width:100%}.plot-group{margin-top:30px}
.plot-group h3{border-bottom:1px solid var(--line);
padding-bottom:8px}
.rank strong{color:var(--good)}
@media(max-width:650px){header{padding:22px 16px}main{padding:0 16px 36px}
.grid{gap:14px}.plot-grid{grid-template-columns:1fr}th,td{padding:9px 7px}}
"""
    nav = "".join(
        [
            f"<a href='{prefix}index.html'>results</a>",
            f"<a href='{prefix}asr/index.html'>asr samples</a>",
            f"<a href='{prefix}tts/index.html'>tts listening test</a>",
            f"<a href='{prefix}capabilities/index.html'>feature tests</a>",
        ]
    )
    return "".join(
        [
            "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
            "<meta name='viewport' content='width=device-width,initial-scale=1'>",
            "<title>",
            escape(title),
            "</title><style>",
            style,
            "</style></head><body><header><h1>",
            escape(title),
            "</h1><p>local cpu voice benchmark</p><nav>",
            nav,
            "</nav></header><main>",
            body,
            "</main></body></html>",
        ]
    )


def primary_test(model, test_id):
    return next(
        (
            item
            for item in model["per_test_results"]
            if item["test_id"] == test_id
            and item["repetition"] == 0
            and item.get("benchmark_profile") == "deployment"
            and item.get("benchmark_mode") == "native"
            and item.get("benchmark_scope") == "full"
        ),
        None,
    )


def asr_review(master):
    target = OUTPUTS / "review" / "asr"
    target.mkdir(parents=True, exist_ok=True)
    manifest_path = DATA / "asr" / "manifest.json"
    if not manifest_path.exists():
        return
    samples = json.loads(manifest_path.read_text(encoding="utf-8"))["samples"]
    audio_target = OUTPUTS / "audio" / "asr_input"
    audio_target.mkdir(parents=True, exist_ok=True)
    cards = []
    for sample in samples:
        source = Path(sample["path"])
        source = source if source.is_absolute() else ROOT / source
        copied = audio_target / f"{sample['id']}{source.suffix}"
        if source.exists():
            shutil.copy2(source, copied)
        audio = Path("../../audio/asr_input") / copied.name
        outputs = []
        for name, model in master["asr"]["models"].items():
            item = primary_test(model, sample["id"])
            if not item:
                continue
            accuracy = item["accuracy"]
            timing = item["timing"]
            resources = item["resources"]
            outputs.append(
                "".join(
                    [
                        "<div class='output'><h3>",
                        escape(name),
                        "</h3><span class='pill'>wer ",
                        number(accuracy.get("wer")),
                        "</span><span class='pill'>cer ",
                        number(accuracy.get("cer")),
                        "</span><span class='pill'>rtf ",
                        number(timing.get("rtf")),
                        "</span><span class='pill'>peak ",
                        number(resources.get("peak_rss_bytes", 0) / 1024**2, 1),
                        " mb</span><p>",
                        escape(accuracy["hypothesis_raw"]),
                        "</p></div>",
                    ]
                )
            )
        cards.append(
            "".join(
                [
                    "<section class='card sample'><h2>",
                    escape(sample["id"].replace("_", " ")),
                    "</h2><audio controls preload='none' src='",
                    str(audio).replace("\\", "/"),
                    "'></audio><p class='muted'>reference transcript</p><p class='reference'>",
                    escape(sample["transcript"]),
                    "</p>",
                    "".join(outputs) or "<p class='muted'>no measured deployment result</p>",
                    "</section>",
                ]
            )
        )
    body = (
        "<p>listen to the input, read the reference, then inspect each model output.</p>"
        + "".join(cards)
    )
    (target / "index.html").write_text(
        shell("asr transcript review", body, "asr"), encoding="utf-8"
    )


def tts_review():
    blind = OUTPUTS / "audio" / "blind"
    manifest_path = blind / "evaluation.json"
    if not manifest_path.exists():
        return
    target = OUTPUTS / "review" / "tts"
    target.mkdir(parents=True, exist_ok=True)
    items = json.loads(manifest_path.read_text(encoding="utf-8"))
    text_cases = json.loads((ROOT / "config" / "tts_texts.json").read_text(encoding="utf-8"))
    case_order = {item["id"]: index for index, item in enumerate(text_cases)}
    original_order = {item["blind_id"]: index for index, item in enumerate(items)}
    items.sort(key=lambda item: (case_order[item["test_id"]], original_order[item["blind_id"]]))
    key_path = OUTPUTS / "processed" / "blind_key.json"
    identities = {
        item["blind_id"]: item["model"] for item in json.loads(key_path.read_text(encoding="utf-8"))
    }
    cards = {}
    criteria = ["naturalness", "intelligibility", "prosody", "speaker appeal", "artifact free"]
    for item in items:
        source = Path("../../audio/blind") / item["file"]
        controls = []
        for criterion in criteria:
            key = criterion.replace(" ", "_")
            choices = "".join(
                "<label><input type='radio' name='"
                + escape(item["blind_id"] + "_" + key)
                + "' data-field='"
                + key
                + "' value='"
                + str(score)
                + "'>"
                + str(score)
                + "</label>"
                for score in range(1, 6)
            )
            controls.append(
                "<fieldset class='rating'><legend>"
                + escape(criterion)
                + "</legend><div class='scale'>"
                + choices
                + "</div></fieldset>"
            )
        cards.setdefault(item["test_id"], []).append(
            "".join(
                [
                    "<section class='card sample' data-id='",
                    escape(item["blind_id"]),
                    "'><h3>",
                    escape(item["blind_id"]),
                    "</h3><audio controls preload='none' src='",
                    str(source).replace("\\", "/"),
                    "'></audio>",
                    "".join(controls),
                    "<p><label>notes</label><textarea data-field='notes'></textarea></p></section>",
                ]
            )
        )
    groups = []
    for case in text_cases:
        case_cards = cards.get(case["id"], [])
        if case_cards:
            groups.append(
                "<section class='sentence'><h2>"
                + escape(case["id"].replace("_", " "))
                + "</h2><p class='reference'>"
                + escape(case["text"])
                + "</p><div class='comparison'>"
                + "".join(case_cards)
                + "</div></section>"
            )
    script = """
<script>
const key='voice-bench-ratings-v1';
const submittedKey='voice-bench-ratings-submitted-v1';
const fields=['naturalness','intelligibility','prosody','speaker_appeal','artifact_free'];
const labels={naturalness:'naturalness',intelligibility:'intelligibility',
  prosody:'prosody',speaker_appeal:'speaker appeal',artifact_free:'artifact-free'};
const identities=__IDENTITIES__;
const samples=[...document.querySelectorAll('.sample')];
const read=()=>{try{return JSON.parse(localStorage.getItem(key)||'{}')}catch{return {}}};
const values=read();
for(const sample of samples){
  const row=values[sample.dataset.id]||{};
  for(const input of sample.querySelectorAll('[data-field]')){
    if(input.type==='radio')input.checked=String(row[input.dataset.field])===input.value;
    else input.value=row[input.dataset.field]||'';
  }
}
function collect(){
  const result={};
  for(const sample of samples){
    const row={};
    for(const field of fields){
      row[field]=sample.querySelector(`[data-field="${field}"]:checked`)?.value||'';
    }
    row.notes=sample.querySelector('[data-field="notes"]').value;
    result[sample.dataset.id]=row;
  }
  return result;
}
function status(){
  const data=collect();
  const done=samples.filter(sample=>fields.every(field=>data[sample.dataset.id][field])).length;
  for(const item of document.querySelectorAll('.progress')){
    item.textContent=`${done} of ${samples.length} clips complete`;
  }
}
function reveal(){
  const data=collect();
  const missing=samples.filter(sample=>fields.some(field=>!data[sample.dataset.id][field]));
  if(missing.length){
    for(const item of document.querySelectorAll('.submit-error')){
      item.textContent=`rate all ${samples.length} clips first. ${missing.length} remain.`;
    }
    return;
  }
  const scores={};
  for(const sample of samples){
    const model=identities[sample.dataset.id];
    const empty={clips:0,naturalness:0,intelligibility:0,prosody:0,
      speaker_appeal:0,artifact_free:0};
    const total=scores[model]||(scores[model]=empty);
    total.clips+=1;
    for(const field of fields)total[field]+=Number(data[sample.dataset.id][field]);
  }
  const rows=Object.entries(scores).map(([model,total])=>{
    const averages=Object.fromEntries(fields.map(field=>[field,total[field]/total.clips]));
    const overall=fields.reduce((sum,field)=>sum+averages[field],0)/fields.length;
    return {model,overall,...averages};
  }).sort((a,b)=>b.overall-a.overall);
  const cells=rows.map(row=>`<tr><td><strong>${row.model}</strong></td>`+
    `<td>${row.overall.toFixed(2)}</td>`+
    fields.map(field=>`<td>${row[field].toFixed(2)}</td>`).join('')+'</tr>').join('');
  const result=document.querySelector('#result');
  const heads=fields.map(field=>`<th>${labels[field]}</th>`).join('');
  result.innerHTML=`<h2>quality winner: ${rows[0].model}</h2>`+
    `<p>${rows[0].overall.toFixed(2)} out of 5.</p>`+
    `<div class="table-wrap"><table><thead><tr><th>model</th><th>overall</th>`+
    `${heads}</tr></thead><tbody>${cells}</tbody></table></div>`+
    `<p class="muted">overall is the equal mean of all five fields.</p>`;
  result.hidden=false;
  for(const item of document.querySelectorAll('.submit-error'))item.textContent='';
  localStorage.setItem(submittedKey,'1');
  result.scrollIntoView({behavior:'smooth',block:'start'});
}
document.addEventListener('input',()=>{
  localStorage.setItem(key,JSON.stringify(collect()));
  localStorage.removeItem(submittedKey);
  document.querySelector('#result').hidden=true;
  for(const item of document.querySelectorAll('.submit-error'))item.textContent='';
  status();
});
for(const button of document.querySelectorAll('[data-submit]')){
  button.addEventListener('click',reveal);
}
document.querySelector('#clear').addEventListener('click',()=>{
  if(confirm('clear every saved rating?')){
    localStorage.removeItem(key);
    localStorage.removeItem(submittedKey);
    location.reload();
  }
});
status();
if(localStorage.getItem(submittedKey)==='1')reveal();
</script>
""".replace("__IDENTITIES__", json.dumps(identities, separators=(",", ":")))
    body = (
        "<p>each section holds the same sentence from every model. compare within the section. "
        "model order stays hidden. 1 is poor and 5 is strong.</p>"
        "<div class='toolbar'><strong class='progress'></strong>"
        "<button data-submit>submit and reveal winner</button>"
        "<button id='clear' class='secondary'>clear saved ratings</button></div>"
        "<p class='submit-error pill warn'></p>"
        + "".join(groups)
        + "<div class='toolbar'><strong class='progress'></strong>"
        "<button data-submit>submit and reveal winner</button></div>"
        "<p class='submit-error pill warn'></p>"
        "<section id='result' class='result' hidden></section>" + script
    )
    (target / "index.html").write_text(shell("tts blind review", body, "tts"), encoding="utf-8")


def capability_review(master):
    target = OUTPUTS / "review" / "capabilities"
    target.mkdir(parents=True, exist_ok=True)
    cards = []
    for name, model in master["tts"]["models"].items():
        rows = []
        for item in model.get("special_test_results", []):
            feature = item["feature"]
            status = item.get("status", "unknown")
            display_status = status
            detail = item.get("error", "")
            if feature == "voice_cloning" and status != "ok" and "weights" in detail:
                display_status = "gated"
                detail = "optional voice-cloning weights were unavailable; built-in voices passed"
            variant = item.get("variant")
            if variant is None:
                variant = "reference audio" if feature == "voice_cloning" else "default"
            badge = "good" if display_status == "ok" else "warn"
            path = item.get("path")
            player = ""
            if path:
                relative = os.path.relpath(Path(path), target).replace("\\", "/")
                player = f"<audio controls preload='none' src='{escape(relative)}'></audio>"
            rows.append(
                "".join(
                    [
                        "<tr><td>",
                        escape(feature.replace("_", " ")),
                        "</td><td>",
                        escape(variant),
                        "</td><td><span class='pill ",
                        badge,
                        "'>",
                        escape(display_status),
                        "</span></td><td>",
                        player or escape(detail),
                        "</td></tr>",
                    ]
                )
            )
        if rows:
            cards.append(
                "<h2>" + escape(name) + "</h2><div class='table-wrap'><table><thead><tr>"
                "<th>feature</th><th>variant</th><th>status</th><th>output</th></tr></thead><tbody>"
                + "".join(rows)
                + "</tbody></table></div>"
            )
    body = (
        "<p>model-specific controls are tested only where the official runtime supports them.</p>"
        + ("".join(cards) or "<p class='muted'>no measured capability outputs</p>")
    )
    (target / "index.html").write_text(
        shell("tts capability review", body, "capabilities"), encoding="utf-8"
    )


def ranking_cards(master, kind):
    cards = []
    budget = master["benchmark_config"]["memory_budget_mb"]
    for profile, result in master["rankings"].get(kind, {}).items():
        order = result.get("eligible_order", [])
        leader = order[0] if order else None
        cards.append(
            "".join(
                [
                    "<section class='card rank'><p class='muted'>",
                    escape(profile.replace("_", " ")),
                    "</p><strong>",
                    escape(leader or "no eligible result"),
                    "</strong><p>",
                    f"under {number(budget, 0)} mb",
                    "</p></section>",
                ]
            )
        )
    return "".join(cards)


def winner_card(master, kind, profile, reason):
    order = master["rankings"][kind][profile]["order"]
    winner = order[0] if order else "no measured result"
    return (
        "<section class='card'><p class='muted'>"
        + kind
        + " winner</p><p class='stat'>"
        + escape(winner)
        + "</p><p>"
        + escape(reason)
        + "</p></section>"
    )


def measurement_table(master, kind):
    quality = "summary.wer.mean" if kind == "asr" else "summary.asr_wer.mean"
    latency = "summary.total_s.mean" if kind == "asr" else "summary.first_audio_s.mean"
    rows = []
    for name, model in master[kind]["models"].items():
        rows.append(
            "<tr><td><strong>"
            + escape(name)
            + "</strong></td><td>"
            + number(metric(model, quality))
            + "</td><td>"
            + number(metric(model, "summary.rtf.mean"))
            + "</td><td>"
            + number(metric(model, latency), 2)
            + " s</td><td>"
            + number(megabytes(metric(model, "summary.average_rss_bytes.mean")), 1)
            + " mb</td><td>"
            + number(megabytes(metric(model, "summary.peak_rss_bytes.max")), 1)
            + " mb</td><td>"
            + number(megabytes(metric(model, "model_disk_bytes")), 1)
            + " mb</td></tr>"
        )
    latency_name = "mean time" if kind == "asr" else "first audio"
    return (
        "<div class='table-wrap'><table><thead><tr><th>model</th><th>error rate</th>"
        "<th>rtf</th><th>"
        + latency_name
        + "</th><th>average ram</th><th>maximum ram</th><th>model disk</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def human_rating_table():
    path = OUTPUTS / "processed" / "human_ratings_summary.csv"
    if not path.exists():
        return ""
    with path.open(newline="", encoding="utf-8") as source:
        rows = sorted(csv.DictReader(source), key=lambda row: float(row["overall"]), reverse=True)
    body = "".join(
        "<tr><td><strong>"
        + escape(row["model"])
        + "</strong></td><td>"
        + number(float(row["overall"]), 2)
        + "</td><td>"
        + number(float(row["naturalness"]), 2)
        + "</td><td>"
        + number(float(row["intelligibility"]), 2)
        + "</td><td>"
        + number(float(row["prosody"]), 2)
        + "</td><td>"
        + number(float(row["speaker_appeal"]), 2)
        + "</td></tr>"
        for row in rows
    )
    return (
        "<h2>completed blind listening result</h2><p>one listener, six clips per model, "
        "five-point scales. small gaps need more listeners.</p><div class='table-wrap'><table>"
        "<thead><tr><th>model</th><th>overall</th><th>naturalness</th>"
        "<th>intelligibility</th><th>prosody</th><th>speaker appeal</th></tr></thead><tbody>"
        + body
        + "</tbody></table></div>"
    )


def dashboard(master):
    target = OUTPUTS / "review"
    plots = sorted((OUTPUTS / "plots").glob("*.png"))
    groups = {
        "decision trade-offs": ["pareto", "ranking", "tradeoff", "deployment_score"],
        "accuracy and scenarios": ["wer", "cer", "scenario", "intelligibility", "clipping"],
        "latency and throughput": ["latency", "first_audio", "rtf", "cpu_time", "average_cpu"],
        "memory and size": ["peak_ram", "average_ram", "disk_vs", "quality_per"],
    }
    plot_groups = []
    used = set()
    for title, terms in groups.items():
        selected = [path for path in plots if any(term in path.stem for term in terms)]
        selected = [path for path in selected if path not in used]
        used.update(selected)
        if not selected:
            continue
        cards = "".join(
            "<a class='card plot' href='../plots/"
            + escape(path.name)
            + "'><figure><img loading='lazy' src='../plots/"
            + escape(path.name)
            + "' alt='"
            + escape(path.stem.replace("_", " "))
            + "'><figcaption>"
            + escape(path.stem.replace("_", " "))
            + "</figcaption></figure></a>"
            for path in selected
        )
        plot_groups.append(
            "<section class='plot-group'><h3>"
            + escape(title)
            + "</h3><div class='grid plot-grid'>"
            + cards
            + "</div></section>"
        )
    environment = master["environment"]
    export_names = [
        "summary.csv",
        "rankings.csv",
        "per_test.csv",
        "capabilities.csv",
        "failures.csv",
        "human_ratings.csv",
        "human_ratings_summary.csv",
    ]
    links = "".join(
        "<a href='../processed/" + name + "'>" + name + "</a>"
        for name in export_names
        if (OUTPUTS / "processed" / name).exists()
    )
    overview = "".join(
        [
            "<h2>winners</h2><div class='grid'>",
            winner_card(
                master,
                "asr",
                "latency_first",
                "fastest useful accuracy; long-form memory exceeds the target",
            ),
            winner_card(
                master,
                "tts",
                "quality_first",
                "strong blind rating, best evaluator wer, and faster than real time",
            ),
            "</div><h2>machine</h2><div class='grid'>"
            "<section class='card'><p class='muted'>cpu</p><p class='stat'>",
            escape(environment.get("processor") or environment.get("platform", "unknown")),
            "</p></section><section class='card'><p class='muted'>logical cores</p>",
            "<p class='stat'>",
            escape(environment.get("logical_cores", "unknown")),
            "</p></section><section class='card'><p class='muted'>system ram</p><p class='stat'>",
            number(environment.get("ram_bytes", 0) / 1024**3, 1),
            " gb</p></section></div><h2>asr leaders</h2><div class='grid'>",
            ranking_cards(master, "asr"),
            "</div><h2>tts leaders</h2><div class='grid'>",
            ranking_cards(master, "tts"),
            "</div><h2>asr measurements</h2><p>lower is better for every numeric column.</p>",
            measurement_table(master, "asr"),
            "<h2>tts measurements</h2>"
            "<p>tts error rate checks intelligibility, not voice quality.</p>",
            measurement_table(master, "tts"),
            human_rating_table(),
            "<h2>metric guide</h2><div class='grid'>",
            "<section class='card'><strong>error rate</strong>"
            "<p>incorrect words divided by reference words.</p></section>",
            "<section class='card'><strong>real-time factor</strong>"
            "<p>compute time divided by audio time. below 1 is faster than real time.</p>"
            "</section>",
            "<section class='card'><strong>maximum ram</strong>"
            "<p>largest process memory in any measured case.</p></section>",
            "<section class='card'><strong>first audio</strong>"
            "<p>delay before the first playable sound.</p></section></div>",
        ]
    )
    script = """
<script>
const buttons=[...document.querySelectorAll('[data-tab]')];
const panels=[...document.querySelectorAll('.panel')];
for(const button of buttons)button.addEventListener('click',()=>{
  for(const item of buttons)item.classList.toggle('active',item===button);
  for(const panel of panels)panel.classList.toggle('active',panel.id===button.dataset.tab);
});
</script>
"""
    body = "".join(
        [
            "<div class='tabs'><button class='active' data-tab='overview'>summary</button>",
            "<button data-tab='plots'>charts</button>",
            "<button data-tab='asr'>asr samples</button>",
            "<button data-tab='tts'>tts listening test</button>",
            "<button data-tab='capabilities'>feature tests</button>",
            "<button data-tab='data'>exports</button></div>",
            "<section class='panel active' id='overview'>",
            overview,
            "</section><section class='panel' id='plots'>",
            "".join(plot_groups) or "<p class='muted'>no measured plots</p>",
            "</section><section class='panel' id='asr'><iframe class='frame' ",
            "src='asr/index.html' title='asr transcript review'></iframe></section>",
            "<section class='panel' id='tts'><iframe class='frame' src='tts/index.html' ",
            "title='tts blind rating'></iframe></section>",
            "<section class='panel' id='capabilities'><iframe class='frame' ",
            "src='capabilities/index.html' title='capability review'></iframe></section>",
            "<section class='panel' id='data'><h2>exported tables</h2><div class='links'>",
            links,
            "<a href='../master.json'>master.json</a>"
            "<a href='../../conclusion.md'>conclusion.md</a></div>",
            "<p class='muted'>master.json contains the machine-readable result. ",
            "the csv files are flat views for analysis.</p></section>",
            script,
        ]
    )
    (target / "index.html").write_text(shell("voice benchmark results", body), encoding="utf-8")


def report(master):
    lines = ["# measured results", ""]
    budget = master["benchmark_config"]["memory_budget_mb"]
    for kind in ["asr", "tts"]:
        lines.extend([f"## {kind}", ""])
        for profile, ranking in master["rankings"].get(kind, {}).items():
            order = ranking.get("eligible_order", [])
            leader = order[0] if order else "none"
            lines.append(f"{profile.replace('_', ' ')} leader under {budget} mb: {leader}")
        lines.extend(["", "| model | measured | failures |", "|---|---:|---:|"])
        for name, model in master[kind]["models"].items():
            measured = "yes" if model["warm_metrics"] else "no"
            lines.append(f"| {name} | {measured} | {len(model['failures'])} |")
        lines.append("")
    (OUTPUTS / "report.md").write_text("\n".join(lines), encoding="utf-8")


def build_review(master):
    target = OUTPUTS / "review"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    asr_review(master)
    tts_review()
    capability_review(master)
    dashboard(master)
    report(master)
