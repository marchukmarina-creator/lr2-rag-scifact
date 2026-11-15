"""Re-score saved responses with explicit DOC-prefix normalization; retain raw data."""
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

import lr2_eval as ev
from lr2_core import parse_response, save_json


def main():
    archive = Path(sys.argv[1])
    out = Path(sys.argv[2])
    with zipfile.ZipFile(archive) as z:
        rows = [json.loads(line) for line in z.read('runs/lr2_final/results.jsonl').splitlines()]
    assert len(rows) == 420 and len({r['job_id'] for r in rows}) == 420
    changed = 0
    for r in rows:
        raw = r.get('raw_response')
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if not isinstance(parsed, dict) or not isinstance(parsed.get('doc_id'), str):
            continue
        match = re.fullmatch(r'DOC (\d+)', parsed['doc_id'])
        if not match:
            continue
        parsed['doc_id'] = match.group(1)
        ok, value, error = parse_response(json.dumps(parsed), r['mode'], r['transmitted'])
        if ok:
            r['parse_ok'], r['parsed'], r['parse_error'], r['status'] = True, value, None, 'ok'
            r['normalization'] = 'Removed literal DOC prefix from doc_id; raw_response unchanged'
            changed += 1
    groups = {g: [r for r in rows if r['group'] == g] for g in sorted({r['group'] for r in rows})}
    summary = {'interpretation': 'Post-hoc sensitivity analysis; original pre-specified parser results remain authoritative.',
               'normalized_count': changed, 'status_counts': dict(Counter(r['status'] for r in rows)), 'groups': {}}
    for g, rs in groups.items():
        pos = [r for r in rs if r['gold_label'] != 'NOINFO']
        summary['groups'][g] = {'n':len(rs), 'accuracy':ev.accuracy(rs), 'class_recall':ev.class_recall(rs),
                                 'correct_link_positive':sum(ev.link_is_correct(r) for r in pos)/len(pos) if pos and g!='exp1_llm_only' else None}
    summary['paired_llm_to_rag3'] = ev.paired_changes(groups['exp1_llm_only'], groups['exp1_rag_k3'])
    summary['oracle_same_60'] = {g:ev.accuracy([r for r in groups[g] if r['gold_label'] != 'NOINFO']) for g in ('exp1_llm_only','exp1_rag_k3','exp1_oracle')}
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / 'sensitivity_summary.json', summary)
    with (out / 'sensitivity_results.jsonl').open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
