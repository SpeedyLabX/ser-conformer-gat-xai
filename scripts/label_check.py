#!/usr/bin/env python3
import json, collections, sys
from pathlib import Path
p = Path('data/iemocap_manifest.jsonl')
if not p.exists():
    print('MANIFEST NOT FOUND:', p)
    sys.exit(1)
canon = set(['neu','hap','ang','sad','exc','fru'])
label_cnt = collections.Counter()
sess_label_presence = {}
noncanon_examples = collections.defaultdict(list)
with p.open('r', encoding='utf-8') as fh:
    for i,line in enumerate(fh):
        try:
            r = json.loads(line)
        except Exception as e:
            print('JSON parse error on line', i+1, e)
            continue
        lab = r.get('label')
        sess = r.get('session') or 'UNK'
        label_cnt['__NONE__' if lab is None else lab] += 1
        sess_label_presence.setdefault(sess, {'labeled':0,'unlabeled':0})
        if lab is None:
            sess_label_presence[sess]['unlabeled'] += 1
        else:
            sess_label_presence[sess]['labeled'] += 1
            if lab not in canon:
                if len(noncanon_examples[lab])<5:
                    noncanon_examples[lab].append({ 'utterance_id': r.get('utterance_id'), 'text': r.get('text') })

print('Total records:', sum(label_cnt.values()))
print('\nOverall label counts (showing top 50):')
for k,v in label_cnt.most_common(50):
    print(f'  {k}: {v}')
print('\nSessions (labeled / unlabeled):')
for s in sorted(sess_label_presence.keys()):
    d = sess_label_presence[s]
    print(f'  {s}: labeled={d["labeled"]}, unlabeled={d["unlabeled"]}')

if noncanon_examples:
    print('\nNon-canonical labels found (examples):')
    for lab, exs in noncanon_examples.items():
        print(' ', lab, 'count_sample_examples=', len(exs))
        for e in exs:
            print('    -', e['utterance_id'], '|', (e['text'] or '')[:120])
else:
    print('\nNo non-canonical labels found.')
