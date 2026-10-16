"""Extract the GWTC-4.0 fiducial BBH population hyperparameters from the LVK paper source.

Downloads the arXiv source of the GWTC-4.0 population paper (arXiv:2508.18083), parses the
machine-generated macro ``\\defaultbbh`` (posterior summaries of the Broken Power Law + 2 Peaks
model) and writes the medians and 90 per cent intervals to JSON.
"""
import argparse, io, json, re, tarfile, urllib.request
from pathlib import Path

URL = 'https://arxiv.org/e-print/2508.18083'


def parse_macro(text, name='defaultbbh'):
    pattern = r'newcommand\\' + name + r'\[1\]\[all\]\{\\ifnum\\pdfstrcmp\{#1\}\{all\}=0\\def\\' + name + r'@out\{(.*?)\}\\else'
    body = re.search(pattern, text, re.S).group(1)
    body = body.replace('\\{', '{').replace('\\}', '}').replace('\\_', '_').replace('{-}', '-').replace('\\', '')
    return json.loads(body)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=Path('realdata/outputs/gwtc4_bp2p_summary.json'))
    a = p.parse_args()
    raw = urllib.request.urlopen(URL, timeout=120).read()
    with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
        text = tar.extractfile('macros_from_json.tex').read().decode()
    model = parse_macro(text)
    summary = {k: {'median': float(v['median']), 'p05': float(v['5th percentile']), 'p95': float(v['95th percentile'])}
               for k, v in model.items() if isinstance(v, dict) and 'median' in v}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({'source': URL, 'model': 'BP2P (fiducial BBH)', 'hyperparameters': summary}, indent=2))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
