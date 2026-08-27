#!/usr/bin/env python3
"""用 golden 標註檔自動核對一組 agent log。

用法:
  python3 check_log.py <log_dir> [--case testNN] [--verbose]

<log_dir> 底下要有 test01.log ~ test91.log(或 .txt)。
每行結果三態:
  PASS   規則自動判定通過
  FAIL   規則自動判定失敗(必含數字/名稱缺失、yes/no 相反)
  REVIEW 無法自動判定(team_dependent / opt_qor / llm_judge / 語意模糊),需人工或 LLM 複核
"""
import re, os, sys, json

GOLDEN_DIR = os.path.dirname(os.path.abspath(__file__))
FAIL_WORDS = re.compile(r'\b(error|failed|cannot|unable|not supported|no such file)\b', re.I)
NEG_WORDS = re.compile(r'\b(no|none|not|zero|0|doesn\'t|does not|were no|are no)\b', re.I)
YES_WORDS = re.compile(r'\b(yes|exists?|found|there (is|are)|confirmed|equivalent|equivalence (check )?passed|verified|preserved)\b', re.I)
WORD_NUMS = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
             'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12,
             'a single': 1, 'single': 1, 'both': 2}


def parse_log(path):
    txt = open(path, errors='replace').read()
    return {int(n): b.strip() for n, b in re.findall(r'#RESPONSE (\d+)\n(.*?)#END \1', txt, re.S)}


def resp_numbers(s):
    s = re.sub(r'(\d),(\d)', r'\1\2', s)          # 1,234 -> 1234
    s = re.sub(r'\b[gnR]\d+\b', ' ', s)           # 去掉 g123/n45/R1 識別字
    s = re.sub(r'test\d+', ' ', s)
    s = re.sub(r'\[\d+(:\d+)?\]', ' ', s)         # 去掉位寬 [31:0]
    nums = {int(x) for x in re.findall(r'\d+', s)}
    low = s.lower()
    for w, v in WORD_NUMS.items():                # "Two BUF gates" 等英文數詞
        if re.search(rf'\b{w}\b', low):
            nums.add(v)
    return nums


def check_line(entry, resp):
    g = entry['golden']
    rule = g['rule']
    if resp is None:
        return 'FAIL', 'no #RESPONSE block for this line'
    if rule in ('ack', 'transform_ack'):
        if FAIL_WORDS.search(resp):
            return 'REVIEW', 'response contains failure wording'
        return 'PASS', ''
    if rule == 'exact_number':
        need = set(g.get('must_numbers') or [])
        have = resp_numbers(resp)
        missing = need - have
        # 備援:數字可能貼在識別字裡,做原文子字串檢查
        missing = {n for n in missing if not re.search(rf'\b{n}\b', resp.replace(',', ''))}
        if missing:
            # 0/1 常以文字隱含表達("a DFF instance"、"no gates"),交人工複核而非直接判錯
            if missing <= {0, 1}:
                return 'REVIEW', f'small numbers {sorted(missing)} not literal; likely phrased in words'
            return 'FAIL', f'missing numbers {sorted(missing)}'
        return 'PASS', ''
    if rule == 'name_set':
        need = g.get('must_names') or []
        missing = [x for x in need if not re.search(re.escape(x) + r'(?!\w)', resp)]
        if missing:
            if 'written to' in resp or 'result_page' in resp:
                return 'REVIEW', f'names {missing[:5]}... likely in external file'
            return 'FAIL', f'missing names {missing}'
        return 'PASS', ''
    if rule == 'yes_no':
        exp = (g.get('expected') or '').lower()
        clean = re.sub(r'\[\d+(:\d+)?\]', ' ', resp)      # 位寬/位索引的數字不參與語意判斷
        clean = re.sub(r'\b[gnR]\d+\b', ' ', clean)
        neg, yes = bool(NEG_WORDS.search(clean)), bool(YES_WORDS.search(clean))
        if exp == 'no':
            return ('PASS', '') if neg else ('REVIEW', 'expected NO; no negation wording found')
        if exp == 'yes':
            if yes or not neg:                            # 明確肯定或無否定的宣告句都算 yes
                return 'PASS', ''
            return 'REVIEW', 'expected YES; response contains negation wording'
        return 'REVIEW', 'no expected value in golden'
    # team_dependent / opt_qor / llm_judge
    return 'REVIEW', f'rule={rule}; ref: {g["reference"][:120]}'


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    verbose = '--verbose' in sys.argv
    only = None
    if '--case' in sys.argv:
        only = sys.argv[sys.argv.index('--case') + 1]
    if not args:
        print(__doc__)
        sys.exit(1)
    log_dir = args[0]
    golden = json.load(open(os.path.join(GOLDEN_DIR, 'golden_all.json')))

    tot = {'PASS': 0, 'FAIL': 0, 'REVIEW': 0}
    fails, reviews = [], []
    for case, data in sorted(golden.items()):
        if only and case != only:
            continue
        log = None
        for ext in ('.log', '.txt'):
            p = os.path.join(log_dir, case + ext)
            if os.path.exists(p):
                log = parse_log(p)
                break
        if log is None:
            print(f'{case}: LOG MISSING')
            continue
        for entry in data['lines']:
            verdict, why = check_line(entry, log.get(entry['line']))
            tot[verdict] += 1
            tag = f"{case} L{entry['line']} [{entry['golden']['rule']}]"
            if verdict == 'FAIL':
                fails.append(f'{tag} {why}')
            elif verdict == 'REVIEW':
                reviews.append(f'{tag} {why}')
            if verbose and verdict != 'PASS':
                print(f'{verdict:6} {tag} {why}')

    print(f"\n== summary: PASS {tot['PASS']} / FAIL {tot['FAIL']} / REVIEW {tot['REVIEW']}")
    if fails:
        print(f'\n-- FAIL ({len(fails)}):')
        for f in fails:
            print('  ', f)
    if reviews and not verbose:
        print(f'\n-- REVIEW ({len(reviews)}): rerun with --verbose to list')


if __name__ == '__main__':
    main()
