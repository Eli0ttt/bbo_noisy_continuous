#!/usr/bin/env python3
"""Install AST-validated candidate extraction without changing EvoMAS operators."""
from pathlib import Path
import ast
import hashlib

adapter = Path(__file__).resolve().parent
runtime = adapter / 'native_bbo_runtime.py'
replacement = (adapter / 'candidate_extractor_v2.py').read_text(encoding='utf-8')
replacement = replacement[replacement.index('def extract_candidate('):].strip()
source = runtime.read_text(encoding='utf-8')
start = source.index('def extract_candidate(')
end = source.index('\n\ndef validate(', start)
updated = source[:start] + replacement + source[end:]
if 'import ast\n' not in updated:
    anchor = 'import argparse\n'
    if anchor not in updated:
        raise SystemExit('Cannot insert ast import safely')
    updated = updated.replace(anchor, anchor + 'import ast\n', 1)
compile(updated, str(runtime), 'exec')
if updated != source:
    backup = runtime.with_name('native_bbo_runtime.py.before_ast_candidate_extraction')
    if not backup.exists():
        backup.write_text(source, encoding='utf-8')
    runtime.write_text(updated, encoding='utf-8')
print('[EXTRACTOR PATCH PASS]',hashlib.sha256(updated.encode()).hexdigest())
