import json
import logging

logging.basicConfig(level=logging.INFO, format='%(name)s | %(message)s')

from src.orchestrator import Orchestrator

memo = Orchestrator(
    'Apple Inc',
    ticker='AAPL',
    on_agent_start=lambda name: print(f'\n▶  Starting: {name}'),
    on_agent_complete=lambda name, r: print(f'✓  Done: {name} — {len(r.findings)} findings, confidence={r.confidence_score:.2f}'),
    on_synthesis_start=lambda: print('\n⚙  Synthesizing...'),
).run()

print('\n' + '='*60)
print(f'Company: {memo.company_name}')
print(f'Overall confidence: {memo.overall_confidence:.2f}')
print(f'Sections: {[s.title for s in memo.sections]}')
print()
print('EXECUTIVE SUMMARY:')
print(memo.executive_summary[:800])

with open('memo_apple.json', 'w') as f:
    json.dump(memo.model_dump(mode='json'), f, indent=2)
print('\nFull memo saved → memo_apple.json')
