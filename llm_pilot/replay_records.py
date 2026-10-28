"""Summarize the saved cloud pilot records without making any model/API calls."""
from collections import Counter
import json
from pathlib import Path
import re


def summarize(records):
    conditions = Counter(
        (row['task'], row.get('k'), row.get('m'), row.get('prev'),
         row.get('val'), row.get('ref')) for row in records
    )
    answers = []
    for row in records:
        match = re.search(r'COOPERATE|DEFECT|ACCEPT|REJECT', (row.get('text') or '').upper())
        answers.append(match.group(0) if match else None)
    return {
        'records': len(records),
        'tasks': dict(Counter(row['task'] for row in records)),
        'conditions': len(conditions),
        'repetitions_per_condition': sorted(set(conditions.values())),
        'parsed': sum(answer is not None for answer in answers),
        'answers': dict(Counter(answer for answer in answers if answer is not None)),
        'models': sorted(set(row['model'] for row in records)),
        'errors': sum(bool(row.get('err')) for row in records),
    }


if __name__ == '__main__':
    source = Path(__file__).with_name('llm_elicitation_responses.json')
    print(json.dumps(summarize(json.loads(source.read_text(encoding='utf-8'))), indent=2))
