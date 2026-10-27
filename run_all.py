from pathlib import Path
import runpy
ROOT=Path(__file__).resolve().parent
# Run each stage in this interpreter to avoid spawning parallel worker processes.
for script in ['generate_data.py','scripts/part_a.py','scripts/part_b.py','scripts/part_c.py','scripts/part_d.py','validate.py']:
    print('\n'+'='*18, 'RUNNING', script, '='*18, flush=True)
    runpy.run_path(str(ROOT/script),run_name='__main__')
