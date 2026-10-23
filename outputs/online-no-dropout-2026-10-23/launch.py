import json, subprocess
from pathlib import Path
subprocess.run(json.loads(Path(__file__).with_name("launch-command.json").read_text()), check=True)
