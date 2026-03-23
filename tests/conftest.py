from pathlib import Path
import sys


ROOT = Path("/Users/jqwang/31-allinone/src")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
