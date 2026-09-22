import warnings, logging, sys
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
import mteb, pandas as pd
from mteb.cache import ResultCache
S="/tmp/claude-0/-home-user-qwen-38-agent-setup/370f9318-8560-5ca7-b874-d65868720d0a/scratchpad"
cache = ResultCache(cache_path=S+"/results")
bench = mteb.get_benchmark(sys.argv[1] if len(sys.argv)>1 else "MTEB(Multilingual, v2)")
res = cache.load_results(tasks=bench, include_remote=False, validate_and_filter=True)
df = res.get_benchmark_result()
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 100)
print(df.to_string())
