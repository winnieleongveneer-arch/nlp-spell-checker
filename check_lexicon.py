
import sys
sys.path.insert(0, "online")
import online.service as service

service.init_packs()
lex = service.POS_LEXICON
print(f"main: {lex.get('main')}")
print(f"trough: {lex.get('trough')}")
