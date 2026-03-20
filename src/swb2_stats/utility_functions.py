from __future__ import annotations

import re

camel_pat = re.compile(r'([A-Z])')
under_pat = re.compile(r'_([a-z])')

def pause() -> None:
    programPause = input("Press the <ENTER> key to continue...")
     
def camel_to_underscore(name: str) -> str:
    return camel_pat.sub(lambda x: '_' + x.group(1).lower(), name)

def underscore_to_camel(name: str) -> str:
    return under_pat.sub(lambda x: x.group(1).upper(), name)

def underscore_to_kebab(name: str) -> str:
    return under_pat.sub(lambda x: '-' + x.group(1), name)
