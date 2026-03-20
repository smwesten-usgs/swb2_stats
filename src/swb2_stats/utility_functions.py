import re

camel_pat = re.compile(r'([A-Z])')
under_pat = re.compile(r'_([a-z])')

def pause():
    programPause = input("Press the <ENTER> key to continue...")
     
def camel_to_underscore(name):
    return camel_pat.sub(lambda x: '_' + x.group(1).lower(), name)

def underscore_to_camel(name):
    return under_pat.sub(lambda x: x.group(1).upper(), name)

def underscore_to_kebab(name):
    return under_pat.sub(lambda x: '-' + x.group(1), name)
