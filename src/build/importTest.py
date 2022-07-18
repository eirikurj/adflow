#! /usr/bin/env python
import sys

name = "libadflow"
print("Testing if module %s can be imported..." % name)
import_cmd = "import %s" % name
try:
    exec(import_cmd)
except Exception as err:
    print("Error: libadflow was not imported correctly")
    raise err
    sys.exit(1)
# end try

print("\033[92mModule %s was successfully imported.\033[0m" % name)
