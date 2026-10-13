"""Stand-in for espeak-ng in tests: same CLI shape, writes a silent WAV.

Duration is 0.05 s per character so longer messages give longer clips.
Text "FAIL" makes it exit non-zero; "EMPTY" makes it write no file.
"""

import sys
import wave

args = sys.argv[1:]
out = args[args.index("-w") + 1]
text = args[-1]
if text == "FAIL":
    sys.stderr.write("simulated engine failure\n")
    sys.exit(2)
if text != "EMPTY":
    rate = 22050
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * 0.05 * len(text)))
