"""Generates samples/are_re_are_yeh_kya_hua.ass (karaoke word timing)."""
import os
L = [("Are re are yeh kya hua", [0.00, 1.00, 1.75, 2.00, 2.50, 2.85], 3.25),
     ("Maine na yeh jaana", [3.25, 3.75, 4.25, 4.75], 6.30),
     ("Are re are ban jaaye na", [6.75, 7.25, 7.50, 8.00, 8.25, 8.75], 9.60),
     ("kahin koi afsaana", [9.75, 10.50, 10.75], 12.40),
     ("Are re are kuch ho gaya", [12.50, 13.40, 14.00, 14.50, 14.75, 15.00], 16.20),
     ("koi na pehchaana", [16.25, 17.00, 17.50], 19.10),
     ("Are re are banta hai to", [19.25, 19.75, 20.25, 21.00, 21.75, 22.25], 22.45),
     ("ban jaaye afsaana", [22.50, 23.25, 23.75], 25.60),
     ("la la la la la", [26.00, 26.50, 26.75, 27.00, 27.25], 29.20)]


def ts(x):
    return f"{int(x // 3600)}:{int(x % 3600 // 60):02d}:{x % 60:05.2f}"


out = ["[Script Info]", "ScriptType: v4.00+", "PlayResX: 1080", "PlayResY: 1920", "", "[V4+ Styles]",
       "Format: Name, Fontname, Fontsize, PrimaryColour", "Style: Default,Poppins,60,&H00FFFFFF", "",
       "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
for text, starts, end in L:
    ws = text.split()
    assert len(ws) == len(starts), text
    parts = []
    for i, w in enumerate(ws):
        nxt = starts[i + 1] if i + 1 < len(starts) else end
        parts.append("{\\k%d}%s" % (round((nxt - starts[i]) * 100), w))
    out.append(f"Dialogue: 0,{ts(starts[0])},{ts(end)},Default,,0,0,0,," + " ".join(parts))
here = os.path.dirname(os.path.abspath(__file__))
open(os.path.join(here, "are_re_are_yeh_kya_hua.ass"), "w", encoding="utf-8").write("\n".join(out) + "\n")
