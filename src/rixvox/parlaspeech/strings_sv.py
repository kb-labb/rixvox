# Normalize variations of abbreviations

# These aren't escaped in code, so they need to be escaped here
ocr_corrections = {
    "\$": "§",
    "bl\.a ": "bl.a.",
    "[D|d]\.v\.s ": "d.v.s. ",
    "[D|d]\. v\.s.": "d.v.s.",
    "[F|f]r\.o\.m ": "fr.o.m.",
    "[K|k]ungl\. maj\: t": "kungl. maj:t",
    "m\. m\.": "m.m.",
    "m\.m ": "m.m. ",
    "m\. fl\.": "m.fl.",
    "milj\. kr.": "milj.kr.",
    "o\. s\.v\.": "o.s.v.",
    "s\. k\.": "s.k.",
    "t\. ex\.": "t.ex.",
    "t\.o\.m,": "t.o.m.",
    "t\.o\. m\.": "t.o.m.",
}

# Escaped in code, so they don't need to be escaped here
abbreviations = {
    "bl.a.": "bland annat",
    "d.v.s.": "det vill säga",
    "dvs.": "det vill säga",
    "e.d.": "eller dylikt",
    "f.d.": "före detta",
    "f.n.": "för närvarande",
    "f.ö.": "för övrigt",
    "fr.o.m.": "från och med",
    "inkl.": "inklusive",
    "k.m:t": "kunglig majestät",
    "kungl. maj:t": "kunglig majestät",
    "m.a.o.": "med andra ord",
    "milj.kr.": "miljoner kronor",
    "m.m.": "med mera",
    "m.fl.": "med flera",
    "o.d.": "och dylikt",
    "o.dyl.": "och dylikt",
    "o.s.v.": "och så vidare",
    "osv.": "och så vidare",
    "p.g.a.": "på grund av",
    "s.a.s.": "så att säga",
    "s.k.": "så kallad",
    "t.o.m.": "till och med",
    "t.ex.": "till exempel",
    "t.v.": "tills vidare",
    "kungl.": "kungliga",
}

symbols = {
    "%": "procent",
}
