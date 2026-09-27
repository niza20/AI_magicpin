"""Generates the original flat illustrations used by the /demo chat (demo_assets/photos/*.svg).

    python scripts/make_demo_art.py

Each file is a 720x400 SVG of one subject (thali, pizza, dental care, gym...). Real photos dropped into
demo_assets/photos/ with the same key (e.g. pizza.jpg) are picked before these.
"""
from __future__ import annotations

import math
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "demo_assets" / "photos"
W, H = 720, 400


def svg(bg1: str, bg2: str, body: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">
<defs>
 <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{bg1}"/><stop offset="1" stop-color="{bg2}"/></linearGradient>
 <radialGradient id="glow" cx=".5" cy=".45" r=".6"><stop offset="0" stop-color="#fff" stop-opacity=".28"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></radialGradient>
 <filter id="sh" x="-20%" y="-20%" width="140%" height="140%"><feDropShadow dx="0" dy="8" stdDeviation="9" flood-color="#000" flood-opacity=".25"/></filter>
</defs>
<rect width="{W}" height="{H}" fill="url(#bg)"/><rect width="{W}" height="{H}" fill="url(#glow)"/>
{body}
</svg>
"""


def bowl(x, y, r, fill, rim="#f4efe6", inner=None):
    s = f'<circle cx="{x}" cy="{y}" r="{r}" fill="{rim}" filter="url(#sh)"/><circle cx="{x}" cy="{y}" r="{r * .8:.1f}" fill="{fill}"/>'
    if inner:
        s += inner
    return s


def dots(cx, cy, rx, ry, n, color, r=3, seed=1):
    out = []
    for i in range(n):
        a = (i * 137.508 + seed * 31) % 360
        d = math.sqrt(((i * 0.618 + seed) % 1))
        out.append(f'<circle cx="{cx + math.cos(math.radians(a)) * rx * d:.1f}" cy="{cy + math.sin(math.radians(a)) * ry * d:.1f}" r="{r}" fill="{color}"/>')
    return "".join(out)


def south_indian():
    leaf = ('<g filter="url(#sh)"><path d="M70 330 C120 90 560 40 660 120 C640 300 260 380 70 330Z" fill="#2e8b3a"/>'
            '<path d="M70 330 C250 250 470 160 660 120" stroke="#9fd48a" stroke-width="5" fill="none"/>'
            + "".join(f'<path d="M{150 + i * 60} {300 - i * 22} l{-10 - i} {-60 + i * 4}" stroke="#57a84d" stroke-width="2"/>' for i in range(8))
            + '</g>')
    rice = '<ellipse cx="350" cy="235" rx="95" ry="55" fill="#fbf8ef"/>' + dots(350, 235, 85, 45, 70, "#e9e3d2", 2.2, 3)
    katoris = (bowl(200, 175, 42, "#d9822b") + bowl(290, 120, 36, "#b8321f") + bowl(420, 115, 38, "#f6f1e3")
               + bowl(520, 165, 40, "#8fb339", inner=dots(520, 165, 26, 26, 18, "#e3c24b", 3, 2))
               + bowl(560, 255, 30, "#c0392b"))
    papad = '<circle cx="180" cy="280" r="46" fill="#f2d49b" filter="url(#sh)"/>' + dots(180, 280, 38, 38, 22, "#c9a063", 2.5, 5)
    return svg("#f6e7c8", "#e3b36b", leaf + rice + katoris + papad)


def dosa():
    plate = '<ellipse cx="360" cy="215" rx="300" ry="150" fill="#f7f4ee" filter="url(#sh)"/><ellipse cx="360" cy="215" rx="270" ry="130" fill="#fff"/>'
    roll = ('<g transform="rotate(-12 360 200)"><rect x="110" y="160" width="430" height="80" rx="40" fill="#d98a2b"/>'
            '<rect x="110" y="160" width="430" height="34" rx="17" fill="#e8a94a"/>'
            + "".join(f'<path d="M{150 + i * 45} 170 q10 25 0 60" stroke="#b86e1c" stroke-width="3" fill="none" opacity=".6"/>' for i in range(8))
            + '<ellipse cx="540" cy="200" rx="22" ry="40" fill="#f5c26b"/><ellipse cx="540" cy="200" rx="12" ry="26" fill="#e8b04f"/></g>')
    chut = (bowl(190, 315, 34, "#f4f1e4") + bowl(280, 330, 30, "#c43d27") + bowl(560, 315, 40, "#cf7a2a",
            inner=dots(560, 315, 22, 22, 8, "#f2b233", 4, 4)))
    return svg("#2f5d50", "#16362e", plate + roll + chut)


def pizza():
    board = '<circle cx="360" cy="205" r="175" fill="#9a6a3a" filter="url(#sh)"/><rect x="520" y="180" width="160" height="50" rx="22" fill="#9a6a3a"/>'
    base = '<circle cx="360" cy="205" r="150" fill="#e0a55a"/><circle cx="360" cy="205" r="128" fill="#c8402b"/>'
    cheese = "".join(f'<circle cx="{360 + math.cos(a) * d:.0f}" cy="{205 + math.sin(a) * d:.0f}" r="{22 - (i % 3) * 4}" fill="#f7d774" opacity=".95"/>'
                     for i, (a, d) in enumerate((math.radians(i * 47), 30 + (i * 23) % 85) for i in range(26)))
    pep = "".join(f'<circle cx="{360 + math.cos(math.radians(i * 72 + 20)) * 80:.0f}" cy="{205 + math.sin(math.radians(i * 72 + 20)) * 80:.0f}" r="16" fill="#9e2a1c"/>' for i in range(5))
    pep += '<circle cx="360" cy="205" r="16" fill="#9e2a1c"/>'
    basil = "".join(f'<ellipse cx="{360 + math.cos(math.radians(i * 72 + 55)) * 45:.0f}" cy="{205 + math.sin(math.radians(i * 72 + 55)) * 45:.0f}" rx="12" ry="6" fill="#2f8a3b" transform="rotate({i * 40} {360 + math.cos(math.radians(i * 72 + 55)) * 45:.0f} {205 + math.sin(math.radians(i * 72 + 55)) * 45:.0f})"/>' for i in range(5))
    cuts = "".join(f'<line x1="360" y1="205" x2="{360 + math.cos(math.radians(i * 60)) * 150:.0f}" y2="{205 + math.sin(math.radians(i * 60)) * 150:.0f}" stroke="#8a5a2b" stroke-width="3" opacity=".5"/>' for i in range(6))
    return svg("#3a2a22", "#1f1612", board + base + cheese + pep + basil + cuts)


def chai():
    saucer = '<ellipse cx="330" cy="310" rx="170" ry="40" fill="#f3e9da" filter="url(#sh)"/><ellipse cx="330" cy="305" rx="140" ry="28" fill="#e7dac5"/>'
    cup = ('<path d="M230 160 L430 160 L405 300 Q330 322 255 300Z" fill="#b5532b" filter="url(#sh)"/>'
           '<path d="M250 205 L412 205" stroke="#8f3d1d" stroke-width="4"/><ellipse cx="330" cy="160" rx="100" ry="22" fill="#c86a3c"/>'
           '<ellipse cx="330" cy="162" rx="88" ry="16" fill="#b9814a"/><ellipse cx="315" cy="160" rx="30" ry="5" fill="#d8a878" opacity=".7"/>')
    steam = "".join(f'<path d="M{295 + i * 35} 135 q-18 -30 0 -55 q18 -25 0 -50" stroke="#fff" stroke-width="7" fill="none" stroke-linecap="round" opacity=".55"/>' for i in range(3))
    spice = ('<rect x="500" y="250" width="130" height="18" rx="9" fill="#8b4a22" transform="rotate(-10 565 259)"/>'
             '<rect x="505" y="275" width="120" height="16" rx="8" fill="#a05a2c" transform="rotate(-6 565 283)"/>'
             '<ellipse cx="120" cy="300" rx="16" ry="10" fill="#7a9a3a"/><ellipse cx="150" cy="315" rx="16" ry="10" fill="#6d8c32"/>'
             '<rect x="530" y="130" width="120" height="70" rx="10" fill="#e8c27a" filter="url(#sh)"/>'
             + dots(590, 165, 45, 22, 12, "#c99a4d", 3, 6))
    return svg("#f0c987", "#c9793d", saucer + cup + steam + spice)


def _plate(extra_bg=""):
    return '<ellipse cx="360" cy="215" rx="290" ry="150" fill="#1d1d1d" filter="url(#sh)"/><ellipse cx="360" cy="215" rx="265" ry="130" fill="#2b2b2b"/>' + extra_bg


def _garnish():
    onions = "".join(f'<ellipse cx="{150 + i * 28}" cy="{300 - (i % 2) * 12}" rx="26" ry="14" fill="none" stroke="#e7a7c4" stroke-width="6"/>' for i in range(4))
    lemon = '<path d="M540 280 a42 42 0 0 1 84 0Z" fill="#f4e04d"/><path d="M548 280 a34 34 0 0 1 68 0Z" fill="#fbf1a0"/>'
    mint = bowl(560, 150, 36, "#3f9b4f")
    return onions + lemon + mint


def kebab():
    ke = ""
    for i in range(4):
        y = 150 + i * 38
        ke += (f'<g transform="rotate(-8 360 {y})"><rect x="150" y="{y - 6}" width="400" height="6" fill="#b9b9b9"/>'
               f'<rect x="185" y="{y - 20}" width="320" height="34" rx="17" fill="#7a3a17"/>'
               + "".join(f'<rect x="{205 + k * 38}" y="{y - 18}" width="8" height="30" rx="4" fill="#3d1a08" opacity=".7"/>' for k in range(8))
               + '</g>')
    return svg("#e9b06a", "#b8561f", _plate() + ke + _garnish())


def tandoori():
    legs = ""
    for i, (x, y, r) in enumerate(((260, 190, -25), (380, 170, 10), (330, 260, -5), (460, 245, 25))):
        legs += (f'<g transform="rotate({r} {x} {y})"><ellipse cx="{x}" cy="{y}" rx="70" ry="42" fill="#c43a1a"/>'
                 f'<ellipse cx="{x - 15}" cy="{y - 12}" rx="40" ry="16" fill="#e2622f" opacity=".8"/>'
                 f'<rect x="{x + 55}" y="{y - 8}" width="50" height="16" rx="8" fill="#f1e2c9"/>'
                 f'<circle cx="{x + 105}" cy="{y - 6}" r="11" fill="#f7eedd"/><circle cx="{x + 105}" cy="{y + 7}" r="11" fill="#f7eedd"/>'
                 + dots(x, y, 50, 28, 10, "#6b1a0a", 3, i + 1) + '</g>')
    return svg("#f1b56a", "#b4401c", _plate() + legs + _garnish())


def biryani():
    handi = ('<ellipse cx="360" cy="330" rx="230" ry="40" fill="#000" opacity=".2"/>'
             '<path d="M140 190 Q140 350 360 350 Q580 350 580 190Z" fill="#8a5a2b" filter="url(#sh)"/>'
             '<ellipse cx="360" cy="190" rx="225" ry="60" fill="#6e4520"/><ellipse cx="360" cy="190" rx="200" ry="48" fill="#f3d27a"/>')
    rice = dots(360, 190, 190, 44, 160, "#fff6dc", 2.5, 7) + dots(360, 190, 180, 40, 70, "#e89a2c", 2.5, 9)
    top = ('<ellipse cx="300" cy="180" rx="30" ry="22" fill="#fff"/><circle cx="300" cy="180" r="11" fill="#f5b700"/>'
           '<ellipse cx="420" cy="195" rx="30" ry="22" fill="#fff"/><circle cx="420" cy="195" r="11" fill="#f5b700"/>'
           + dots(360, 185, 150, 32, 30, "#8a4a1c", 4, 11) + dots(360, 185, 120, 28, 10, "#2f8a3b", 5, 13))
    return svg("#d9a441", "#7a3f12", handi + rice + top)


def burger():
    b = ('<ellipse cx="360" cy="340" rx="200" ry="26" fill="#000" opacity=".2"/>'
         '<rect x="190" y="280" width="340" height="50" rx="22" fill="#d9912f"/>'
         '<path d="M180 262 q30 26 60 0 q30 26 60 0 q30 26 60 0 q30 26 60 0 q30 26 60 0 q30 26 60 0 L540 280 L180 280Z" fill="#f2c22e"/>'
         '<rect x="185" y="225" width="350" height="45" rx="20" fill="#5a2e14"/>'
         '<rect x="195" y="210" width="330" height="20" rx="10" fill="#d8342a"/>'
         '<path d="M175 205 q25 -22 50 0 q25 -22 50 0 q25 -22 50 0 q25 -22 50 0 q25 -22 50 0 q25 -22 50 0 q25 -22 50 0" stroke="#58b045" stroke-width="16" fill="none" stroke-linecap="round"/>'
         '<path d="M190 200 Q360 40 530 200Z" fill="#e3a23e" filter="url(#sh)"/>'
         + dots(360, 140, 120, 40, 22, "#fff4d6", 4, 3))
    return svg("#ffcf6b", "#ef7b2a", b)


def restaurant():
    table = '<rect x="0" y="250" width="720" height="150" fill="#6b4226"/><rect x="0" y="250" width="720" height="10" fill="#80512f"/>'
    d = (bowl(200, 250, 70, "#c9651f", inner=dots(200, 250, 40, 40, 10, "#f4d35e", 5, 2))
         + bowl(380, 230, 80, "#8b2e14", inner=dots(380, 230, 50, 50, 14, "#f2e8cf", 6, 4))
         + '<ellipse cx="560" cy="260" rx="95" ry="55" fill="#e8b765" filter="url(#sh)"/>' + dots(560, 260, 70, 38, 16, "#b9772d", 5, 8)
         + '<rect x="620" y="110" width="50" height="110" rx="10" fill="#dff2ff" opacity=".85"/><rect x="620" y="150" width="50" height="70" rx="8" fill="#f4b73f"/>')
    lights = "".join(f'<circle cx="{80 + i * 140}" cy="60" r="18" fill="#ffd98a" opacity=".8"/><line x1="{80 + i * 140}" y1="0" x2="{80 + i * 140}" y2="42" stroke="#3a2a1a" stroke-width="3"/>' for i in range(5))
    return svg("#3b2a20", "#1b120d", lights + table + d)


def dental():
    tooth = ('<path d="M290 110 C230 110 220 190 250 250 C265 280 270 330 290 330 C315 330 305 270 330 270 C355 270 345 330 370 330 '
             'C390 330 395 280 410 250 C440 190 430 110 370 110 C345 110 340 125 330 125 C320 125 315 110 290 110Z" fill="#fff" filter="url(#sh)"/>'
             '<path d="M270 150 C265 175 268 195 278 210" stroke="#dff3ff" stroke-width="12" fill="none" stroke-linecap="round"/>')
    brush = ('<g transform="rotate(-30 560 230)"><rect x="470" y="220" width="220" height="22" rx="11" fill="#1e88e5"/>'
             '<rect x="470" y="190" width="70" height="30" rx="6" fill="#e3f2fd"/>'
             + "".join(f'<rect x="{474 + k * 8}" y="178" width="5" height="16" fill="#90caf9"/>' for k in range(8)) + '</g>')
    spark = "".join(f'<path d="M{x} {y - 16} L{x + 5} {y - 5} L{x + 16} {y} L{x + 5} {y + 5} L{x} {y + 16} L{x - 5} {y + 5} L{x - 16} {y} L{x - 5} {y - 5}Z" fill="#fff"/>'
                    for x, y in ((200, 110), (460, 90), (170, 290)))
    return svg("#7fd3c8", "#2a8c9e", tooth + brush + spark)


def haircut():
    mirror = '<ellipse cx="250" cy="200" rx="140" ry="170" fill="#e9e1f2" filter="url(#sh)"/><ellipse cx="250" cy="200" rx="118" ry="148" fill="#b8a7d6"/>'
    head = ('<circle cx="250" cy="185" r="55" fill="#e2a47c"/><path d="M190 190 Q185 100 250 105 Q320 100 312 190 Q300 140 250 140 Q200 140 190 190Z" fill="#3a2418"/>'
            '<path d="M180 330 Q250 250 320 330Z" fill="#6c5ba7"/>')
    scis = ('<g transform="rotate(-35 530 200)"><circle cx="470" cy="170" r="26" fill="none" stroke="#333" stroke-width="10"/>'
            '<circle cx="470" cy="240" r="26" fill="none" stroke="#333" stroke-width="10"/>'
            '<path d="M492 180 L640 225 L495 205Z" fill="#cfd8dc"/><path d="M492 230 L640 212 L495 215Z" fill="#b0bec5"/>'
            '<circle cx="505" cy="208" r="6" fill="#555"/></g>')
    comb = '<rect x="460" y="300" width="200" height="26" rx="6" fill="#ec407a"/>' + "".join(f'<rect x="{468 + k * 12}" y="326" width="6" height="26" fill="#ec407a"/>' for k in range(16))
    return svg("#f7d9e3", "#c77ca1", mirror + head + scis + comb)


def spa():
    stones = ('<ellipse cx="300" cy="320" rx="130" ry="36" fill="#5d6a66" filter="url(#sh)"/><ellipse cx="300" cy="265" rx="105" ry="32" fill="#76847f"/>'
              '<ellipse cx="300" cy="215" rx="80" ry="27" fill="#8e9c97"/><ellipse cx="300" cy="172" rx="56" ry="21" fill="#a8b5b0"/>')
    candle = ('<rect x="470" y="220" width="70" height="100" rx="8" fill="#fff4e0" filter="url(#sh)"/><line x1="505" y1="220" x2="505" y2="205" stroke="#333" stroke-width="3"/>'
              '<path d="M505 170 Q520 195 505 207 Q490 195 505 170Z" fill="#ffb300"/>')
    towel = '<rect x="560" y="270" width="120" height="50" rx="25" fill="#f7f1ea" filter="url(#sh)"/><circle cx="585" cy="295" r="18" fill="#efe6db"/>'
    leaves = "".join(f'<ellipse cx="{130 + i * 18}" cy="{150 + i * 30}" rx="40" ry="12" fill="#4caf50" transform="rotate({-30 + i * 15} {130 + i * 18} {150 + i * 30})"/>' for i in range(5))
    orchid = "".join(f'<ellipse cx="{430 + math.cos(math.radians(a)) * 18:.0f}" cy="{110 + math.sin(math.radians(a)) * 18:.0f}" rx="16" ry="10" fill="#f8bbd0" transform="rotate({a} {430 + math.cos(math.radians(a)) * 18:.0f} {110 + math.sin(math.radians(a)) * 18:.0f})"/>' for a in range(0, 360, 72)) + '<circle cx="430" cy="110" r="8" fill="#e91e63"/>'
    return svg("#cfe8dc", "#7fb7a1", stones + candle + towel + leaves + orchid)


def salon():
    mirror = '<rect x="120" y="50" width="260" height="230" rx="20" fill="#f4e9ef" filter="url(#sh)"/><rect x="140" y="70" width="220" height="190" rx="12" fill="#c9d6e8"/>'
    bulbs = "".join(f'<circle cx="{140 + k * 44}" cy="60" r="9" fill="#fff3c4"/>' for k in range(6))
    chair = ('<rect x="170" y="290" width="160" height="30" rx="12" fill="#2d2d2d"/><rect x="185" y="200" width="130" height="95" rx="18" fill="#3d3d3d"/>'
             '<rect x="245" y="320" width="12" height="40" fill="#888"/><ellipse cx="251" cy="365" rx="60" ry="10" fill="#666"/>')
    dryer = ('<g transform="rotate(-15 540 200)"><rect x="470" y="160" width="150" height="70" rx="35" fill="#e91e63" filter="url(#sh)"/>'
             '<circle cx="605" cy="195" r="30" fill="#ad1457"/><rect x="495" y="220" width="36" height="110" rx="14" fill="#c2185b"/></g>')
    return svg("#fde2ea", "#e89ab5", mirror + bulbs + chair + dryer)


def yoga():
    mat = '<path d="M140 320 L600 320 L640 350 L100 350Z" fill="#7e57c2"/>'
    sun = '<circle cx="560" cy="110" r="55" fill="#ffd54f" opacity=".85"/>'
    fig = ('<g fill="#3e2a5c"><circle cx="360" cy="95" r="24"/>'
           '<path d="M345 120 L375 120 L372 220 L348 220Z"/>'
           '<path d="M350 128 L300 60" stroke="#3e2a5c" stroke-width="14" stroke-linecap="round"/>'
           '<path d="M370 128 L420 60" stroke="#3e2a5c" stroke-width="14" stroke-linecap="round"/>'
           '<path d="M355 215 L355 320" stroke="#3e2a5c" stroke-width="16" stroke-linecap="round"/>'
           '<path d="M367 215 L410 255 L360 265" stroke="#3e2a5c" stroke-width="15" fill="none" stroke-linecap="round" stroke-linejoin="round"/></g>')
    leaves = "".join(f'<ellipse cx="{110 + i * 16}" cy="{130 + i * 38}" rx="36" ry="11" fill="#66bb6a" transform="rotate({-25 + i * 12} {110 + i * 16} {130 + i * 38})"/>' for i in range(5))
    return svg("#ffe0b2", "#f48fb1", sun + leaves + mat + fig)


def gym():
    def db(x, y, rot, col):
        return (f'<g transform="rotate({rot} {x} {y})" filter="url(#sh)"><rect x="{x - 90}" y="{y - 9}" width="180" height="18" rx="6" fill="#9e9e9e"/>'
                f'<rect x="{x - 110}" y="{y - 45}" width="34" height="90" rx="8" fill="{col}"/><rect x="{x - 75}" y="{y - 35}" width="18" height="70" rx="6" fill="{col}"/>'
                f'<rect x="{x + 76}" y="{y - 45}" width="34" height="90" rx="8" fill="{col}"/><rect x="{x + 57}" y="{y - 35}" width="18" height="70" rx="6" fill="{col}"/></g>')
    kb = ('<g filter="url(#sh)"><path d="M510 150 Q510 95 560 95 Q610 95 610 150" stroke="#263238" stroke-width="22" fill="none"/>'
          '<circle cx="560" cy="235" r="92" fill="#263238"/><circle cx="535" cy="210" r="22" fill="#37474f"/>'
          '<text x="560" y="255" font-family="Arial, sans-serif" font-size="40" font-weight="700" fill="#ff7043" text-anchor="middle">16</text></g>')
    floor = '<rect x="0" y="330" width="720" height="70" fill="#1a1a1a"/>'
    return svg("#455a64", "#1c262b", floor + db(230, 300, -8, "#ff7043") + db(250, 190, 6, "#29b6f6") + kb)


def pharmacy():
    cross = ('<g filter="url(#sh)"><rect x="95" y="80" width="170" height="170" rx="22" fill="#fff"/>'
             '<rect x="155" y="100" width="50" height="130" rx="8" fill="#2e9d57"/><rect x="115" y="140" width="130" height="50" rx="8" fill="#2e9d57"/></g>')
    bottle = ('<g filter="url(#sh)"><rect x="330" y="120" width="120" height="30" rx="6" fill="#ffffff"/><rect x="320" y="150" width="140" height="190" rx="18" fill="#ff8f00"/>'
              '<rect x="335" y="200" width="110" height="80" rx="6" fill="#fff8e1"/><rect x="350" y="220" width="80" height="10" rx="5" fill="#90a4ae"/>'
              '<rect x="350" y="240" width="60" height="10" rx="5" fill="#b0bec5"/></g>')
    caps = "".join(f'<g transform="rotate({r} {x} {y})"><rect x="{x - 36}" y="{y - 14}" width="72" height="28" rx="14" fill="#fff"/><rect x="{x - 36}" y="{y - 14}" width="36" height="28" rx="14" fill="{c}"/></g>'
                   for x, y, r, c in ((530, 300, 20, "#e53935"), (600, 250, -30, "#1e88e5"), (560, 180, 45, "#43a047"), (640, 330, 5, "#8e24aa")))
    blister = '<rect x="500" y="60" width="170" height="90" rx="10" fill="#cfd8dc" filter="url(#sh)"/>' + "".join(f'<circle cx="{525 + (k % 5) * 30}" cy="{85 + (k // 5) * 38}" r="11" fill="#eceff1" stroke="#b0bec5"/>' for k in range(10))
    return svg("#d6f5e3", "#6cc198", cross + bottle + caps + blister)


ART = {"south_indian": south_indian, "dosa": dosa, "pizza": pizza, "chai": chai, "kebab": kebab, "tandoori": tandoori,
       "biryani": biryani, "burger": burger, "restaurant": restaurant, "dental": dental, "haircut": haircut, "spa": spa,
       "salon": salon, "yoga": yoga, "gym": gym, "pharmacy": pharmacy}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for key, fn in ART.items():
        (OUT / f"{key}.svg").write_text(fn(), encoding="utf-8")
    print(f"wrote {len(ART)} illustrations to {OUT}")
