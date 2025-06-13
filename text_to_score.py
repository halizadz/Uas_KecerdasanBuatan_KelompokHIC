# File: text_to_score.py

def text_to_score(text, category):
    text = text.lower()

    keyword_scores = {
        "scope": [
            ("berbagai sektor", 0.9),
            ("multisektor", 0.8),
            ("industri dan sosial", 0.85),
            ("terbatas", 0.3),
            ("spesifik saja", 0.4)
        ],
        "prospects": [
            ("pertumbuhan tinggi", 0.9),
            ("prospek besar", 0.85),
            ("permintaan tinggi", 0.8),
            ("stagnan", 0.3),
            ("menurun", 0.2)
        ],
        "potential": [
            ("mengubah struktur", 0.9),
            ("disruptif", 0.85),
            ("potensi besar", 0.8),
            ("tidak berdampak", 0.3),
            ("rendah", 0.2)
        ],
        "economy": [
            ("kontribusi perekonomian", 0.85),
            ("profit tinggi", 0.8),
            ("menguntungkan", 0.75),
            ("tidak signifikan", 0.3),
            ("mahal", 0.2)
        ],
        "efficiency": [
            ("efisien", 0.9),
            ("hemat energi", 0.85),
            ("otomatisasi tinggi", 0.8),
            ("tidak efisien", 0.3),
            ("boros", 0.2)
        ]
    }

    for keyword, score in keyword_scores.get(category, []):
        if keyword in text:
            return score

    return 0.5  # default jika tidak ada kata kunci yang cocok
