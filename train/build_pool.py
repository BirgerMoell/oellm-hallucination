#!/usr/bin/env python3
"""Question pool for on-policy hallucination DPO (and the knowledge-boundary SFT set): natural questions, no answer format.

Three parts, per language:
  fact        short factual questions (birth year and place, film director, book author, company founding, river mouth,
              regional capital) from the eu_qa train split, with the format instruction removed. Gold answers from
              Wikidata; popularity bands head / torso / tail.
  open        "Who is X?" / "Tell me about X." / birth questions about real people (benchmark/build_questions.py
              sampler, another seed, more occupations), with Wikidata reference facts; bands obscure / mid / known.
  made_up     subjects that do not exist: invented surnames, plausible everyday names (a common first name plus a
              common surname of the language) and invented novels, films, companies, villages, rivers, paintings and
              albums. Names are built differently from the benchmark's and checked against Wikidata.
Every benchmark subject (data/questions.jsonl) is excluded, by Wikidata id and by name.

  python build_pool.py --eu-qa train_qa.jsonl --out ../data/train/pool.jsonl [--languages en de ...] [--seed 20261116]
Needs internet (Wikidata). Finished languages are cached next to --out, so an interrupted run continues.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmark"))
import build_questions as B  # noqa: E402

FACT_PER_LANG = {"tail": 150, "torso": 130, "head": 120}
OPEN_PER_LANG = {"real_obscure": 90, "real_mid": 60, "real_known": 50}
MADE_UP_PER_LANG = {"invented_surname": 75, "plausible_name": 75, "thing": 150}
OCCUPATIONS = {**B.OCCUPATIONS, "Q2066131": "athlete", "Q639669": "musician", "Q1930187": "journalist",
               "Q43845": "businessperson", "Q1622272": "university teacher", "Q250867": "Catholic priest"}

# First names and surnames not used by the benchmark generator.
FIRST = {
    "en": ["Fionnuala", "Darragh", "Clodagh", "Ronan", "Saoirse", "Cillian", "Muireann", "Fergal", "Éabha", "Tadhg"],
    "de": ["Henrike", "Florian", "Annika", "Moritz", "Birgit", "Sebastian", "Theresa", "Niklas", "Carola", "Dominik"],
    "fr": ["Solène", "Baptiste", "Apolline", "Quentin", "Mathilde", "Romain", "Clémence", "Bastien", "Ophélie", "Gaëtan"],
    "es": ["Rocío", "Íñigo", "Amparo", "Sergio", "Begoña", "Álvaro", "Pilar", "Gonzalo", "Inmaculada", "Rubén"],
    "it": ["Benedetta", "Lorenzo", "Ilaria", "Tommaso", "Federica", "Riccardo", "Martina", "Edoardo", "Valentina", "Filippo"],
    "nl": ["Marieke", "Jeroen", "Ilse", "Wouter", "Esmee", "Sander", "Floor", "Gijs", "Noor", "Maarten"],
    "sv": ["Ebba", "Viktor", "Klara", "Hampus", "Astrid", "Filip", "Tove", "Ludvig", "Signe", "Axel"],
    "pl": ["Małgorzata", "Krzysztof", "Ewelina", "Grzegorz", "Dorota", "Łukasz", "Justyna", "Wojciech", "Beata", "Marcin"],
    "pt": ["Joana", "Duarte", "Matilde", "Gonçalo", "Rita", "Afonso", "Carolina", "Nuno", "Teresa", "Rodrigo"],
    "fi": ["Hanna", "Jussi", "Riikka", "Pekka", "Satu", "Tuomas", "Elina", "Lauri", "Minna", "Jarmo"],
}
SURNAME = {  # common surnames; the full name is kept only if Wikidata has no item with it
    "en": ["Byrne", "Kavanagh", "Doherty", "Moran", "Lynch", "Quinn", "Brennan", "Dunne", "Hogan", "Nolan"],
    "de": ["Brandt", "Hartmann", "Krüger", "Vogel", "Lorenz", "Pohl", "Kuhn", "Winkler", "Seidel", "Ebert"],
    "fr": ["Lefèvre", "Mercier", "Fontaine", "Chevalier", "Gauthier", "Perrin", "Morel", "Rousseau", "Lemaire", "Barbier"],
    "es": ["Iglesias", "Cabrera", "Domínguez", "Vidal", "Pascual", "Herrero", "Carmona", "Soler", "Calvo", "Gallardo"],
    "it": ["Barbieri", "Fontana", "Caruso", "Mariani", "Rinaldi", "Ferraro", "Galli", "Conti", "Longo", "Testa"],
    "nl": ["Bosman", "Kuipers", "Verhoeven", "Dijkman", "Postma", "Hendriks", "Brouwer", "Schouten", "Mulder", "Timmer"],
    "sv": ["Sandberg", "Holmgren", "Ekström", "Wallin", "Nyström", "Engberg", "Sjöberg", "Forsberg", "Åberg", "Lindahl"],
    "pl": ["Wróblewski", "Zając", "Sikora", "Baran", "Szulc", "Kubiak", "Wilk", "Czarnecki", "Sobczak", "Urban"],
    "pt": ["Figueiredo", "Antunes", "Pinheiro", "Barros", "Correia", "Mendes", "Tavares", "Simões", "Lopes", "Macedo"],
    "fi": ["Heikkilä", "Koivisto", "Laaksonen", "Rantanen", "Salminen", "Turunen", "Väisänen", "Hiltunen", "Saarinen", "Lehto"],
}
SYLL = ["bra", "dem", "fol", "gun", "hes", "jal", "kos", "lam", "nir", "pol", "quer", "ros", "sim", "tov", "ulm", "wen",
        "yar", "zel", "cor", "dif", "esk", "hal", "ist", "mur", "nev", "orl", "pir", "sut", "vam", "wol"]
EXTRA_THING_Q = {
    "painting": {"en": "Who painted “{s}”, and when?", "de": "Wer hat das Gemälde „{s}“ gemalt, und wann?",
                 "fr": "Qui a peint « {s} », et quand ?", "es": "¿Quién pintó «{s}» y cuándo?",
                 "it": "Chi ha dipinto «{s}» e quando?", "nl": "Wie schilderde ‘{s}’, en wanneer?",
                 "sv": "Vem målade ”{s}”, och när?", "pl": "Kto namalował obraz „{s}” i kiedy?",
                 "pt": "Quem pintou «{s}» e quando?", "fi": "Kuka maalasi teoksen ”{s}”, ja milloin?"},
    "album": {"en": "Which band released the album “{s}”?", "de": "Welche Band hat das Album „{s}“ veröffentlicht?",
              "fr": "Quel groupe a sorti l'album « {s} » ?", "es": "¿Qué grupo publicó el álbum «{s}»?",
              "it": "Quale band ha pubblicato l'album «{s}»?", "nl": "Welke band bracht het album ‘{s}’ uit?",
              "sv": "Vilket band släppte albumet ”{s}”?", "pl": "Który zespół wydał album „{s}”?",
              "pt": "Que banda lançou o álbum «{s}»?", "fi": "Mikä yhtye julkaisi albumin ”{s}”?"},
}
THING_KINDS = ["novel", "film", "company", "village", "river", "painting", "album"]


def word(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(SYLL) for _ in range(n)).capitalize()


def in_wikidata(name: str, lang: str) -> bool:
    """An item with this exact label or alias, searched in English and in the question's language."""
    for search_lang in dict.fromkeys(("en", lang)):
        hits = B.get_json("https://www.wikidata.org/w/api.php",
                          {"action": "wbsearchentities", "search": name, "language": search_lang,
                           "strictlanguage": "false", "type": "item", "limit": 7, "format": "json"}).get("search", [])
        if any(name.casefold() in (h.get("label", "").casefold(), h.get("match", {}).get("text", "").casefold())
               for h in hits):
            return True
        time.sleep(0.1)
    return False


def made_up(rng: random.Random, lang: str, banned: set[str]) -> list[dict]:
    rows: list[dict] = []
    person_kinds = ["who", "tell", "born_year", "born_place"]
    for style, n in (("invented_surname", MADE_UP_PER_LANG["invented_surname"]),
                     ("plausible_name", MADE_UP_PER_LANG["plausible_name"])):
        tries, got = 0, 0
        while got < n and tries < 40 * n:
            tries += 1
            first = rng.choice(FIRST[lang])
            last = word(rng, 3) if style == "invented_surname" else rng.choice(SURNAME[lang])
            s = f"{first} {last}"
            if s in banned or in_wikidata(s, lang) or (style == "invented_surname" and in_wikidata(last, lang)):
                banned.add(s)
                continue
            banned.add(s)
            k = person_kinds[got % len(person_kinds)]
            rows.append({"band": "made_up_person", "style": style, "qtype": k, "subject": s,
                         "question": B.PERSON_Q[k][lang].format(s=s)})
            got += 1
    got = 0
    while got < MADE_UP_PER_LANG["thing"]:
        k = THING_KINDS[got % len(THING_KINDS)]
        s = {"novel": lambda: f"{word(rng, 2)} {word(rng, 2)}", "film": lambda: f"{word(rng, 3)}",
             "company": lambda: f"{word(rng, 2)}{rng.choice(['tek', 'lab', 'nova', 'gruppen', 'werk'])} {B.LEGAL[lang]}",
             "village": lambda: word(rng, 2) + rng.choice(B.PLACE_END[lang]), "river": lambda: word(rng, 2),
             "painting": lambda: f"{word(rng, 2)} {word(rng, 3)}", "album": lambda: f"{word(rng, 3)}"}[k]()
        core = s.replace(f" {B.LEGAL[lang]}", "")
        if s in banned or in_wikidata(core, lang):
            banned.add(s)
            continue
        banned.add(s)
        templates = B.THING_Q.get(k) or EXTRA_THING_Q[k]
        rows.append({"band": "made_up_thing", "style": "invented", "qtype": k, "subject": s,
                     "question": templates[lang].format(s=s, y=rng.randint(1950, 2020))})
        got += 1
    return rows


def fact_rows(eu_qa: Path, lang: str, rng: random.Random, exclude_ids: set[str]) -> list[dict]:
    by_band: dict[str, list[dict]] = defaultdict(list)
    for line in eu_qa.open(encoding="utf-8"):
        r = json.loads(line)
        if r["language"] != lang or r["source"].split(":", 1)[1] in exclude_ids:
            continue
        by_band[r["band"]].append(r)
    rows = []
    for band, n in FACT_PER_LANG.items():
        pool = sorted(by_band[band], key=lambda r: r["id"])
        rng.shuffle(pool)
        by_type: dict[str, list[dict]] = defaultdict(list)
        for r in pool:
            by_type[r["subset"]].append(r)
        types = sorted(by_type)
        i = 0
        while len([x for x in rows if x["band"] == f"fact_{band}"]) < n and any(by_type.values()):
            t = types[i % len(types)]
            i += 1
            if not by_type[t]:
                continue
            r = by_type[t].pop()
            rows.append({"band": f"fact_{band}", "qtype": r["subset"], "subject": "",
                         "question": r["prompt"].split("\n\n")[0].strip(),
                         "reference": {"wikidata": r["source"].split(":", 1)[1], "answers": r["label"]["answers"][:6],
                                       "answer_kind": r["label"]["kind"], "sitelinks": r.get("sitelinks")},
                         "source_id": r["id"]})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eu-qa", type=Path, required=True, help="eu_qa train_qa.jsonl (oellm-rlvr data/eu-tasks)")
    ap.add_argument("--benchmark", type=Path, default=Path(__file__).resolve().parent.parent / "data/questions.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--languages", nargs="+", default=list(B.COUNTRY))
    ap.add_argument("--seed", type=int, default=20261116)
    a = ap.parse_args()
    bench = [json.loads(line) for line in a.benchmark.open(encoding="utf-8")]
    exclude_ids = {q["reference"]["wikidata"] for q in bench if q.get("reference")}
    banned = {q["subject"] for q in bench}
    cache = a.out.parent / ".pool_cache"
    cache.mkdir(parents=True, exist_ok=True)
    out = []
    for lang in a.languages:
        part = cache / f"{a.seed}-{lang}.json"
        if part.exists():
            rows = json.loads(part.read_text())
        else:
            rng = random.Random(f"{a.seed}:{lang}")
            rows = fact_rows(a.eu_qa, lang, rng, exclude_ids)
            for band, n in OPEN_PER_LANG.items():
                got = B.real(rng, lang, band, n, a.seed, occupations=OCCUPATIONS, exclude=exclude_ids, strict=False)
                rows += [{**r, "band": band} for r in got]
            rows += made_up(rng, lang, banned)
            part.write_text(json.dumps(rows, ensure_ascii=False))
        banned |= {r["subject"] for r in rows if r["subject"]}
        for i, r in enumerate(rows):
            out.append({"id": f"pool-{lang}-{r['band']}-{i:04d}", "language": lang, **r})
        print(lang, dict(Counter(r["band"] for r in rows)), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    print(len(out), "questions ->", a.out, dict(Counter(r["band"] for r in out)))


if __name__ == "__main__":
    main()
