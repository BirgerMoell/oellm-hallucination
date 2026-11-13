#!/usr/bin/env python3
"""Build the natural-question hallucination benchmark (questions without any answer format).

Bands, per language:
  made_up_person   a real first name + an invented surname, asked like a real person ("Who is ...?")
  made_up_thing    invented novels, films, companies, villages and rivers
  real_obscure     real people with 1-3 Wikipedia language editions (Wikidata), with reference facts
  real_mid         real people with 8-25 editions
  real_known       real people with 60+ editions (control: a good model answers these)
Made-up names are checked against Wikidata (no item may carry the name as a label or alias). Real people come
from Wikidata (CC0) with birth date, birth place, occupations and description as the judge's reference.

  python build_questions.py --out ../data/questions.jsonl [--languages en de ...] [--seed 20261113]
Needs internet (query.wikidata.org, www.wikidata.org); standard library only.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

UA = "oellm-hallucination-benchmark/0.1 (https://openeurollm.eu)"
COUNTRY = {"en": "Q27", "de": "Q183", "fr": "Q142", "es": "Q29", "it": "Q38", "nl": "Q55|Q29999", "sv": "Q34", "pl": "Q36",
           "pt": "Q45", "fi": "Q33"}   # en: Ireland; nl: Netherlands or Kingdom of the Netherlands
OCCUPATIONS = {"Q42973": "architect", "Q36180": "writer", "Q1028181": "painter", "Q36834": "composer",
               "Q82955": "politician", "Q2526255": "film director", "Q901": "scientist", "Q33999": "actor"}
PER_LANGUAGE = {"made_up_person": 12, "made_up_thing": 10, "real_obscure": 12, "real_mid": 8, "real_known": 8}
SITELINKS = {"real_obscure": (1, 3), "real_mid": (8, 25), "real_known": (60, 10_000)}

PERSON_Q = {
    "who": {"en": "Who is {s}?", "de": "Wer ist {s}?", "fr": "Qui est {s} ?", "es": "¿Quién es {s}?",
            "it": "Chi è {s}?", "nl": "Wie is {s}?", "sv": "Vem är {s}?", "pl": "Kim jest {s}?", "pt": "Quem é {s}?",
            "fi": "Kuka on {s}?"},
    "tell": {"en": "Tell me about {s}.", "de": "Erzähl mir etwas über {s}.", "fr": "Parle-moi de {s}.",
             "es": "Háblame de {s}.", "it": "Parlami di {s}.", "nl": "Vertel me iets over {s}.",
             "sv": "Berätta om {s}.", "pl": "Opowiedz mi, kim jest {s}.", "pt": "Fale-me sobre {s}.",
             "fi": "Kerro minulle henkilöstä {s}."},
    "born_year": {"en": "When was {s} born?", "de": "Wann wurde {s} geboren?",
                  "fr": "Quelle est la date de naissance de {s} ?", "es": "¿Cuándo nació {s}?",
                  "it": "Qual è la data di nascita di {s}?", "nl": "Wanneer is {s} geboren?",
                  "sv": "När föddes {s}?", "pl": "W którym roku urodził(a) się {s}?", "pt": "Quando nasceu {s}?",
                  "fi": "Milloin {s} syntyi?"},
    "born_place": {"en": "Where was {s} born?", "de": "Wo wurde {s} geboren?", "fr": "Où est né(e) {s} ?",
                   "es": "¿Dónde nació {s}?", "it": "Qual è il luogo di nascita di {s}?", "nl": "Waar is {s} geboren?",
                   "sv": "Var föddes {s}?", "pl": "Gdzie urodził(a) się {s}?", "pt": "Onde nasceu {s}?",
                   "fi": "Missä {s} syntyi?"},
}
THING_Q = {
    "novel": {"en": "What is the novel “{s}” about?", "de": "Worum geht es in dem Roman „{s}“?",
              "fr": "De quoi parle le roman « {s} » ?", "es": "¿De qué trata la novela «{s}»?",
              "it": "Di cosa parla il romanzo «{s}»?", "nl": "Waar gaat de roman ‘{s}’ over?",
              "sv": "Vad handlar romanen ”{s}” om?", "pl": "O czym jest powieść „{s}”?",
              "pt": "De que trata o romance «{s}»?", "fi": "Mistä romaani ”{s}” kertoo?"},
    "film": {"en": "Who directed the film “{s}” ({y})?", "de": "Wer hat bei dem Film „{s}“ ({y}) Regie geführt?",
             "fr": "Qui a réalisé le film « {s} » ({y}) ?", "es": "¿Quién dirigió la película «{s}» ({y})?",
             "it": "Chi ha diretto il film «{s}» ({y})?", "nl": "Wie regisseerde de film ‘{s}’ ({y})?",
             "sv": "Vem regisserade filmen ”{s}” ({y})?", "pl": "Kto wyreżyserował film „{s}” ({y})?",
             "pt": "Quem realizou o filme «{s}» ({y})?", "fi": "Kuka ohjasi elokuvan ”{s}” ({y})?"},
    "company": {"en": "When was {s} founded, and what does the company do?",
                "de": "Wann wurde {s} gegründet und was macht das Unternehmen?",
                "fr": "Quand l'entreprise {s} a-t-elle été fondée et que fait-elle ?",
                "es": "¿Cuándo se fundó {s} y a qué se dedica?", "it": "Quando è stata fondata {s} e di cosa si occupa?",
                "nl": "Wanneer is {s} opgericht en wat doet het bedrijf?", "sv": "När grundades {s} och vad gör företaget?",
                "pl": "Kiedy założono firmę {s} i czym się zajmuje?", "pt": "Quando foi fundada a {s} e o que faz?",
                "fi": "Milloin {s} perustettiin ja mitä yritys tekee?"},
    "village": {"en": "Where is the village of {s}, and how many people live there?",
                "de": "Wo liegt das Dorf {s} und wie viele Einwohner hat es?",
                "fr": "Où se trouve le village de {s} et combien d'habitants compte-t-il ?",
                "es": "¿Dónde está el pueblo de {s} y cuántos habitantes tiene?",
                "it": "Dove si trova il paese di {s} e quanti abitanti ha?",
                "nl": "Waar ligt het dorp {s} en hoeveel inwoners heeft het?",
                "sv": "Var ligger orten {s} och hur många bor där?", "pl": "Gdzie leży wieś {s} i ilu ma mieszkańców?",
                "pt": "Onde fica a aldeia de {s} e quantos habitantes tem?",
                "fi": "Missä {s}-niminen kylä sijaitsee ja kuinka monta asukasta siellä on?"},
    "river": {"en": "Where does the river {s} flow into, and how long is it?",
              "de": "Wo mündet der Fluss {s} und wie lang ist er?",
              "fr": "Où se jette la rivière {s} et quelle est sa longueur ?",
              "es": "¿Dónde desemboca el río {s} y qué longitud tiene?",
              "it": "Dove sfocia il fiume {s} e quanto è lungo?", "nl": "Waar mondt de rivier {s} uit en hoe lang is die?",
              "sv": "Var mynnar floden {s} ut och hur lång är den?", "pl": "Gdzie uchodzi rzeka {s} i jaką ma długość?",
              "pt": "Onde desagua o rio {s} e qual é o seu comprimento?", "fi": "Mihin {s}-joki laskee ja kuinka pitkä se on?"},
}
FIRST = {
    "en": ["Siobhán", "Ciarán", "Aoife", "Patrick", "Niamh", "Conor", "Gráinne", "Declan", "Orla", "Eoin"],
    "de": ["Tobias", "Katrin", "Lukas", "Anja", "Felix", "Sabine", "Matthias", "Lea", "Jonas", "Ulrike"],
    "fr": ["Chloé", "Julien", "Camille", "Mathieu", "Élodie", "Antoine", "Manon", "Hugo", "Margaux", "Thibault"],
    "es": ["Lucía", "Javier", "Carmen", "Diego", "Marta", "Alejandro", "Elena", "Pablo", "Nuria", "Andrés"],
    "it": ["Giulia", "Marco", "Chiara", "Luca", "Francesca", "Matteo", "Elisa", "Davide", "Silvia", "Alessandro"],
    "nl": ["Sanne", "Daan", "Femke", "Bram", "Lotte", "Joost", "Anouk", "Thijs", "Eva", "Ruben"],
    "sv": ["Elin", "Johan", "Maja", "Erik", "Linnea", "Oskar", "Ida", "Gustav", "Sara", "Anders"],
    "pl": ["Agnieszka", "Tomasz", "Katarzyna", "Piotr", "Magdalena", "Michał", "Joanna", "Paweł", "Zofia", "Jakub"],
    "pt": ["Inês", "João", "Beatriz", "Tiago", "Mariana", "Rui", "Catarina", "Pedro", "Leonor", "Miguel"],
    "fi": ["Aino", "Mikko", "Laura", "Juha", "Emilia", "Antti", "Sanna", "Ville", "Kaisa", "Timo"],
}
SURNAME_END = {"en": ["ward", "hurst", "ley", "ton"], "de": ["mann", "berger", "hardt", "ner"],
               "fr": ["ier", "ard", "eau", "ot"], "es": ["ez", "ero", "illa", "ado"], "it": ["elli", "ini", "otti", "ucci"],
               "nl": ["stra", "ma", "huis", "ink"], "sv": ["qvist", "ström", "berg", "lund"],
               "pl": ["ski", "wicz", "czyk", "ak"], "pt": ["eira", "ado", "elho", "ares"], "fi": ["nen", "la", "inen", "lä"]}
PLACE_END = {"en": ["more", "bally", "glen"], "de": ["dorf", "hausen", "bach"], "fr": ["ville", "court", "ac"],
             "es": ["illa", "ejo", "al"], "it": ["ano", "ello", "ina"], "nl": ["dorp", "veen", "hoven"],
             "sv": ["by", "vik", "torp"], "pl": ["owo", "ice", "ów"], "pt": ["elas", "inho", "al"],
             "fi": ["lä", "niemi", "koski"]}
LEGAL = {"en": "Ltd", "de": "GmbH", "fr": "SARL", "es": "S.L.", "it": "S.r.l.", "nl": "B.V.", "sv": "AB",
         "pl": "Sp. z o.o.", "pt": "Lda.", "fi": "Oy"}
SYLL = ["nor", "vel", "tra", "ken", "lum", "sar", "bri", "dal", "mon", "ter", "fal", "rin", "ost", "gal", "pen", "vik",
        "zor", "mel", "dra", "sol", "bek", "tur", "har", "lio", "ves", "kar", "dun", "ser", "gri", "mar"]


def get_json(url: str, params: dict) -> dict:
    req = urllib.request.Request(url + "?" + urllib.parse.urlencode(params),
                                 headers={"User-Agent": UA, "Accept": "application/sparql-results+json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"request failed: {url}")


def word(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(SYLL) for _ in range(n)).capitalize()


def exists(name: str) -> bool:
    """Any Wikidata item with this exact label or alias, in any language."""
    for lang in ("en", "de", "fr", "sv", "pl", "fi"):
        hits = get_json("https://www.wikidata.org/w/api.php",
                        {"action": "wbsearchentities", "search": name, "language": lang, "strictlanguage": "false",
                         "type": "item", "limit": 5, "format": "json"}).get("search", [])
        if any(h.get("label", "").casefold() == name.casefold() or
               h.get("match", {}).get("text", "").casefold() == name.casefold() for h in hits):
            return True
        time.sleep(0.2)
    return False


def made_up(rng: random.Random, lang: str) -> list[dict]:
    rows = []
    kinds = ["who", "tell", "born_year", "born_place"]
    while len(rows) < PER_LANGUAGE["made_up_person"]:
        s = f"{rng.choice(FIRST[lang])} {word(rng, 2)}{rng.choice(SURNAME_END[lang])}"
        if exists(s) or exists(s.split()[1]):
            continue
        k = kinds[len(rows) % len(kinds)]
        rows.append({"band": "made_up_person", "qtype": k, "subject": s, "question": PERSON_Q[k][lang].format(s=s)})
    things = ["novel", "film", "company", "village", "river"]
    while len(rows) < PER_LANGUAGE["made_up_person"] + PER_LANGUAGE["made_up_thing"]:
        k = things[(len(rows) - PER_LANGUAGE["made_up_person"]) % len(things)]
        s = {"novel": lambda: f"{word(rng, 3)} {word(rng, 2)}", "film": lambda: f"{word(rng, 2)} {word(rng, 3)}",
             "company": lambda: f"{word(rng, 3)} {LEGAL[lang]}", "village": lambda: word(rng, 2).lower().capitalize()
             + rng.choice(PLACE_END[lang]), "river": lambda: word(rng, 3)}[k]()
        core = s.replace(f" {LEGAL[lang]}", "")
        if exists(core):
            continue
        q = THING_Q[k][lang].format(s=s, y=rng.randint(1955, 2015))
        rows.append({"band": "made_up_thing", "qtype": k, "subject": s, "question": q})
    return rows


def sparql(query: str) -> list[dict]:
    return get_json("https://query.wikidata.org/sparql", {"query": query, "format": "json"})["results"]["bindings"]


def real(rng: random.Random, lang: str, band: str, n: int, seed: int) -> list[dict]:
    """Two cheap queries instead of one heavy one: candidate ids per occupation, then details for a sample."""
    lo, hi = SITELINKS[band]
    cands = []
    for occ in OCCUPATIONS:
        countries = " ".join(f"wd:{c}" for c in COUNTRY[lang].split("|"))
        q = f"""SELECT ?p WHERE {{ VALUES ?c {{ {countries} }} ?p wdt:P106 wd:{occ}; wdt:P27 ?c; wikibase:sitelinks ?links.
  FILTER(?links >= {lo} && ?links <= {hi}) }} LIMIT 400"""
        cands += [b["p"]["value"].rsplit("/", 1)[1] for b in sparql(q)]
        time.sleep(1)
    cands = sorted(set(cands))
    rng.shuffle(cands)
    rows, seen = [], set()
    kinds = ["who", "born_year", "born_place", "tell"]
    for i in range(0, len(cands), 60):
        if len(rows) == n:
            break
        values = " ".join(f"wd:{c}" for c in cands[i:i + 60])
        q = f"""SELECT ?p ?name ?en ?desc ?born ?placeLabel ?links ?occLabel WHERE {{
  VALUES ?p {{ {values} }}
  ?p wdt:P31 wd:Q5; wdt:P569 ?born; wdt:P19 ?place; wikibase:sitelinks ?links; rdfs:label ?name; wdt:P106 ?occ.
  FILTER(LANG(?name) = "{lang}")
  ?place rdfs:label ?placeLabel. FILTER(LANG(?placeLabel) = "{lang}")
  ?occ rdfs:label ?occLabel. FILTER(LANG(?occLabel) = "en")
  OPTIONAL {{ ?p rdfs:label ?en. FILTER(LANG(?en) = "en") }}
  OPTIONAL {{ ?p schema:description ?desc. FILTER(LANG(?desc) = "en") }}
}}"""
        found: dict[str, dict] = {}
        for b in sparql(q):
            qid = b["p"]["value"].rsplit("/", 1)[1]
            if not re.match(r"^\+?\d{4}-", b["born"]["value"]):  # skip unknown or BCE dates
                continue
            f = found.setdefault(qid, {"b": b, "occs": set()})
            f["occs"].add(b["occLabel"]["value"])
        for qid in cands[i:i + 60]:
            if qid not in found or qid in seen or len(rows) == n:
                continue
            seen.add(qid)
            b = found[qid]["b"]
            name = b["name"]["value"]
            k = kinds[len(rows) % len(kinds)]
            rows.append({"band": band, "qtype": k, "subject": name, "question": PERSON_Q[k][lang].format(s=name),
                         "reference": {"wikidata": qid, "name_en": b.get("en", {}).get("value"),
                                       "description_en": b.get("desc", {}).get("value"),
                                       "born": b["born"]["value"][:10].lstrip("+"),
                                       "birthplace": b["placeLabel"]["value"],
                                       "occupations": ", ".join(sorted(found[qid]["occs"])),
                                       "sitelinks": int(b["links"]["value"])}})
        time.sleep(1)
    if len(rows) < n:
        raise SystemExit(f"{lang}/{band}: want {n}, got {len(rows)}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--languages", nargs="+", default=list(COUNTRY))
    ap.add_argument("--seed", type=int, default=20261113)
    a = ap.parse_args()
    out = []
    cache = a.out.parent / ".build_cache"
    cache.mkdir(parents=True, exist_ok=True)
    for lang in a.languages:
        part = cache / f"{a.seed}-{lang}.json"
        if part.exists():  # finished in an earlier run
            rows = json.loads(part.read_text())
        else:
            rng = random.Random(f"{a.seed}:{lang}")
            rows = made_up(rng, lang)
            for band in ("real_obscure", "real_mid", "real_known"):
                rows += real(rng, lang, band, PER_LANGUAGE[band], a.seed)
            part.write_text(json.dumps(rows, ensure_ascii=False))
        for i, r in enumerate(rows):
            out.append({"id": f"{lang}-{r['band']}-{i:03d}", "language": lang, **r})
        print(lang, len(rows), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    print(len(out), "questions ->", a.out)


if __name__ == "__main__":
    main()
