"""Build the calibration set of sentence pairs and its appendix table.

Each pair differs from a base sentence in one controlled way. The set is used to
build a sensitivity profile of bge-m3 cosine similarity (an ordering of contrast
types), not a threshold.

Writes:
  thesis_v2/notes/calibration_set.csv          (curation copy, has a `keep` column)
  thesis_v2/04_Appendix/calibration_set.tex    (longtable, regenerated from the CSV)

Usage:
  python -m thesis_corpus.build_calibration_set            # write CSV from the sentences below
  python -m thesis_corpus.build_calibration_set --tex-only # rebuild the appendix from the curated CSV
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CSV_PATH = ROOT / "thesis_v2" / "notes" / "calibration_set.csv"
TEX_PATH = ROOT / "thesis_v2" / "04_Appendix" / "calibration_set.tex"
COLUMNS = ["pair_id", "contrast", "domain", "register", "base_id", "sentence_a", "sentence_b",
           "lang_a", "lang_b", "notes", "keep"]

# Per-base variants. Keys: base, near (0), para (1), opp (2), neg (3), fr (4),
# unrel (6), sp (8, None = skipped), plus notes per contrast.
BASES = {
    # --- cult, neutral written ---
    "N1": dict(domain="cult", register="neutral", claim="isolation from family",
        base="A cult isolates its members from their families.",
        near="a cult isolates its members from their families",
        para="A cult cuts its followers off from their relatives.",
        opp="A cult keeps its members close to their families.",
        neg="A cult does not isolate its members from their families.",
        fr="Une secte isole ses membres de leur famille.",
        unrel="The library lends its books to local schools.",
        sp="Cults isolate their members from their families."),
    "N2": dict(domain="cult", register="neutral", claim="danger",
        base="Cults are dangerous for the people who join them.",
        near="cults are dangerous for the people who join them",
        para="Joining a cult puts people at serious risk.",
        opp="Cults are harmless for the people who join them.",
        neg="Cults are not dangerous for the people who join them.",
        fr="Les sectes sont dangereuses pour les personnes qui les rejoignent.",
        unrel="Bicycles are practical for people who live in cities.",
        sp="A cult is dangerous for the people who join it."),
    "N3": dict(domain="cult", register="neutral", claim="obedience to a leader",
        base="Cult members obey their leader without question.",
        near="cult members obey their leader without question",
        para="People in a cult do whatever their leader tells them.",
        opp="Cult members openly question their leader.",
        neg="Cult members do not obey their leader without question.",
        fr="Les membres d'une secte obéissent à leur chef sans discuter.",
        unrel="Train passengers wait on the platform without complaint.",
        sp="A cult member obeys the leader without question."),
    "N4": dict(domain="cult", register="neutral", claim="financial pressure",
        base="A cult pressures its members to give it their money.",
        near="a cult pressures its members to give it their money",
        para="A cult pushes its followers to hand over their savings.",
        opp="A cult lets its members keep their money.",
        neg="A cult does not pressure its members to give it their money.",
        fr="Une secte fait pression sur ses membres pour qu'ils lui donnent leur argent.",
        unrel="A gym encourages its members to come three times a week.",
        sp="Cults pressure their members to give them their money."),
    "N5": dict(domain="cult", register="neutral", claim="free entry and exit",
        base="People are free to leave a cult whenever they want.",
        near="people are free to leave a cult whenever they want",
        para="Anyone can walk out of a cult at any time.",
        opp="People are trapped in a cult once they have joined.",
        neg="People are not free to leave a cult whenever they want.",
        fr="Les gens sont libres de quitter une secte quand ils le veulent.",
        unrel="Visitors can buy tickets at the entrance of the museum.",
        sp="A person is free to leave a cult whenever they want."),
    "N6": dict(domain="cult", register="neutral", claim="stigmatising use of the word",
        base="Calling a group a cult is a way of stigmatising it.",
        near="calling a group a cult is a way of stigmatising it",
        para="Labelling a group a cult is meant to discredit it.",
        opp="Calling a group a cult is a fair description of it.",
        neg="Calling a group a cult is not a way of stigmatising it.",
        fr="Qualifier un groupe de secte est une façon de le stigmatiser.",
        unrel="Painting a room white is a way of making it brighter.",
        sp="Calling groups cults is a way of stigmatising them."),
    # --- cult, spoken (interview-like) ---
    "S1": dict(domain="cult", register="spoken", claim="brainwashing",
        base="i mean in a cult they brainwash you, you know",
        near="I mean, in a cult they brainwash you, you know.",
        para="like in a cult they basically mess with your head, you know",
        opp="i mean in a cult they let you think for yourself, you know",
        neg="i mean in a cult they don't brainwash you, you know",
        fr="enfin dans une secte ils te lavent le cerveau, tu vois",
        unrel="i mean in the morning they open the bakery early, you know",
        sp="i mean in cults they brainwash you, you know"),
    "S2": dict(domain="cult", register="spoken", claim="shame after leaving",
        base="yeah people who leave a cult feel really ashamed after, like",
        near="Yeah, people who leave a cult feel really ashamed after, like.",
        para="yeah when you get out of a cult you're like super embarrassed about it",
        opp="yeah people who leave a cult feel really proud after, like",
        neg="yeah people who leave a cult don't feel really ashamed after, like",
        fr="ouais les gens qui quittent une secte ont vraiment honte après, genre",
        unrel="yeah people who move to berlin feel really cold in winter, like",
        sp="yeah a person who leaves a cult feels really ashamed after, like"),
    "S3": dict(domain="cult", register="spoken", claim="reporting to authorities",
        base="honestly if a friend joined a cult i'd call the police, you know",
        near="Honestly, if a friend joined a cult I'd call the police, you know.",
        para="honestly if a mate got into a cult i'd go to the cops, you know",
        opp="honestly if a friend joined a cult i'd leave them alone, you know",
        neg="honestly if a friend joined a cult i wouldn't call the police, you know",
        fr="franchement si un ami entrait dans une secte j'appellerais la police, tu vois",
        unrel="honestly if a friend moved to lisbon i'd visit them, you know",
        sp="honestly if friends joined cults i'd call the police, you know"),
    # --- cult, academic ---
    "A1": dict(domain="cult", register="academic", claim="isolation from family",
        base="Cult membership is associated with the progressive isolation of members from their kinship networks.",
        near="cult membership is associated with the progressive isolation of members from their kinship networks",
        para="Joining a cult tends to be accompanied by a gradual separation of adherents from their families.",
        opp="Cult membership is associated with the progressive strengthening of members' kinship networks.",
        neg="Cult membership is not associated with the progressive isolation of members from their kinship networks.",
        fr="L'appartenance à une secte est associée à l'isolement progressif des membres de leurs réseaux de parenté.",
        unrel="Urban cycling is associated with the progressive reduction of commuting times.",
        sp=None, sp_note="skipped: 'cult membership' has no natural plural that changes only number"),
    "A2": dict(domain="cult", register="academic", claim="brainwashing",
        base="The brainwashing model of cult conversion has been widely contested in the sociology of religion.",
        near="the brainwashing model of cult conversion has been widely contested in the sociology of religion",
        para="Sociologists of religion have largely disputed the idea that people are brainwashed into cults.",
        opp="The brainwashing model of cult conversion has been widely accepted in the sociology of religion.",
        neg="The brainwashing model of cult conversion has not been widely contested in the sociology of religion.",
        fr="Le modèle du lavage de cerveau pour expliquer la conversion aux sectes a été largement contesté en sociologie des religions.",
        unrel="The wave model of light propagation has been widely discussed in the history of physics.",
        sp=None, sp_note="skipped: 'cult' is used as a modifier, number does not apply"),
    "A3": dict(domain="cult", register="academic", claim="obedience to a leader",
        base="Cults are characterised by the concentration of authority in a charismatic leader.",
        near="cults are characterised by the concentration of authority in a charismatic leader",
        para="In cults, power tends to be held by a single charismatic figure.",
        opp="Cults are characterised by the distribution of authority among their members.",
        neg="Cults are not characterised by the concentration of authority in a charismatic leader.",
        fr="Les sectes se caractérisent par la concentration de l'autorité entre les mains d'un chef charismatique.",
        unrel="Alpine glaciers are characterised by the slow accumulation of compacted snow.",
        sp="A cult is characterised by the concentration of authority in a charismatic leader."),
    # --- cult, administrative (MIVILUDES-like) ---
    "M1": dict(domain="cult", register="administrative", claim="the state does not judge beliefs",
        base="The state's action against cults targets harmful behaviours rather than beliefs.",
        near="the state's action against cults targets harmful behaviours rather than beliefs",
        para="Public action against cults addresses damaging conduct, not doctrines.",
        opp="The state's action against cults targets beliefs rather than harmful behaviours.",
        neg="The state's action against cults does not target harmful behaviours rather than beliefs.",
        fr="L'action de l'État contre les sectes vise les comportements préjudiciables plutôt que les croyances.",
        fr_note="loaded: MIVILUDES frames this as action against 'dérives sectaires', not against 'sectes'",
        unrel="The city's action on traffic targets heavy vehicles rather than bicycles.",
        sp="The state's action against a cult targets harmful behaviours rather than beliefs."),
    "M2": dict(domain="cult", register="administrative", claim="reporting to authorities",
        base="Any person who suspects that a relative is under the influence of a cult may report the situation to the authorities.",
        near="any person who suspects that a relative is under the influence of a cult may report the situation to the authorities",
        para="Anyone worried that a family member is under a cult's influence can alert the authorities.",
        opp="Any person who suspects that a relative is under the influence of a cult should keep the situation private.",
        neg="Any person who suspects that a relative is under the influence of a cult may not report the situation to the authorities.",
        neg_note="the negation turns a permission into a prohibition",
        fr="Toute personne qui soupçonne qu'un proche est sous l'emprise d'une secte peut signaler la situation aux autorités.",
        fr_note="loaded: 'emprise' is a strong term (psychological hold), stronger than 'influence'",
        unrel="Any person who notices a damaged road sign may report it to the municipal services.",
        sp="Any person who suspects that a relative is under the influence of cults may report the situation to the authorities."),
    "M3": dict(domain="cult", register="administrative", claim="financial pressure",
        base="Exorbitant financial demands are one of the criteria used to identify a cult.",
        near="exorbitant financial demands are one of the criteria used to identify a cult",
        para="Excessive requests for money are among the signs used to recognise a cult.",
        opp="Financial demands are irrelevant to identifying a cult.",
        opp_note="more than one word changes, since the contrary position needs a different predicate",
        neg="Exorbitant financial demands are not one of the criteria used to identify a cult.",
        fr="Les exigences financières exorbitantes sont l'un des critères utilisés pour identifier une secte.",
        fr_note="loaded: the MIVILUDES criteria identify a 'dérive sectaire', not a 'secte'",
        unrel="Proof of residence is one of the documents required to register a vehicle.",
        sp="Exorbitant financial demands are one of the criteria used to identify cults."),
    # --- choir control (C1 is an exact parallel of N1) ---
    "C1": dict(domain="choir", register="neutral", claim="isolation from family (parallel of N1)",
        base="A choir isolates its members from their families.",
        near="a choir isolates its members from their families",
        para="A choir cuts its singers off from their relatives.",
        opp="A choir keeps its members close to their families.",
        neg="A choir does not isolate its members from their families.",
        fr="Une chorale isole ses membres de leur famille.",
        unrel="The library lends its books to local schools.",
        sp="Choirs isolate their members from their families."),
    "C2": dict(domain="choir", register="neutral", claim="rehearsals",
        base="The choir rehearses every Tuesday evening.",
        near="the choir rehearses every Tuesday evening",
        para="The singers meet each Tuesday night to practise.",
        opp="The choir rehearses only once a year.",
        opp_note="a factual sentence has no stance; the contrary is approximated",
        neg="The choir does not rehearse every Tuesday evening.",
        fr="La chorale répète tous les mardis soir.",
        unrel="The bakery opens every Sunday morning.",
        sp="Choirs rehearse every Tuesday evening."),
    "C3": dict(domain="choir", register="neutral", claim="effect on wellbeing",
        base="Singing in a choir is good for your mental health.",
        near="singing in a choir is good for your mental health",
        para="Choir singing improves psychological well-being.",
        opp="Singing in a choir is bad for your mental health.",
        neg="Singing in a choir is not good for your mental health.",
        fr="Chanter dans une chorale est bon pour la santé mentale.",
        unrel="Fresh bread tastes best in the morning.",
        sp="Singing in choirs is good for your mental health."),
    "C4": dict(domain="choir", register="neutral", claim="following the conductor",
        base="Choir members follow the conductor's instructions.",
        near="choir members follow the conductor's instructions",
        para="The singers do what the conductor tells them.",
        opp="Choir members ignore the conductor's instructions.",
        neg="Choir members do not follow the conductor's instructions.",
        fr="Les choristes suivent les consignes du chef de chœur.",
        unrel="Hotel guests leave their keys at the reception.",
        sp="A choir member follows the conductor's instructions."),
    "C5": dict(domain="choir", register="neutral", claim="membership fee",
        base="Members pay a small yearly fee to the choir.",
        near="members pay a small yearly fee to the choir",
        para="Singers contribute a modest annual subscription to the choir.",
        opp="Members are paid a small yearly fee by the choir.",
        neg="Members do not pay a small yearly fee to the choir.",
        fr="Les membres versent une petite cotisation annuelle à la chorale.",
        unrel="Tenants pay a small monthly fee for the shared garden.",
        sp="A member pays a small yearly fee to the choir."),
}

# Same topic, different claim: each base against the next base in its domain and register (cyclic).
SAME_TOPIC_GROUPS = [["N1", "N2", "N3", "N4", "N5", "N6"], ["S1", "S2", "S3"],
                     ["A1", "A2", "A3"], ["M1", "M2", "M3"], ["C1", "C2", "C3", "C4", "C5"]]

# Name vs description. Facts to check before use; see the notes column.
GROUPS = [
    ("Scientology", "A religious movement founded in the 1950s by the science-fiction writer L. Ron Hubbard, built around a practice called auditing."),
    ("Peoples Temple", "A movement led by Jim Jones whose members died in a mass murder-suicide at Jonestown, Guyana, in 1978."),
    ("Aum Shinrikyo", "A Japanese religious group founded by Shoko Asahara that carried out a sarin gas attack on the Tokyo subway in 1995."),
    ("Heaven's Gate", "A UFO religious group whose 39 members died in a collective suicide near San Diego in 1997."),
    ("Branch Davidians", "A religious group led by David Koresh whose compound near Waco, Texas, burned down at the end of a 51-day siege in 1993."),
    ("Order of the Solar Temple", "An esoteric order whose members died in a series of murders and suicides in Switzerland, France and Canada between 1994 and 1997."),
    ("Unification Church", "A religious movement founded in South Korea by Sun Myung Moon, known for its mass wedding ceremonies."),
    ("Raëlian Movement", "A UFO religion founded in France by Claude Vorilhon, known as Raël, which teaches that humans were created by extraterrestrials."),
    ("Hare Krishna movement", "A Hindu devotional movement founded in New York in 1966 by A. C. Bhaktivedanta Swami Prabhupada, known for chanting the Hare Krishna mantra."),
    ("NXIVM", "A self-help organisation whose founder, Keith Raniere, was convicted in 2019 of racketeering and sex trafficking."),
    ("Children of God", "A Christian movement founded in California in 1968 by David Berg, later known as The Family International."),
    ("Rajneesh movement", "A movement around the Indian guru Bhagwan Shree Rajneesh that built a commune in Oregon in the early 1980s."),
    ("Jehovah's Witnesses", "A Christian denomination that grew out of the Bible Student movement in nineteenth-century America and is known for door-to-door preaching."),
    ("Transcendental Meditation", "A meditation technique and movement introduced by Maharishi Mahesh Yogi in the 1950s, based on the silent repetition of a mantra."),
    ("Falun Gong", "A spiritual practice combining meditation and qigong exercises, founded in China in 1992 by Li Hongzhi."),
]
CHOIRS = [
    ("Vienna Boys' Choir", "A boys' choir founded in Vienna in 1498 that tours internationally."),
    ("Choir of King's College, Cambridge", "The choir of a Cambridge college, known for its Christmas Eve service of Nine Lessons and Carols."),
]

SINGLE_WORDS = [("cult", "cults", "en", "en", "cult"), ("secte", "sectes", "fr", "fr", "cult"),
                ("choir", "choirs", "en", "en", "choir"), ("cult", "sect", "en", "en", "cult"),
                ("cult", "secte", "en", "fr", "cult")]

LABEL_FRAMES_EN = [
    "A {} isolates its members from their families.",
    "The {} asks its members for money.",
    "She left the {} after ten years.",
    "The leader of the {} lives abroad.",
    "People join a {} when they feel lonely.",
]
LABELS_EN = ["sect", "new religious movement", "religious community", "choir", "company"]
LABEL_FRAMES_FR = [  # all replacement labels are feminine, so agreement is unchanged
    "La {} isole ses membres de leur famille.",
    "Elle a quitté la {} après dix ans.",
    "Le chef de la {} vit à l'étranger.",
]
LABELS_FR = ["communauté religieuse", "chorale", "entreprise"]

REGISTER_SHIFT = {
    "N1": ("like a cult just cuts you off from your family, you know",
           "Cult membership tends to entail the isolation of adherents from their families.",
           "A cult may isolate its members from their family environment."),
    "N2": ("honestly cults are just dangerous for the people who join, like",
           "Cult involvement presents significant risks for adherents.",
           "Cults present a danger to the persons who join them."),
    "N3": ("in a cult you just do what the leader says, no questions, you know",
           "Cult adherents exhibit unconditional obedience to the leader.",
           "Members of a cult are required to obey the leader without question."),
    "N4": ("they like push you to give them all your money, in a cult",
           "Cults exert financial pressure on their adherents.",
           "A cult may exert pressure on its members in order to obtain financial contributions."),
    "N5": ("i mean you can just leave a cult whenever you want, right",
           "Exit from cults is, in principle, voluntary for adherents.",
           "Any person is free to leave a cult at any time."),
    "N6": ("people just say cult to trash a group, like",
           "The label cult functions as a stigmatising designation.",
           "The designation of a group as a cult may constitute a form of stigmatisation."),
}


def build_rows() -> list[dict]:
    rows: list[dict] = []

    def add(contrast, domain, register, base_id, a, b, la="en", lb="en", notes=""):
        rows.append(dict(pair_id="", contrast=contrast, domain=domain, register=register, base_id=base_id,
                         sentence_a=a, sentence_b=b, lang_a=la, lang_b=lb, notes=notes, keep=""))

    for bid, d in BASES.items():
        dom, reg, base = d["domain"], d["register"], d["base"]
        add("0_near_identical", dom, reg, bid, base, d["near"], notes="upper anchor")
        add("1_paraphrase", dom, reg, bid, base, d["para"])
        add("2_opposite_stance", dom, reg, bid, base, d["opp"], notes=d.get("opp_note", ""))
        add("3_negation", dom, reg, bid, base, d["neg"], notes=d.get("neg_note", ""))
        fr_note = d.get("fr_note", "")
        if dom == "cult":
            fr_note = "; ".join(x for x in ["'secte' is not a neutral equivalent of 'cult'", fr_note] if x)
        add("4_translation", dom, reg, bid, base, d["fr"], "en", "fr", fr_note)
        add("6_unrelated", dom, reg, bid, base, d["unrel"], notes="lower anchor, same register")
        if d["sp"]:
            add("8_singular_plural", dom, reg, bid, base, d["sp"])

    for group in SAME_TOPIC_GROUPS:
        for i, bid in enumerate(group):
            other = group[(i + 1) % len(group)]
            d, o = BASES[bid], BASES[other]
            add("7_same_topic_different_claim", d["domain"], d["register"], f"{bid}|{other}", d["base"], o["base"],
                notes=f"{d['claim']} / {o['claim']}")

    for items, dom in ((GROUPS, "cult"), (CHOIRS, "choir")):
        for i, (name, desc) in enumerate(items):
            other_name, other_desc = items[(i + 1) % len(items)]
            add("5_name_vs_description", dom, "neutral", "", name, desc, notes="matched; fact-check the description")
            add("5_name_vs_description", dom, "neutral", "", name, other_desc,
                notes=f"mismatched control: description of {other_name}")

    for a, b, la, lb, dom in SINGLE_WORDS:
        add("8b_single_word", dom, "", "", a, b, la, lb, "out of distribution: bare words, weak")

    for frame in LABEL_FRAMES_EN:
        for label in LABELS_EN:
            article = frame
            if frame.startswith("A {}") and label[0] in "aeiou":
                article = "An" + frame[1:]
            add("9_label_swap", "choir" if label == "choir" else "cult", "neutral", "",
                frame.format("cult"), article.format(label), notes=f"label: cult -> {label}")
    for frame in LABEL_FRAMES_FR:
        for label in LABELS_FR:
            add("9_label_swap", "choir" if label == "chorale" else "cult", "neutral", "",
                frame.format("secte"), frame.format(label), "fr", "fr", f"label: secte -> {label}")

    for bid, (spoken, academic, admin) in REGISTER_SHIFT.items():
        base = BASES[bid]["base"]
        for reg, text in (("spoken", spoken), ("academic", academic), ("administrative", admin)):
            add("10_register_shift", "cult", reg, bid, base, text, notes=f"neutral -> {reg}")

    counters: Counter = Counter()
    prefix = {"0": "ni", "1": "pa", "2": "os", "3": "ne", "4": "tr", "5": "nd", "6": "un", "7": "st",
              "8_": "sp", "8b": "sw", "9": "ls", "10": "rs"}
    for r in rows:
        c = r["contrast"]
        key = "10" if c.startswith("10") else "8b" if c.startswith("8b") else "8_" if c.startswith("8_") else c[0]
        counters[key] += 1
        r["pair_id"] = f"{prefix[key]}{counters[key]:02d}"
    return rows


TEX_ESCAPES = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
               "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "\\": r"\textbackslash{}"}


def tex(s: str) -> str:
    return "".join(TEX_ESCAPES.get(ch, ch) for ch in s)


def write_tex(rows: list[dict]) -> None:
    lines = [
        "% Generated by thesis_corpus/build_calibration_set.py from notes/calibration_set.csv. Do not edit by hand.",
        r"\chapter{Calibration set}",
        r"\label{app:calibration}",
        "",
        r"{\footnotesize",
        r"\begin{longtable}{@{}p{0.07\textwidth}p{0.17\textwidth}p{0.36\textwidth}p{0.36\textwidth}@{}}",
        r"\caption{Calibration sentence pairs, by contrast type}\label{tab:calibration_set}\\",
        r"\toprule ID & Contrast & Sentence A & Sentence B \\ \midrule \endfirsthead",
        r"\toprule ID & Contrast & Sentence A & Sentence B \\ \midrule \endhead",
        r"\bottomrule \endfoot",
    ]
    for r in rows:
        contrast = r["contrast"].split("_", 1)[1].replace("_", " ")
        lines.append(f"{tex(r['pair_id'])} & {tex(contrast)} & {tex(r['sentence_a'])} & {tex(r['sentence_b'])} \\\\")
    lines += [r"\end{longtable}", "}", ""]
    TEX_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex-only", action="store_true", help="rebuild the appendix from the curated CSV")
    args = ap.parse_args()
    if args.tex_only:
        rows = list(csv.DictReader(CSV_PATH.open(encoding="utf-8")))
        rows = [r for r in rows if r.get("keep", "").strip().lower() not in ("n", "no")]
    else:
        rows = build_rows()
        with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows)
    write_tex(rows)
    for (c, d), n in sorted(Counter((r["contrast"], r["domain"]) for r in rows).items()):
        print(f"{c:32s} {d:6s} {n}")
    print("total", len(rows))


if __name__ == "__main__":
    main()
