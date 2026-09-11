TRANSCRIPTS EDITED — README

This folder holds cleaned-up, dialogue-formatted versions of the 2 raw files in
../Transcript-Raw, one edited file per interview:

- Sep 10 at 18-38.txt
- Sep 10 at 20-03.txt

ON MERGING
Same day, close-ish timestamps (18:38 and 20:03), but clearly two different
interviewees — different ages, genders, languages, and nationalities, and the
first is a French-language interview while the second is in English. No merging
was needed; both are treated as standalone interviews.

WHAT WAS DONE TO EACH FILE
1. Speaker turns: the Sep 10, 20:03 raw file has no diarization at all —
   interviewer and interviewee speech runs together in undifferentiated
   paragraphs, split here into "Interviewer:"/"Interviewee:" turns by reading for
   who's asking vs. answering (best-effort reconstruction, not verified). The
   Sep 10, 18:38 raw file is unusual for the project so far: it already came with
   a two-speaker ASR diarization and a built-in English translation alongside the
   French original — that existing split was used as the basis, cross-checked
   against the French original, with quick question-and-answer pairs that the tool
   had bundled into one speaker's turn separated out into proper turns.
2. Light cleanup: filler words, false starts, and obvious repeated words were
   smoothed for readability without changing meaning. Recurring, unambiguous ASR
   mishearings of "cult" (e.g. "Colts," "the levers are super abuse" for "the
   leaders are super abusive") were corrected silently, matching the pattern
   established in earlier batches.
3. Likely mistakes: less certain corrections are flagged inline with [?] and
   explained in each file's own "TRANSCRIPTION / EDITING NOTES" section. The most
   notable one this batch:
   - The interviewee in the Sep 10, 20:03 interview names a documentary herself
     as "Wild Wild Country" — a high-confidence match to the 2018 documentary
     series about the Rajneeshpuram commune in Oregon, led by Bhagwan Shree
     Rajneesh (Osho). A separate garbled phrase right after ("and the guy you're
     talking about this while") likely reflects the interviewer trying to supply
     the leader's name in response, but the actual words weren't recoverable.
   - Also in that interview, "by Paula" and "Mike Polo" are both almost certainly
     mishearings of "[my] polo" — the surrounding conversation is entirely about
     the interviewee's friend and the sport of polo — but the exact original
     phrasing is uncertain, so both are flagged with [?] rather than silently
     changed.
4. Demographics: both interviews had their demographic fields captured, though
   the Sep 10, 20:03 interviewee stated a compound main language (Spanish and
   English) and dual nationality (Mexican and United States) — both are
   reproduced as-is rather than picking one.

WHAT WASN'T DONE
- No audio was available to verify any of this — everything above is inferred
  from the raw text alone. Anywhere the notes flag uncertainty, treat it as worth
  double-checking against the original recordings if it matters for the thesis.
- Nothing was removed for content reasons.

NEXT STEP
These files are pending manual review. Once reviewed, move each into the
sibling "Transcripts Finished" folder (create it if needed), per the same
workflow used for Batches 1-3.
