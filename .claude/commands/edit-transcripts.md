---
description: Merge, diarize, and clean up raw interview transcripts into a "Transcripts Edited" folder
---

Process the raw interview transcripts in this folder: $ARGUMENTS

If no folder was given, ask the user which folder to process before doing anything else.

## Context

These are auto-transcribed (ASR) interviews for a thesis on cults. The interviewer
(the user) asks a short set of demographic questions, then one main question ("when
you hear the word cult, what comes to your mind?") with follow-ups. The raw
transcripts have **no speaker labels** — interviewer and interviewee speech is run
together in undifferentiated paragraphs — and contain typical ASR errors (garbled
words, misheard names, and a recurring mishearing of "cult" as "cold" / "call" /
"color" / "college" / "clues" / etc.).

## What to do

1. **List the raw files** in the given folder and read every one of them in full
   before writing anything.

2. **Detect split recordings.** Some interviews were recorded in two takes (either
   as separate files, e.g. named with a "1-2"/"2-2" suffix, or as one raw file with
   an internal gap where the recording clearly restarts). Use content and timing to
   decide what belongs together:
   - If two files/segments are close in time and the content clearly continues
     (same topic, no re-introduction, references that only make sense as a
     continuation), treat them as one interview.
   - If a file's name implies a missing other half but no matching part exists
     anywhere in the folder, and the recording reaches a natural close, treat it as
     a complete standalone interview rather than an incomplete fragment.
   - If genuinely ambiguous, say so in that file's notes rather than guessing
     silently.

3. **Diarize into speaker turns.** Split each interview into `Interviewer:` /
   `Interviewee:` turns by reading for who's asking vs. answering. This is
   inherently a best-effort reconstruction (the raw text has zero diarization) —
   flag any genuinely uncertain turn boundaries rather than presenting guesses as
   fact.

4. **Light cleanup, not rewriting.** Smooth filler words, false starts, and
   obviously repeated words for readability without changing meaning or paraphrasing
   away real content. Correct recurring, unambiguous ASR errors silently (the
   cult/cold/call/color/college pattern above). Leave genuinely uncertain words or
   phrases in place, flagged inline with `[?]`.

5. **Flag likely mistakes, don't silently invent them.** Where a name, place, or
   term is probably mistranscribed, use context (and general knowledge / a quick
   web search if it would nail down a real person, event, or documentary) to
   propose a correction — but always explain the reasoning in the notes section
   rather than just swapping the word in silently, unless the correction is
   completely unambiguous (like the recurring cult/cold pattern).

6. **Capture demographics in a header.** Each output file should open with the
   interviewee's stated age, gender, main language, language spoken, country of
   residence, and nationality, exactly as captured in that interview's raw text.
   If any are missing or illegible in the raw transcript, say so explicitly rather
   than leaving them out silently — don't invent them.
   - **Check for an existing edited file first.** If an edited output file for this
     interview already exists and its header contains demographic info that isn't
     in the raw transcript (e.g. the user filled in an age by hand), treat that as
     ground truth from the user and preserve it — don't overwrite it with "not
     captured."

7. **Format each output file** consistently:
   ```
   INTERVIEW — [Date], [time]

   Interviewee: [demographics]

   NOTE ON METHOD: [1-3 sentences — no speaker labels in the raw transcript,
   turns are reconstructed, light cleanup applied, uncertain spots flagged inline
   with [?]. Mention here if this file merges two takes/recordings.]

   ---

   Interviewer: ...

   Interviewee: ...

   [... full conversation ...]

   ---

   TRANSCRIPTION / EDITING NOTES
   - [each flagged correction/uncertainty, with reasoning]
   ```

8. **Output location.** Write one file per interview to a `Transcripts Edited`
   folder — a sibling of the raw folder (create it if it doesn't exist). Name each
   output file after its source recording (same date/time naming).

9. **Write or update a `README.txt`** in the `Transcripts Edited` folder listing
   the files present, briefly describing the method (steps 3-6 above, condensed),
   and calling out the most notable/confident corrections across the batch (real
   names, events, or documentaries identified from context).

10. **Report back concisely**: how many interviews were produced, which raw files
    got merged into which output file, and a short list of the most notable
    flagged corrections — not a full transcript dump.
