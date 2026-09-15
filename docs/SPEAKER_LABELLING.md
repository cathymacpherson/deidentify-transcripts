# Speaker labelling

Assigns a role to each turn of a transcript — `C` for the client, `T` for the therapist — where a
human has not already done so. Built for corpora where turns are short ASR fragments, so a turn on
its own often carries no signal and only the surrounding conversation resolves it.

This document covers how it works, how to run it, what the output means, and what it cannot do.
The reasoning behind the design, and how it was measured, are at the end.

**It does not replace human review.** On held-out transcripts labels are roughly 91% accurate, and
about a third of the errors carry no warning. The output is built to make review efficient, not to
avoid it.

Results measured on one specific corpus live in that project's own notes, not here — this document
holds nothing dataset-specific.

---

## How it works

Two passes over the transcript, both using the configured language model, followed by a
confidence calculation that decides what a human should look at.

### Pass 1 — find the certain turns

The model reads the transcript and marks **only turns whose speaker is beyond doubt**, skipping
everything else. Low recall is the goal: it may mark one turn in ten, or one in two, depending on
the transcript.

A turn such as "we're coming to the end of our time today" is unmistakably the therapist. A long
first-person account of someone's week is unmistakably the client. A short "yeah" is neither, and
is left alone.

Two jobs:

- **Fix which role is which.** Labelling an entire transcript backwards is easy to do, internally
  consistent, and hard to spot on a skim. A handful of unmistakable turns rules it out.
- **Constrain Pass 2**, which then fills gaps between known values rather than labelling from
  nothing.

A returned turn number that is not in the transcript is discarded rather than trusted — a wrong
anchor propagates through everything built on it.

Turns already labelled by a human are added to this set directly. They are better than anything the
model can infer, which is what makes a partially coded transcript the *easiest* case rather than an
awkward one.

### Pass 2 — label everything, several times

The model labels every turn in a window of consecutive turns, with anchors marked `FIXED` and not
to be changed. Windows overlap heavily, so **each turn is labelled three or four times** against
different surrounding context.

```text
  [412] T FIXED: And how did that land for you?
  [413] ?: I don't know
  [414] ?: yeah
  [415] ?: I think I just shut down
```

Turn 414 alone is unlabellable. In this window it is plainly part of the client's answer. This is
the pass that finds speaker changes, which no per-turn feature can.

Two details that matter in practice:

- **Edge coverage.** Plain sliding windows leave the first and last turns in a single window, with
  no second opinion and therefore no confidence. Shorter windows anchored at each end fix this. The
  opening of a session is where third-party and framing turns appear, so it is the wrong place to
  have no signal.
- **A label for a turn outside the window is discarded**, and a turn labelled twice in one window
  counts once. Silently accepting either would corrupt a turn the model was not looking at.

The model may answer `unclear`. Some turns genuinely cannot be settled from text — see
[The information ceiling](#the-information-ceiling) — and saying so is better than guessing.

#### Sizing a window

At 40 turns a window is roughly 900 input tokens and 280 output — well under what a served model
can hold. Larger windows give more conversational context *and* cost fewer calls:

| Turns | ~tokens | Calls per 700-turn transcript |
|---|---|---|
| 40 | ~1,200 | 56 |
| 120 | ~3,000 | 20 |
| 200 | ~4,800 | 13 |

The limit is output reliability, not context: at 200 turns the model must return 200 correctly
numbered JSON objects, and long structured outputs are where models drift — skipping turns,
renumbering, losing track near the end. A window returning fewer labels than it was given costs
those turns a vote and degrades confidence with no error raised.

`--window` and `--step` are exposed so this can be settled by measurement. Keep turns-per-view
constant when comparing sizes, so the comparison is fair.

### Confidence: where the votes disagree

Where every window agrees, the turn is confident; where they split, it is flagged. This measures
the model's **stability under changed context** rather than asking it to rate itself, which models
do poorly.

#### What the confidence number means

Confidence is **not continuous**. It is a fraction of votes, so it takes about eight discrete
values — which is why a summary of it produces neat rows rather than a spread.

Each turn is seen by three or four windows, so agreement can only be 3/3, 2/3, 4/4, 3/4 and so on.
That figure is then multiplied by 0.85 where the second opinion disagrees.

| Value | How it arises |
|---|---|
| 1.000 | Every view agreed, and the second opinion agreed too — or the turn was labelled by hand |
| 0.850 | Every view agreed, but the second opinion disagreed |
| 0.750 | 3 of 4 views agreed |
| 0.667 | 2 of 3 views agreed |
| 0.637 | 3 of 4 agreed, and the second opinion disagreed |
| 0.567 | 2 of 3 agreed, and the second opinion disagreed |
| 0.000 | No usable view, or a label kept verbatim for a human to confirm |

The number has no meaning as a probability. It is a **rank**: turns with lower values were wrong
more often when measured. How much more often is a property of the corpus — measure it with
`label-summary --calibration` rather than assuming.

#### Choosing the flag threshold

`label --flag-threshold` sets the value below which a turn is listed for review. Because confidence
is discrete, **only a few thresholds change anything** — anything between two adjacent values gives
an identical result:

| Threshold | Lists |
|---|---|
| `0.999` (default) | everything except perfect agreement |
| `0.8` | drops turns only the second opinion disputed |
| `0.7` | also drops 3-of-4 agreement |
| `0.6` | only the least confident turns |

It is a display choice, not a labelling one. Confidence is stored per turn either way, so a
different threshold changes which turns are listed and nothing else.

**A longer list is not automatically better.** Where a quarter of every transcript is flagged, the
flag stops commanding attention — a reviewer who meets seven correct turns in a row begins skimming
the flags too. A short list that is right about half the time is worth more than a long one that is
right one time in five. `label-summary --calibration` gives the yield at each level, which is the
basis for choosing.

#### The two labellers

Confidence comes from comparing two systems that work in completely different ways.

| | The LLM | The second opinion |
|---|---|---|
| Runs on | The configured inference server | The local machine — no server, no network |
| How it decides | Reads the surrounding conversation and judges | Counts word patterns: question marks, first- and second-person density, turn length, neighbouring turns, how often a speaker holds the floor |
| Typical accuracy | ~90% | ~70% |
| Speed | Minutes per transcript | Instant |
| Implementation | [labelling.py](../src/deidentify_transcripts/labelling.py) | [supervised.py](../src/deidentify_transcripts/supervised.py) |

**The LLM assigns every label. The second opinion never changes one** — it is much weaker, and
letting it overrule would trade accuracy for confidence.

**The second opinion is optional.** It has to learn from transcripts someone has already checked,
so a corpus with none runs without it. Labelling and accuracy are unaffected; only confidence is
weaker, since it then rests on the model disagreeing with itself — which catches turns it is unsure
about but not ones it is consistently wrong about. Expect roughly half as many real errors to be
flagged. This is the situation every new project starts in, and it improves as soon as a few
transcripts have been reviewed and `--reference` is pointed at them.

It exists purely to disagree. Because it reaches its answers another way, it is wrong about
different turns, so a disagreement is evidence neither system can produce alone.

The second opinion is *trained*: it reads the already-labelled transcripts and works out from the
human labels how much each pattern is worth. It is the only part of this system that learns from
the corpus — the LLM never does.

Training takes about four seconds on a corpus of ~80,000 turns, so it is **redone on every run
rather than cached**. There is no model file to go stale, no cache to invalidate, and newly
labelled transcripts are picked up automatically.

#### Self-consistency is not enough on its own

Asking one model the same question several times measures **instability, not correctness**. Where
a model is systematically wrong — the same mistake from every angle — its votes agree and the turn
looks confident.

This is not a marginal effect. In testing, around a third of all errors sat at unanimous
agreement, and across hundreds of anchored turns the window pass never once contradicted the anchor
pass. Two passes of the same model at temperature 0, over overlapping context, are not independent
observers of each other.

**The fix is a genuinely different system, not more samples.** A second labeller that reaches its
answers another way — the trained model in
[supervised.py](../src/deidentify_transcripts/supervised.py) — fails on different turns. Measured
on the same corpus: where the two agreed, 5.3% of turns were wrong; where they disagreed, 20%.
Flagging every disagreement cost 24% of turns and caught 54% of errors, a signal no amount of
self-consistency produces.

Two rules for using it:

- **The second opinion never changes a label.** It only adjusts how suspicious a turn looks. A
  weaker system overruling a stronger one trades accuracy for confidence.
- **When scoring, it must not have seen the speaker it is judging.** Gold labels are present, so a
  second system trained on that speaker agrees for the wrong reason and the disagreement signal is
  worth less than it appears. `label-eval` therefore retrains per fold, excluding each
  transcript's own speaker, which is what its `--mapping` is for.

  **When labelling for real this does not apply.** There are no gold labels to leak, so the second
  opinion is trained once over every labelled transcript available — more training data, a stronger
  cross-check, and one training run instead of many. `label` needs no mapping; point `--reference`
  at the labelled transcripts, or pass `none` to skip the second opinion entirely.

Raising the sampling temperature or varying the window size decorrelates votes far more weakly than
this, and was not worth the cost when measured.

### Label vocabulary

`unknown` in the source data means *no speaker was recorded* — it is the default
`load_transcript()` assigns to a turn with no speaker field, not a judgement anyone made. The
output vocabulary keeps that distinct from a determination the labeller made and failed:

| Label | Meaning |
|---|---|
| `C` | Client |
| `T` | Therapist |
| `other` | A speaker present in the room who is neither client nor therapist |
| `uncertain` | The labeller considered this turn and could not determine the speaker; routed to review |

Every input `unknown` becomes one of these four, so `unknown` never appears in output. Keeping
`uncertain` separate preserves the difference between "never processed" and "processed, could not
tell", which the review queue needs and which a single reused label would destroy.

#### `other` is proposed, not observed

No third-party label has been confirmed in any corpus this specification has been applied to. It is
a class the design proposes, and it cannot be learned or measured until labelled instances exist.

Implement it as a **precision-oriented anchor rule** — session-framing utterances addressed to the
room from outside the dyad, such as a pre- or post-session handover — that routes to review rather
than asserting a label, and do not report recall for it. If a corpus turns out to have its own
third-party convention, record that in the project's dataset notes and extend the vocabulary.

#### Existing labels are never modified

Three categories, not two. The distinction matters: the third was silently overwritten in an early
implementation, discarding human judgements on exactly the turns a person had found hardest.

| Existing value | Behaviour |
|---|---|
| A known role (`C`, `T`) | Kept, **used as an anchor**, confidence 1.0, never shown in the review report |
| A recognised blank (`unknown`, empty, `n/a`, `none`, `-`) | Filled by the labeller — this is the job |
| Anything else (`C/T`, `X`, a mistyped label) | **Kept verbatim**, confidence 0.0, sorted to the top of the review report |

The third row is the one worth defending. A merged-speaker mark records that a human decided the
turn covers both speakers; replacing it with a single label silently resolves something they marked
as unresolvable. An unrecognised convention means something to whoever wrote it. The system cannot
know what — but it can make sure a person is asked, which is what confidence 0.0 achieves.

A partially coded transcript is therefore the **easiest** case, not an awkward one: its manual
labels are verified anchors, strictly better than anchors the model guesses at.

**Evaluation must switch this off.** Gold labels live in the same `speaker` field, so a labeller
that reads existing labels is handed its own answers and reports near-perfect accuracy that means
nothing. `label_transcript(..., use_existing=False)` is mandatory for scoring.

- `--relabel` is the explicit escape hatch for regenerating labels over existing ones. It is never
  the default.

---

## Using it

### Step 1 — sort transcripts by what labels they already have

`deidentify-transcripts inventory` sorts a directory of transcripts into `labelled/`,
`unlabelled/` and `review/` according to the speaker labels they already carry. It is a
deterministic metadata check — it reads only the `speaker` field of each turn. No model, no
network, no transcript text is read, printed or written.

```bash
deidentify-transcripts inventory data/                    # dry run: counts only
deidentify-transcripts inventory data/ --apply            # actually move the files
deidentify-transcripts inventory data/ --manifest m.csv   # per-file detail, written locally
```

- **Dry run by default.** Nothing moves without `--apply`.
- **Terminal output is aggregate only** — counts, never filenames or content. Per-file detail goes
  to `--manifest`, which contains filenames and so must stay outside version control.
- `--threshold` (default 0.5) sets the proportion of turns needing a `C`/`T` label for a transcript
  to count as labelled. This is the completeness threshold discussed in
  [Uneven labelling constrains the supervised path only](#uneven-labelling-constrains-the-supervised-path-only);
  whatever value is used should be recorded alongside any results derived from the split.

Four buckets, each naming what happens to the transcript next:

| Bucket | Meaning | Next step |
|---|---|---|
| `labelled/` | Role labels on at least `--threshold` of turns | Nothing, though it may still have gaps |
| `partial/` | Some role labels, below threshold | Label the gaps; keep every human label |
| `unlabelled/` | No role labels at all | Label from scratch |
| `review/` | Unrecognised speaker vocabulary, or unreadable | A human decides |

**`partial/` is a processing input, not an error state.** A partially coded transcript keeps all of
its human labels and has only its `unknown` turns filled, per
[Existing labels are never modified](#existing-labels-are-never-modified). It is also the *easiest*
case for the labeller rather than the most awkward: the existing manual labels are verified anchors,
supplying for free what [Pass 1](#pass-1--find-the-certain-turns) otherwise has to infer, and constraining
[Pass 2](#pass-2--label-everything-several-times) directly. Expect higher accuracy on these than on wholly unlabelled
transcripts.

**A few odd turns do not block a file.** A mistyped label, or a turn marked as covering both
speakers, is a turn-level problem handled by the review queue. Only when such turns exceed
`--anomaly-tolerance` (default 2%) is the file itself treated as having a vocabulary problem and
sent to `review/`. Without this, a single typo in a thousand-turn transcript would block the whole
file — the opposite of useful.

**Clearing the threshold does not mean having no gaps.** A transcript can sit in `labelled/` and
still contain `unknown` turns; the command reports how many files and turns this affects. Those
files are valid inputs to the labeller too, on the same never-overwrite terms.

A transcript whose speaker column uses a different vocabulary — interviewer/participant naming, say
— is *labelled*, just not in this one. It goes to `review/` rather than being filed as unlabelled,
because treating it as unlabelled would send genuine human labels to be overwritten by the
labeller. Recognised role spellings are listed in
[inventory.py](../src/deidentify_transcripts/inventory.py); extend them there rather than
reclassifying files by hand.

Discovery is non-recursive, so re-running the command does not re-sort files already filed.

### Step 2 — label

`deidentify-transcripts label <path>` runs both passes and writes, per transcript:

- `output/labelled/<id>.json` — the transcript with `speaker`, `speaker_confidence` and
  `speaker_source` (`manual` or `model`) per turn;
- `output/review/<id>.review.json` — the reviewer's report.

Point it at the transcripts that need labels, not at already-labelled ones.

### Step 3 — work through the review list

`output/review/<id>.review.json` lists the turns the labeller is unsure about:

```json
{
  "transcript_id": "...",
  "turns_total": 1146,
  "turns_flagged": 283,
  "note": "A priority list, not a filter...",
  "items": [
    {
      "turn_id": 412,
      "speaker": "C",
      "confidence": 0.667,
      "reason": "model disagreed with itself across views",
      "votes": ["C", "T", "T"],
      "text": "yeah"
    }
  ]
}
```

Items are ordered most suspicious first — by how error-dense their reason was in testing, then by
confidence. Consecutive `turn_id` values indicate a run flagged together, which is usually one
correction rather than several.

Surrounding context is deliberately not included: the reviewer has the transcript, and `turn_id`
locates the turn in it. Duplicating context would make the file large without adding anything.

Turns labelled by hand never appear. Labels the tool does not recognise — a merged `C/T` mark, or
a local convention — are kept exactly as written and listed first, because only a person can say
what they mean.

**It is a priority list, not a filter**, and the `note` field says so. Errors also occur in turns
the system was confident about, and those are not listed, so the whole transcript still needs
reading.

### Triage: judge on lift, not correlation

Per-file error rates vary several-fold, so it is worth asking whether a transcript's own flag rate
predicts how wrong it is. `label-diagnose` reports both a rank correlation and the **lift** — how
much more error the worst-looking files carry than their share of turns.

Judge on lift. A high correlation with low lift means the ordering is right but the files barely
differ, so routing on it saves little. Measured on one corpus: rho +0.66 with only 1.3× lift —
a real effect, too weak to justify a separate manual-coding workflow.

---

## What it cannot do

### Merged-speaker turns

Some manually coded turns may carry a combined label such as `C/T`, marking a segment where the
transcription ran an exchange between both speakers into a single turn. `speaker-audit` reports how
often this occurs.

Handling depends on the rate, which is a property of the corpus:

**Rare (a few turns).** Handle per turn: route to the review queue for a human to split or relabel,
never assign a single role label, and **exclude from evaluation as ground truth** — a merged label
is not something the system can be scored against producing, so scoring against it would penalise
correct behaviour. `--anomaly-tolerance` keeps a handful from blocking an otherwise processable
file.

**Common (more than a few percent).** The one-turn-one-speaker assumption no longer holds and the
labeller needs a sub-turn split: detect the speaker change within a turn and emit two turns. This
is a real extension — it breaks the 1:1 mapping between input rows and output turns that anything
joining on `turn_id` depends on — and should not be built speculatively.

Run `speaker-audit` on any new corpus before labelling it, and record the rate and the resulting
decision in that project's dataset notes.

### The information ceiling

Establish **what information the human annotator had** before setting an accuracy target. Where
labels were assigned from an original recording, the annotator heard the voices while the system
sees only text. Those labels are then reliable ground truth *and* partly underivable: a two-word
acknowledgement is trivial by voice and may be genuinely indeterminate in the transcript.

This is a harder ceiling than annotation noise, and no prompt or model closes it. The correct
response is not to tune towards it but to detect it: a turn the text cannot settle should be
flagged, not guessed at. Reported as `unclear` rather than counted as an error.

Where a second annotator is available, double-coding a few hundred turns (in contiguous chunks, so
the coder has context) measures the ceiling directly. Where one is not, a cheaper substitute is to
collect turns where two independent systems confidently agree with each other and both contradict
the gold label; a human check of that small set estimates the label error rate. It finds only
errors the systems happen to catch, so it is a floor on the error rate, not a true ceiling.

---

## Why it is built this way

### Design decisions

#### Not an agentic system

A staged pipeline, not a self-directed agent loop.

The task has a known, fixed shape, and this repository has deliberately optimised for
reproducibility and auditability: a deterministic multi-pass structure (detect → propagate →
independent gate → review queue) where every intermediate is inspectable and `temperature=0` makes
runs repeatable. An agent with free tool use would trade exactly those properties away for
flexibility this problem does not need.

What the task *does* need is several passes with distinct jobs — which is staging, not agency.

#### Labelled data enters through evaluation, not the prompt

Multiple clients and clinicians are mixed across the labelled and unlabelled sets. This makes
few-shot exemplars drawn from other dyads actively risky: what marks a line as `C` is partly role
and partly *that individual's* speech style, so exemplars teach the model to match one client's
voice, which misleads it on the next transcript. The failure is silent — nothing in the output
shows it happened.

Labelled data is therefore spent on **held-out evaluation and threshold calibration**, where it has
no leakage risk and high value. The model already knows what a therapist sounds like; it does not
know this project's acceptable error rate.

Few-shot prompting is retained as an **off-by-default ablation arm** (`--few-shot`) so the question
can be settled by measurement rather than by argument. Exemplars, if used, must come from a
fold disjoint from the transcript being labelled.

#### Each transcript is its own training set

The problem factors into two sub-problems of very different difficulty:

1. **Segmentation** — which lines belong to the same speaker's run. Local, and needs no labels.
2. **Role assignment** — which of the resulting clusters is the therapist. Needs only a handful of
   unmistakable lines per transcript (`"you signed a consent form for therapy"`,
   `"we're coming to the end of our time"`).

Because role assignment needs only a few in-transcript anchors, a corpus with zero labelled
transcripts works as well as a labelled one. This is what makes the required flexibility possible,
and it is why labelled data is not an input to the method.

### Architecture

Four passes, mirroring the structure of the de-identification pipeline.

#### Deterministic components vs. server LLM

The system is a **deterministic skeleton with LLM passes slotted in only where judgement is
required** — the same division the de-identification pipeline already uses, where regex patterns
handle what is decidable and the model handles what is not.

| Component | Deterministic | Server LLM |
|---|---|---|
| Evaluation harness, splits, metrics | ✅ | |
| Baselines 1–2 (majority, rule-based) | ✅ | |
| Baseline 3 (supervised classifier) | ✅ (needs labelled training data) | |
| Pass 1 — anchor detection | partly (lexical cues) | ✅ |
| Pass 2 — window fill | | ✅ |
| Label propagation and run smoothing | ✅ | |
| Pass 3 — coherence gate | partly (run-length outliers, flip check) | ✅ |
| Pass 4 — review queue, thresholds | ✅ | |

**Which path ships is deliberately not decided yet.** Phase 0 exists to answer it, and the answer
may differ by deployment:

- **A corpus with labelled data** (this one) may be served best by the deterministic supervised
  path: milliseconds per transcript, no server, no Tailscale, perfectly reproducible, and no
  transcript text leaving the machine at all. Given how strongly the measured features separate the
  two roles, this could be competitive.
- **A corpus with no labelled data** cannot use the supervised path at all — there is nothing to
  train on. The zero-shot LLM path is the only option, and it is what makes the stated flexibility
  requirement achievable.

These are not competing designs to choose between; they are two paths behind one interface. Both
produce the same output contract (label, confidence, provenance, review queue), are measured by the
same harness, and should be selectable at run time. Build the deterministic path first because it
is cheap and sets the bar; build the LLM path because it is the only one that generalises to a new
corpus.

### Two passes that were planned and not built

**A coherence gate** was specified: an independent pass re-reading the labelled dialogue for
sequences that do not cohere — a question and its answer given to the same speaker, a therapist
making a sustained personal disclosure, a run far outside the transcript's own distribution.

Only its **global flip check** was built, as part of scoring. The rest was dropped because an
independent second labeller addressed the same need more directly: it catches systematic error,
which is what the gate was for, and it does so without a second set of prompts to maintain. The
idea remains sound if a second labeller is ever unavailable.

**A separate review queue file** in the de-identification pipeline's format was specified —
`sensitive/<id>.speaker-review.jsonl`, `needs_review` status, exit code `2`. The per-transcript
review report replaced it. A reviewer needs flagged turns *in conversational context*, which a
flat queue of items cannot give, and the report can be read directly rather than joined back
against the transcript.

---

## Measuring it

### Measure the corpus before designing

The design below rests on properties of the transcripts, not on assumptions about them. Measure
these on any new corpus before applying this specification; several decisions flip depending on
the answers.

| Measure | Why it matters |
|---|---|
| Turn granularity — are turns conversational, or ASR fragments? | Fragmented turns carry no role signal individually and cannot be classified independently |
| Speaker-change rate per boundary | A low rate means a strong sequential prior that any per-line approach discards |
| Run-length distribution per role | Sets the window size, and gives the coherence gate its outlier test |
| Class balance | Determines how misleading raw accuracy will be |
| Majority-class accuracy | The floor every approach must clear |
| Inter-turn timestamp gaps at changes vs non-changes | Decides whether timing is usable signal or a segmentation artifact |
| Separation on cheap lexical features (question marks, first-person density) | Sets the rule-based baseline, and indicates whether a supervised model is viable |

Two findings recur and are worth expecting:

**Timestamps are often useless.** In ASR output the "timestamps" may be segmentation boundaries
rather than voice-activity edges, in which case gaps at speaker changes look identical to gaps
mid-utterance. Verify before building on them.

**Accuracy is usually a misleading metric.** With one role dominating, labelling everything as that
role scores high while being worthless. Never report headline accuracy alone — see
[Metrics](#metrics).

This project's own measurements are in
[project/DATASET_NOTES.md](../project/DATASET_NOTES.md).

### Evaluation

**Build this before any model code.** Until the baselines below are measured, there is no way to
tell whether an LLM pass is earning its keep.

#### Splits

Speakers recur across a corpus, so a random split will place the same person on both sides and
report a score that measures memorisation of that person's speech style rather than generalisation.
Splits must therefore be built over **speakers, not transcripts**.

Where each participant is associated with exactly one counterpart, the participant–counterpart
graph is a set of disjoint stars and a split that is clean on both is simply a partition of the
counterparts — nothing needs to be discarded. Where participants share counterparts, the graph has
larger connected components and a clean split must partition those instead, discarding whatever
bridges them. **Compute the graph's shape first**; it determines which splits are achievable.

Prefer **grouped k-fold over the larger grouping** to leave-one-out:

- assign whole groups to folds, greedily balancing member count per fold;
- never split a group across folds — that reintroduces the leakage the design exists to prevent;
- all sessions belonging to one participant follow that participant into the same fold, which also
  keeps multi-session and multi-file participants intact automatically.

Group sizes are usually skewed. Where one group can dominate whichever fold holds it, **report
per-fold scores, not only the mean** — a mean that hides one fold at chance level is worse than no
number.

Joining an identity mapping to transcript files usually requires normalising identifiers, since
mapping keys and filenames rarely share a format. Normalise both sides before joining, and fail
loudly on any transcript that matches no mapping row rather than silently dropping it.

#### Uneven labelling constrains the supervised path only

Manual labelling is rarely distributed evenly across a corpus: some participants have every session
labelled, others none. The two paths are affected very differently.

**Supervised path.** Labelled sessions are the entire usable dataset — unlabelled sessions serve
neither as training nor as test data. Folds can only be built over groups having at least one
labelled session, so `k` is bounded by *that* count, not by the size of the corpus. If the count is
small, cross-group generalisation cannot be measured honestly at any fold size, and the result
should be reported as indicative rather than as a generalisation estimate.

**Zero-shot LLM path.** Unaffected. There is no training set, so there is nothing to split and no
leakage to prevent; every labelled session is valid test data no matter how labelling clusters. This
path does not require a held-out speaker to exist.

Per-group scores should still be reported for the zero-shot path — not to prevent leakage, but to
detect whether accuracy varies by speaker, which a single pooled number would hide.

Two things to check before trusting any score, and to state in any write-up:

- how many distinct groups have labelled sessions — the real upper bound on `k`;
- whether the labelled subset is representative, or whether labelling followed convenience (one
  caseload, one annotator, earliest sessions first). A non-random labelled subset biases both the
  supervised model and every score computed from it, in a direction nothing in the output reveals.

#### Leakage discipline differs per arm

| Arm | Leakage vector | Defence |
|---|---|---|
| Supervised baseline | Training on the same dyad | Strict dyad-disjoint folds |
| Few-shot LLM | Exemplars from the same dyad | Exemplars drawn from a disjoint fold |
| Zero-shot LLM | **None from training** — the model never saw this data | Prompt iteration on dev only; locked test set |

The zero-shot arm's only leakage vector is us: iterating on prompts while watching test scores is
how a test set gets burned. All prompt iteration happens on dev; the test set is opened when we
believe we are finished.

This asymmetry means the zero-shot arm can use nearly all labelled data for honest measurement
while the supervised arm pays the full split cost. If the supervised baseline proves competitive,
that cost is part of its price.

#### Metrics

**When a human reviews everything anyway, accuracy is the wrong headline.** What matters is how
much work the reviewer does and whether the system's mistakes are findable. A system at 95%
accuracy that flags 95% of its own errors is better than one at 98% that flags none — the second
buries its mistakes among confident correct labels where nobody will look.

Reviewer-facing, in priority order:

- **Error capture** — of the turns the system got wrong, what share fall inside the flagged set?
  This is the number to optimise. A system whose confidence is informative concentrates its errors
  in a small flagged set; one whose confidence is noise scatters them.
- **Errors left unflagged** — mistakes the reviewer must catch unaided. The only genuinely
  expensive failure, and the number a target like "99%" should be expressed against.
- **Edit actions** — contiguous blocks of wrong labels, not wrong turns. A whole run mislabelled
  in one block is one correction; the same count scattered is many. Two systems with identical
  accuracy can differ several-fold in what they cost to fix.

System-facing, for comparing approaches:

- **Macro-F1** — the comparison number, robust to class imbalance.
- **Boundary F1** — speaker-change detection, independent of role assignment. Usually the weakest
  number and the one that drags everything else down.
- **Flip rate** — transcripts with globally inverted roles. Catastrophic and invisible in a pooled
  score, so checked separately.
- **Accuracy** — reported only alongside the majority-class figure, which shows how much of it is
  free.


### Baselines

Every one of these is cheap, and each must be beaten to justify the next step up:

1. Majority class — the floor, and usually a high one given the class imbalance.
2. Rule-based: question marks, first-person density, line length.
3. Supervised classifier on lexical and positional features, including neighbouring labels.
   With dozens of labelled transcripts this is trainable, free to run, fully deterministic, and
   needs no server. If it approaches the LLM's score, the cost/benefit of the LLM design changes
   substantially — no Tailscale dependency, instant, reproducible.

`deidentify-transcripts evaluate data/labelled` scores the deterministic baselines against manually
labelled transcripts. It runs entirely offline — no model, no network — and prints aggregate
figures only.

It reports accuracy, macro-F1, per-role F1 and boundary F1 side by side for each baseline, then the
bar to beat, then the review-burden curve. Read them together:

- **accuracy** will look high and mean little wherever one role dominates;
- **macro-F1** is the comparison number;
- **boundary F1** isolates segmentation quality from role assignment — a system can group turns
  perfectly and still assign both roles backwards, which is reported separately as a flip warning;
- the **review-burden curve** shows accuracy against how many turns a human must check. A flat
  curve means the system's confidence is not informative: abstaining on its least-confident turns
  does not buy accuracy, so the confidence signal needs work before any threshold is set.

### The trained model, as a second opinion

For evaluation, `--mapping` adds a trained sequence labeller
([supervised.py](../src/deidentify_transcripts/supervised.py)) to the comparison. Unlike the
baselines it **learns from the data**, so it is scored only on speaker-disjoint folds — without
`--mapping` it is skipped rather than scored dishonestly.

Two learned parts:

- **Emissions** — logistic regression over per-turn features: question form, first/second-person
  and plural rates, reflective and process-talk phrases, length buckets, position in the
  transcript, and the same features for the two turns either side. Sparse and named, so the
  learned weights can be printed and read.
- **Transitions** — how often each role follows each role, learned from the corpus. This replaces
  the rule baseline's hand-set smoothing constant with the actual run-length structure, and is
  aimed directly at boundary detection.

Viterbi decoding picks the best whole-sequence path; forward-backward marginals supply per-turn
confidence, so the review queue has a real probability rather than a hand-tuned score. A turn whose
gold label is neither role breaks the transition chain rather than inventing a transition that
never occurred.

Implemented without third-party dependencies, and reported per fold as well as pooled — a wide
per-fold spread means performance depends on which speakers are held out, which a pooled figure
hides.

### Scoring the LLM passes

`deidentify-transcripts label-eval data/labelled --limit 1` runs
[Pass 1](#pass-1--find-the-certain-turns) and [Pass 2](#pass-2--label-everything-several-times) over labelled transcripts and scores
them the same way. Unlike every other command here, **it sends transcript text to the configured
server**, so start with one transcript: it reports its own per-transcript timing before you commit
to a corpus.

It reports the system-facing scores, then the reviewer view — error capture at each flag budget and
how many mistakes are left unflagged — so the result is directly comparable with the baselines and
the trained model.

There is no training-set leakage to manage: the model never saw this data, so every labelled turn
is valid test data. The live leakage vector is prompt iteration against a visible score, so tune on
a development subset and keep a locked test set for the end.

Supporting modules: [metrics.py](../src/deidentify_transcripts/metrics.py) for scoring,
[splits.py](../src/deidentify_transcripts/splits.py) for leakage-free grouped k-fold over an
identity mapping, [baselines.py](../src/deidentify_transcripts/baselines.py) for the baselines
themselves.

---

## Open questions

- Does the corpus's manual labelling convention have a marker for "present but deliberately not one
  of the two roles"? If so, it is a protected human label and must not be overwritten. If `unknown`
  is being used for that purpose, the "input `unknown` is in scope" rule above is unsafe as written
  and needs revisiting before implementation.
- Does any transcript involve more than two speakers in the coded roles (supervision, handover)?
  This breaks the disjoint-stars property the split design depends on.
