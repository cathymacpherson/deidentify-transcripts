# Dataset notes — therapy session corpus

Findings, measurements and decisions that apply to **this dataset only**. Everything here was
derived from the corpus in `data/`; none of it generalises, and none of it should be read as
describing how the tool behaves.

The general design — architecture, evaluation methodology, label vocabulary, the `inventory` and
`speaker-audit` commands — is in [docs/SPEAKER_LABELLING.md](../docs/SPEAKER_LABELLING.md) and
carries no assumptions from this dataset. This file supplies the numbers that document asks you to
measure for your own corpus.

Per [CLAUDE.md](../CLAUDE.md), no transcript content, filename or participant identifier appears
here. Statistics are aggregate and approximate.

## What the corpus actually looks like

Aggregate measurements from the working corpus. No transcript is identified here and no transcript
content is reproduced — see [CLAUDE.md](../CLAUDE.md). Statistics are approximate.

**Measured across the labelled set** (119 files, ~83,000 turns with a usable `C`/`T` label; ~249
turns carry no usable gold label, which is ~0.3%):

| Property | Value |
|---|---|
| Class balance | ~62% client, ~38% therapist |
| Majority-class accuracy ("everything is C") | **~62%** |
| Rule baseline, macro-F1 | 0.673 (accuracy 0.713) |
| Rule baseline, per role | C-F1 0.787, T-F1 0.558 |
| Rule baseline, boundary F1 | **0.430** |
| Trained model, pooled macro-F1 | 0.724 marginal / 0.714 Viterbi (5-fold, clinician-disjoint) |
| Trained model, per-fold macro-F1 | 0.619 – 0.767 (spread ~0.15) |
| Trained model, boundary F1 | 0.440 marginal / 0.383 Viterbi |
| Trained model, error rate | ~24% (19,949 wrong turns in 10,943 blocks) |
| Error capture at 5% / 30% / 50% flagged | 10.0% / 52.6% / 77.0% |
| Role assignment | No flip detected by either |

**LLM passes** (20 labelled transcripts, 13,896 turns, window 40, with second opinion):

| Property | Value |
|---|---|
| Accuracy | 0.912 |
| Macro-F1 | **0.905** (vs 0.724 deterministic) |
| Boundary F1 | **0.826** (vs 0.440 deterministic) |
| Error rate | 8.8% (1,223 wrong turns in 759 blocks) |
| Errors at maximum confidence | **32.8%** — the hard ceiling on capture |
| Error capture at 4% / 14% / 25% flagged | 24% / **49%** / 65% (weighted confidence) |
| Anchored turns | 35.5%, 4.2% error — vs 11.4% for unanchored |
| Per-file error rate | 1.9% – 16.8% (8.9× spread); worst half carry 68% of error |
| Per-file triage | rho +0.66 but only 1.3× lift — too weak to act on |
| Runtime | ~300s per transcript; ~10 hours for the corpus |

Estimates fell as more transcripts were added — 5.5% error on one file, 6.6% on five, 9.8% on ten,
8.8% on twenty. Treat any figure from a handful of transcripts as optimistic.

**Provisional, from a single transcript** — these have *not* been re-measured across the labelled
set and the class balance above shows how far one transcript can mislead (it suggested ~80/20
against the true ~62/38). Treat them as indicative only:

| Property | Value (n=1) |
|---|---|
| De-identification state | Already de-identified (bracketed placeholder tokens present) |
| Speaker-change rate | ~29% of line boundaries |
| Run length | C: mean ~5.6 lines. T: mean ~1.4 |
| Words per line | C: mean ~10. T: mean ~7 |
| Lines ending in `?` | T: ~44%. C: ~2% |
| First-person tokens per line | T: ~0.15. C: ~1.2 |

Four consequences:

**Turns are ASR fragments, not conversational turns.** A single sentence is routinely split across
several rows, so many lines are a bare acknowledgement or a trailing clause of two or three words.
Lines like these carry no role signal whatsoever and are recoverable only from surrounding context.
Per-line independent classification is not viable.

**The sequential prior is strong, but hard to exploit.** Speakers hold the floor for several turns,
so this is a sequence-labelling problem and any approach ignoring run structure discards most of
the signal. The rule baseline's boundary F1 of 0.430 shows that capturing it is the *hard* part:
role assignment is comfortable (no flip, C-F1 0.787) while finding the changeovers is where the
baseline fails.

**Timestamps are useless for this task.** The median inter-line gap is zero at speaker changes
*and* at non-changes; almost no change shows a measurable pause. The timestamps are
segmentation artifacts, not voice-activity boundaries. Do not build on them.

**Accuracy is a misleading metric.** Labelling every turn `C` scores ~62% while knowing nothing;
the rule baseline reaches 71.3% accuracy but only 0.673 macro-F1. Headline accuracy must never be
reported alone — see [Metrics](../docs/SPEAKER_LABELLING.md#metrics).

**The bar for any model-based approach is macro-F1 0.714**, set by the trained sequence labeller.

**The deterministic path has plateaued.** A trained model over ~83,000 labelled turns gained only
+0.041 macro-F1 over a four-feature hand-tuned rule. That is a small return from a large training
set, and it indicates the surface features are close to exhausted rather than that the model is
badly built.

**The trained model's boundary F1 is worse than the rule baseline's**, despite learning transition
probabilities specifically to improve it. This is Viterbi over-smoothing: where most boundaries are
"no change", maximising path likelihood suppresses switches, which lifts token accuracy and costs
changeover recall. Short therapist runs are the first casualty, which matches T-F1 being the weak
side throughout. Marginal decoding — labelling each turn by its own posterior rather than committing to one best
sequence — confirmed this: it recovers boundary F1 from 0.383 to 0.440 and improves macro-F1 to
0.724. It is now the default. The recovery is real but small, so over-smoothing was a contributing
cause and not the main one.

**The confidence signal is weak, and this is what rules the deterministic path out.** Flagging 5%
of turns catches 10% of the model's errors — twice random, and no more. Flagging half the corpus
(41,524 turns) still leaves 4,585 errors unflagged, against a target of roughly 830. The rule
baseline's hand-made score performs almost identically (8.6% and 76.3%), so this is not a defect
in either mechanism: with a ~24% error rate the posteriors sit near 0.5 across a large share of
turns, because the model is genuinely uncertain nearly everywhere rather than confidently right in
some places and confidently wrong in others.

**Errors are scattered, not clustered.** 19,949 wrong turns fall in 10,943 separate blocks — mean
block length under two. That is roughly 11,000 individual corrections rather than a few hundred
run-level fixes, which is the expensive failure mode for a reviewer.

**Conclusion: the deterministic path cannot reach the target.** Reaching ~830 unflagged errors from
a 24% base error rate would require capturing ~96% of errors; the measured curve reaches 77% at a
50% review budget, and extrapolates to needing well over 80% of turns reviewed — at which point the
system is not helping. What is needed is a lower base error rate, not a better-calibrated
confidence signal over the same errors.

**Fold 0 is 97% one clinician.** Its 0.606 is a score about that clinician, not about the corpus,
and it means the model handles them poorly when trained without them. The 0.153 spread across folds
is exactly why per-fold reporting is required here.

## Labelling goal and constraints

**A human reviews every transcript regardless.** The objective is therefore not autonomous accuracy
but minimising reviewer effort: the target is expressed as *errors left unflagged*, not as an
accuracy figure. See [Metrics](../docs/SPEAKER_LABELLING.md#metrics).

**Target: 99%** — understood as 99% of turns correct among those the system commits to, with the
uncertain residue flagged for the reviewer. On ~83,000 turns that allows roughly 830 unflagged
errors.

**This target was not reached, and is not reachable from text alone.** Measured across 20
transcripts: 8.8% of turns are wrong, and 32.8% of those errors sit at maximum confidence, so
capture cannot exceed ~67% at any review budget. Scaled to the corpus that leaves roughly 2,400
unflagged errors — about three times the target.

The limit is not the confidence mechanism. The remaining errors are turns where the LLM is
consistently wrong *and* an independently-built trained model agrees with it. Two systems that fail
differently, both confident, both wrong, is what an information ceiling looks like rather than a
tuning problem — see the recordings constraint below.

**What was achieved instead:** ~91% accurate labels with about two-thirds of errors flagged at a
28% review budget, in per-transcript reports that show each flagged turn in context. Against
labelling ~83,000 turns from scratch, the reviewer instead works through roughly 4,500 error blocks,
most of them signposted.

### Weighting disagreement by its strength

An early version discounted every second-opinion disagreement by the same flat amount, which put
all of them at one confidence value. That made the flag threshold an on/off switch for the second
opinion rather than a dial — the only way to shorten the list was to discard the whole group.

Scaling the discount by the second opinion's own confidence was measured on 20 transcripts:

| Second opinion's certainty | Turns | Error rate |
|---|---|---|
| 0.00 – 0.25 | 766 | 14.0% |
| 0.25 – 0.50 | 737 | 16.1% |
| 0.50 – 0.75 | 751 | 18.4% |
| 0.75 – 0.95 | 811 | 23.1% |
| 0.95 – 1.00 | 260 | **43.5%** |

A strongly-held disagreement is about three times likelier to mark an error than a weak one, so the
weighting is worth keeping. Even the weakest band runs at 3.5× the 4.0% rate of fully confident
turns.

The practical effect is a much more efficient review list: ~14% of turns flagged for ~49% of errors,
against ~30% of turns for ~53% under the flat scheme — roughly the same capture for half the
reading. `--flag-threshold` now defaults to **0.75** on that basis.

Resulting confidence gradient:

| Confidence | Turns | Error rate |
|---|---|---|
| 0.00 – 0.50 | 266 | 71.4% |
| 0.50 – 0.65 | 257 | 40.5% |
| 0.65 – 0.75 | 1,446 | 21.2% |
| 0.75 – 0.85 | 786 | 13.2% |
| 0.85 – 0.95 | 762 | 11.9% |
| 0.95 – 1.00 | 365 | 7.1% |
| 1.000 | 9,996 | 4.0% |

The ceiling is unchanged: weighting sorts the flagged turns better, it does not reach the 401
errors sitting at maximum confidence.

**Anyone using this output downstream must be told** the labels are ~91% accurate and that roughly
a third of the errors carry no warning. That does not follow from the files themselves.

**The original recordings are not accessible.** This is the same constraint that applies to the
rest of the repository. It has two consequences:

- Acoustic diarization is unavailable. It would very likely outperform any text-based approach at
  finding speaker changes, since the waveform carries that information directly, but it is not an
  option here.
- The human coders labelled from the recordings, so they had information the system never will.
  The gold labels are reliable — near-direct observation rather than judgement calls about text —
  but partly *underivable*: a two-word acknowledgement is trivial by voice and may be genuinely
  indeterminate in the transcript. This is an information ceiling, not annotation noise, and no
  model closes it. Turns like these should be flagged rather than chased.

**No second coder is available**, so the ceiling cannot be measured directly by double coding. The
substitute is to collect turns where the LLM and the trained model confidently agree with each
other and both contradict the gold label, and have a human check that small set. It estimates the
label error rate cheaply but finds only errors the systems happen to catch, so it is a floor rather
than a true ceiling — worth stating that way in any write-up.

## Label conventions in this corpus

### `other` has no confirmed examples yet

No labelled transcript seen so far contains a third-party label; the observed vocabulary is `C`,
`T`, one out-of-vocabulary variant, and `unknown`. The labelled sample opens with a short
unlabelled exchange whose content suggests a brief administrative handover before the session
proper — someone outside the dyad speaking, and someone in the room replying.

That reading is an inference from the text, not evidence. The replies are equally consistent with
the client or the therapist, and that ambiguity is a plausible reason the lines were left
unlabelled in the first place.

So `other` is currently a class this specification proposes, not one observed in the data. It
cannot be learned or measured until labelled instances exist. Implement it as a
**precision-oriented anchor rule** — session-framing utterances addressed to the room from outside
the dyad, such as a pre- or post-session handover — that routes to review rather than asserting a
label, and do not report recall for it. If the wider labelled corpus turns out to contain a
third-party convention, replace this section with what the data actually shows.

### The `X` label

One transcript uses a speaker value of `X` on a very small number of turns (well under 1%), in a
file that is otherwise wholly uncoded. Its meaning is not yet established. The three readings imply
different handling:

| If `X` means | Then |
|---|---|
| A third party | It is this corpus's real `other` label — add it to the vocabulary and protect it from overwriting |
| Inaudible or unclear | Treat as unlabelled, or exclude from ground truth |
| Deliberately excluded | Skip those turns entirely: never label, never score |

Until this is settled, `X` is treated as an unrecognised value: reported, never silently
reinterpreted, and below the anomaly tolerance so it does not hold back the file it appears in.

### Merged-speaker turns

Some manually coded turns carry a combined label such as `C/T`, marking a segment where the
transcription ran an exchange between both speakers into a single turn. An audit of the corpus
found these to be rare — a small number of turns, alongside a few mistyped labels, against an
otherwise consistently coded set.

**Decision: handle them per turn; do not build a sub-turn splitter.** Given the observed rate:

- a merged turn is routed to the review queue for a human to split or relabel;
- it is never assigned a single role label, because no single label is correct;
- it is **excluded from evaluation as ground truth** — `C/T` is not a label the system can be
  scored against producing, so scoring against it would penalise correct behaviour;
- `--anomaly-tolerance` keeps a handful of these from blocking an otherwise processable file.

The alternative — detecting the speaker change *within* a turn and emitting two turns — is a real
extension with a cost that is not worth paying at this rate. It would break the 1:1 mapping between
input rows and output turns, which anything downstream joining on `turn_id` depends on.

**What would change this decision:** merged turns becoming common in a new dataset. Re-run
`speaker-audit` on any new corpus before labelling it; if the merged-turn rate is more than a few
percent, revisit this section rather than letting the review queue absorb the volume.

Mistyped labels are handled per
[Existing labels are never modified](../docs/SPEAKER_LABELLING.md#existing-labels-are-never-modified): reported, never silently
corrected, and fixed at source.

## Evaluation splits

Each client in the corpus is seen by exactly one clinician. The client–clinician graph is therefore
a set of disjoint stars, not one connected component, and **a split that is simultaneously
client-disjoint and therapist-disjoint is just a partition of the clinicians.** No transcripts need
to be discarded to achieve it, and there is one honest generalisation number rather than two
compromised ones.

Approximate corpus shape: ~25 clinicians over ~140 clients, median ~4 clients per clinician.

The binding constraint is skew, not connectivity. The largest clinician accounts for roughly a
fifth of all clients and the three largest for close to two-fifths, while around a quarter of
clinicians have a single client. Leave-one-clinician-out would therefore produce folds of wildly
uneven size and uninterpretable variance.

Use **grouped k-fold at the clinician level** instead:

- assign whole clinicians to ~5 folds, greedily balancing client count per fold;
- never split a clinician across folds — that reintroduces the leakage the design exists to prevent;
- all sessions belonging to one client follow that client's clinician into the same fold, which
  also keeps multi-session and multi-file clients intact automatically.

Because one clinician can still dominate whichever fold holds them, **report per-fold scores, not
only the mean.** A mean that hides one fold at chance level is worse than no number.

#### Labelling status is uneven

Manual labelling is not distributed evenly across the corpus: some clients and clinicians have all
their sessions labelled, others none. This has sharply different consequences for the two paths.

**Supervised path.** Labelled sessions are the entire usable dataset — unlabelled sessions can
serve neither as training data nor as test data. Folds can therefore only be built over clinicians
who have at least one labelled session, so `k` is bounded by *that* count, not by the number of
clinicians in the corpus. If the count is small, cross-clinician generalisation cannot be measured
honestly at any fold size, and the result should be reported as indicative rather than as a
generalisation estimate.

**Zero-shot LLM path.** Unaffected. There is no training set, so there is nothing to split and no
leakage to prevent; every labelled session is valid test data no matter how labelling clusters
across clients and clinicians. This path does not require a held-out client or clinician to exist.

This asymmetry is the strongest practical argument for the zero-shot path on this corpus, and it
compounds the one in [Leakage discipline](../docs/SPEAKER_LABELLING.md#leakage-discipline-differs-per-arm): the zero-shot arm
gets the full labelled set as an honest test set, while the supervised arm must carve the same set
into training and test and may be left with too few clinicians to fold over.

Per-clinician scores should still be reported for the zero-shot path — not to prevent leakage, but
to detect whether accuracy varies by clinician, which a single pooled number would hide.

#### Required inventory

The fold design cannot be finalised without a labelling-status inventory. Per session:

- whether any manual speaker labels are present;
- **completeness** — the proportion of turns carrying a label. This is not binary: a labelled
  transcript can still contain unlabelled turns, so a completeness threshold for counting a session
  as "labelled" has to be chosen and recorded.
- the client and, via the client, the clinician.

Two things to check once it exists, and to state in any write-up:

- how many distinct clinicians have labelled sessions — this is the real upper bound on `k`;
- whether the labelled subset is representative, or whether labelling followed convenience (one
  caseload, one RA, earliest sessions first). A non-random labelled subset biases both the
  supervised model and every score computed from it, in a direction nothing in the output reveals.

Joining the identity mapping to transcript files requires normalising the identifier: session IDs
in the mapping are unpadded integers while transcript filenames are zero-padded and carry a
session-letter suffix. Normalise both sides before joining, and fail loudly on any transcript that
matches no mapping row rather than silently dropping it.

A missing clinician value in the mapping means no recording or transcript exists for that session.
Those rows are removed from the mapping and play no part in the pipeline or the evaluation; there
is nothing to label. One clinician per client is guaranteed by the study design, so the
disjoint-stars property the split relies on holds by construction rather than by observation.

## Identity mapping

`therapist_labels.csv` maps each client to their clinician. It is gitignored and must stay that
way — it is identity linkage, and it is the file that makes leakage-free splitting possible.

- One row per client; the session identifier encodes the client.
- Each client is seen by exactly one clinician, guaranteed by the study design.
- Rows whose clinician value is missing mean no recording or transcript exists for that session.
  They are removed from the mapping and play no part in the pipeline or the evaluation.
- Session identifiers are unpadded integers; transcript filenames are zero-padded and carry a
  session-letter suffix. Normalise both sides before joining, and fail loudly on any transcript
  that matches no mapping row rather than silently dropping it.

## Open questions

- How many transcript files exist per client. The mapping has one row per client, while filenames
  carry a session-letter suffix, so a client may contribute several transcripts to its fold.
- **Does the manual labelling convention have a marker for "present but deliberately not C/T"?**
  If so, it is a protected human label and must not be overwritten. If `unknown` is being used for
  that purpose in any transcript, the "input `unknown` is in scope" rule above is unsafe as written
  and needs revisiting before implementation.
- Do any manually labelled transcripts contain a third-party label at all? This decides whether
  `other` is a real class or one this spec invented.
- Whether the recurring non-dyad voice warrants its own identifier.
- Whether the identity mapping file should record session boundaries where one session spans
  several files.
