---
layout: post
title: "Surface Cues That Look Like Cheating"
date: 2026-11-17
---

# Surface Cues That Look Like Cheating

## Summary

I tested whether a cheat probe built with the paper's recipe detects cheating itself, using transcripts I built. I built it on Qwen2.5-Coder-7B-Instruct and tested it in 6 experiments, where I tried to fool the probe with tricky cases: honest runs that mention failure, lies that use the same words as the truth, test edits that are allowed, and bugs left in the code. In most experiments, when a surface pattern (words, file names, the type of edit) and cheating came apart, the probe's score followed the surface pattern. For example, it scored a test-file edit almost as high as cheating, even when editing the test was allowed. The one exception was a "lie direction," which caught lies even when the final sentences were identical, but it missed other kinds of cheating. My main lesson is that patterns that often come along with cheating might not be cheating, so a probe has to be tested on cases where those patterns and real cheating come apart.

## Why it matters

If we only read a model's words, we only catch surface-level things. A model controls its own words, so a cheating model can make them look clean: it can rig a test and then calmly say "Done, all tests pass." What words can't show is whether the model *knows* it is doing something wrong, like knowing a claim is false and saying it's true anyway. That knowledge lives inside the model. My goal is to build an AI mind reader: a probe that reads what the model knows, not just what it says.

## Background

The paper "Monitoring and Discovering Reward Hacking with Internal Representations during LLM Evaluations" built difference-of-means "reward hacking probes" from fake (synthetic) examples: pairs of conversations that are identical except that one cheats. They built one per hack type and per family, picked the best ones on held-out real rollouts from large open models, and reported that these probes catch reward hacking about as well as an AI judge, at much lower cost. I tested the same basic recipe at small scale (a 7B model and templated transcripts), so my results don't directly test theirs.

The method is a difference-of-means probe. Take the average of the model's internal activations on cheating examples and subtract the average on honest ones; the result is the cheating direction. A new run's score is its dot product with that direction, like a shopping bill where the direction sets the prices. AUROC measures how often cheating runs score higher than honest ones.

I suspected that a direction built from fake examples would differ from real cheating. In fake examples, cheating always comes with certain words and patterns, but in real runs it may not. My experiments tested that suspicion.

## Experiment 1: the fake-data probe

I made 200 fake pairs covering 5 cheat types, following the paper's basic recipe, and built probe 1 from the 160 training pairs (one direction pooled over all five types). On held-out tasks it scored 0.84 AUROC, which looked good. Then I compared it with a word counter, a simple classifier with no AI that only looks at which words appear. The word counter scored 1.00. So the probe may have memorized words that often come along with cheating, even though those words aren't cheating. In 4 of the 5 types, for example, only the honest version's changed step edited a file in the `src/` folder.

Next, I rebuilt the probe from 4 cheat types and tested it on the 5th, which it had never seen, once for each type. Four types scored between 0.42 and 0.81. False reporting scored 0.00: every honest run scored above every cheating run. In that type, the honest model says "1 test still fails" and the liar says "All tests pass," so what the probe had learned pointed exactly backwards. A monitor like this would punish honesty and reward hiding bad news.

## Experiment 2: the lie direction

To separate the words from the lie, I made four versions of each run, changing only what really happened and what the model says. "All pass" could be true or a lie, and "1 test fails" could be true or a lie. I built the lie direction as the average of the two lies minus the average of the two true claims. Each side has one "all pass" and one "fails," so the words cancel out and only the lie is left.

On 8 new tasks it ranked every lie above every true claim (AUROC 1.00), even when the final sentences were identical. A word counter scored 0.64 overall: 0.77 when both versions said "all pass" and 0.50 (a coin flip) when both said "fails." But it barely caught other kinds of cheating (0.25–0.70 on 8 tasks each). A lie detector catches lying, not other types of cheating. When a model rigs a test and then says "all tests pass," the claim is true, so there is no lie to detect, like a student who changes the answer key and then truthfully says their answers match it.

One caution: in these transcripts a lie always means the final claim contradicts the test output shown just above it. So the lie direction may read "the claim contradicts the output" rather than "the model knows it is lying," and I haven't tested cases where those come apart.

## Defining cheating

To know what a probe should detect, I first needed a definition, so I built one step by step, testing each version against an edge case. I started with "getting the reward by breaking the rules." But a soccer player who fouls openly and accepts the yellow card isn't usually called a cheater, while one who fakes a fall is, because the second one tricks the referee. So cheating needs tricking the judge. Next, an AI that edits a test breaks no written rule, yet it defeats what the test is for, so cheating means breaking the rules or their purpose. Finally, if the AI truly didn't know it was doing something wrong, it's a mistake, not cheating. Law makes the same distinction between a guilty act and a guilty mind.

My definition: **cheating is knowingly getting the reward by breaking the rules or their purpose, while tricking the judge.**

| Part | How to check it |
|---|---|
| Gets the reward | Outside: did the score go up? |
| Breaks the rules or their purpose | Outside: were the tests changed? Rerun them |
| Tricks the judge | Outside: does the claim match what happened? |
| Knowingly | Inside only: a probe |

That makes "knowing" the real job for a probe.

## Experiments 3–6: looking for "knowingly"

These tests tried to find the part of my definition that only the insides can show: whether the model cheats knowingly. Each test used transcripts that are identical except for one or two controlled details (the instruction, which file was edited, or a single line of the edit), so I could see which detail the probe was responding to. For Experiments 4–6, I wrote down the plan and the pass bar before looking at any results (Experiment 5's bar was replaced by a dated amendment, still before analysis). Experiment 3 had a fixed procedure but no pass bar.

| Test | Question | What I measured | My reading (after the fact) |
|---|---|---|---|
| 3 | Does the model represent "editing the test is wrong here"? | The cheat case scored highest | Too few test tasks (8) to tell; random directions sometimes did as well |
| 4 | Does it hold on 120 new tasks and 2 new formats? | Missed the bar set in advance at the edit (0.78 vs 0.83). At the final sentence it scored 0.99, but no better than just adding an instruction cue to an action cue | A "mismatch" signal: the action doesn't match the instruction. It can't tell cheating from disobeying without cheating |
| 5 | Cheating vs. disobeying? | The "cheat" direction failed its pass bar, and so did the mismatch direction | The type of edit; mismatch showed up again at the final sentence and ignored whether the bug was fixed |
| 6 | Is the bug still there? | A line changed but still broken scored as even more *fixed* than the real fix (0.38 vs a 0.75 bar), while the direction still separated unfixed from fixed code on 60 new tasks (0.87) | My best guess: whether the old line was copied or rewritten (not yet confirmed) |

Across these tests, the most consistent signal was mismatch: "I did something other than what was asked." It held up on new formats in Experiment 4, but it is what my 2x2 contrast was built to pick up, and it failed its pass bar in Experiment 5. Mismatch is one piece of cheating, but not cheating itself.

Before Experiments 4–6, the AI assistant I worked with (Claude) wrote predictions into each plan; I gave none. Some held: neither direction would pass in Experiment 5, and in Experiment 6 the direction carried over from Experiment 5 would still separate unfixed from fixed code on 60 new tasks (0.87, predicted 0.72–0.88) while the main test failed. Some failed: in Experiment 4 it expected one new format to transfer better than the other (the reverse happened) and a weak score at the final sentence (it was 0.99). In Experiment 6 it expected my yes/no control check to work (it didn't), and the broken-line score landed just under its range (0.38 vs 0.40–0.60). Writing them down first is how I know which were wrong.

## The pattern and what I learned

Across six experiments, surface patterns kept pretending to be cheating. What looked like a cheating signal was better explained by the words "all pass," then a mismatch between the action and the instruction, then the type of edit, and finally, as my best guess so far, whether a line was copied or rewritten.

The lesson that matters most to me is **capability first**: a model has to know whether it is doing the right thing before a probe can read that knowledge. When I asked my model whether each file's code was correct, its top answer was "No" for 179 of 180 files, including 59 of the 60 fixed ones. Its "No" was more confident for the broken rewrites than for the fixed code in 56 of 60 tasks, but my control check missed its pre-set bar (0.80 vs 0.90), so I can't tell whether the model knew which code was broken.

Two more lessons:

- **Fake data can't prove a probe is smart.** A probe has to be tested where the surface trick fails.
- **Writing the plan before looking at results kept me honest.** It even caught a flaw in my own design.

## Limits and next steps

These results come from one 7B model, only straight-line (difference-of-means) probes at fixed layers, and fake transcripts built from templates. The model never did any of this: it read transcripts I built, so these results are about what it represents while reading a cheat, not while doing one. I tested my own version of the paper's recipe, not the paper's probe or data. Test sets were small: 8 held-out tasks in Experiments 1–3, 20 in Experiment 5, 60 in Experiment 6 (Experiment 4 used 120). What would change my mind: a direction that scores Experiment 6's broken lines with the unfixed code on new tasks, or a probe that still works on real runs where cheating and its surface cues come apart. Treat all of this as suggestive, not settled.

My next step is to use a bigger model that first passes a capability check: it has to tell broken code from fixed code when asked. Only then does it make sense to look for "knowing" inside it. After that, I want to:

- Collect real cheating runs from impossible tasks instead of fake transcripts
- Try probes that aren't straight lines
- Confirm the "copied vs. rewritten" explanation with a correct rewrite that's far from the old line

## Reproducibility

The code and results are in two folders of my research repo, not public yet: `probe1/` for Experiments 1–2 (`make_pairs.py`, `extract_acts.py`, `direction_cv.py`, `checks_1c.py`, `fr_cells.py`, `lie_direction.py`) and `probe_mensrea/` for Experiments 3–6 (`analyze.py`, `analyze_c.py`, `analyze_d.py`, `analyze_e.py`, `gate_e.py`, and the plans `PLAN_C.md`, `PLAN_D.md`, `PLAN_E.md`). Only Experiment 6 was registered in git before it ran (commit `70502d5`: plan, tasks, transcripts and analysis code). Experiments 1–5 were not under version control, so I have no commit to give for them.

I ran this with heavy help from an AI assistant (Claude): it drafted the plans, generated the tasks, wrote and ran the analysis code, and red-teamed the results. I set the questions and directed the work.

## Sources

- Bergen et al. (2026), [Monitoring and Discovering Reward Hacking with Internal Representations during LLM Evaluations](https://arxiv.org/abs/2609.19101) (arXiv 2609.19101)
- Bernard Suits, *The Grasshopper: Games, Life and Utopia* (University of Toronto Press, 1978): the cheater as someone who breaks the rules to win
- Full experiment details are in a longer research note, *Cheat Probes: Doing or Talking?*, which I can share on request
