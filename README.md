# Engineering Case: Find the Leakage

Welcome, and thanks for taking the time on this. This case is meant to take
**3 to 4 hours of focused work**. Please don't push past that — we'd rather
see what you can do in the time than a polished overrun.

You can use **any AI tools you'd normally use** (Claude, Cursor, Copilot, etc.).
We expect you to. We're evaluating your judgment, your verification habits,
and how you communicate — not whether you typed every line yourself.

---

## The story

You've joined a small healthcare operations team. The team pays provider
claims against a contracted fee schedule. The CFO has asked us to look into
whether we're overpaying somewhere — there's a hunch that something is off,
but no one has had time to dig in. You've been given a year of claims and
payments data, plus a few reference tables and operational notes.

**Your job:** investigate, find what's wrong (if anything), quantify it,
and recommend what to do — including, where appropriate, what *not* to
recover.

There is no rules document. You decide what's worth investigating.

---

## What's in `data/`

| File | What it is |
| --- | --- |
| `claims.csv` | ~50,000 claims submitted in 2025. Each claim has a member, a provider TIN, a CPT, modifier, units, billed amount, and date of service. |
| `payments.csv` | What we actually paid for each claim, and when. |
| `providers.csv` | Provider directory with TIN, specialty, contracted status, and contract dates. |
| `members.csv` | Member directory with eligibility periods. |
| `fee_schedule.csv` | The CPT fee schedule — `allowed_unit_amount` is the contracted unit rate. |
| `members_eligibility_history.csv` | Historical record of member terminations and re-enrollments. |
| `contract_amendments.csv` | Amendments that override `providers.csv` (term extensions, rate changes, etc.). |
| `contract_carveouts.csv` | Provider/CPT-specific rate carve-outs that override the fee schedule. |
| `pricing_notes.md` | Short ops note on modifier-driven pricing rules and how amendments/carve-outs work. **Worth reading.** |

The data is synthetic but modeled on real healthcare claims. It is *not*
clean — assume you may find encoding quirks, format inconsistencies, or
other real-world messiness. Part of what we're evaluating is whether you
notice.

The reference files (`contract_amendments.csv`, `contract_carveouts.csv`,
`members_eligibility_history.csv`, `pricing_notes.md`) can override the
main tables when they apply. Open each one before flagging something as
leakage — a finding that contradicts a reference file is a false positive.

Some claims that *look* like leakage on first pass are actually correct —
your job includes telling us which ones we should NOT chase.

---

## What to deliver

Three things, in a single zip back to us:

### 1. A short memo (`memo.md` or `memo.pdf`)
1-2 pages, written as if to the CFO. Cover:
- What you found
- How much money is involved (best estimate, with confidence)
- **What you considered but recommend NOT recovering, and why**
- Evidence supporting each finding — for each finding, name at least one
  specific `claim_id` so we can audit it
- What you'd recommend the team do — operationally and technically

### 2. Your code
Whatever scripts/notebooks you used. Include a `README.md` so we can run it.
Doesn't have to be production-grade — readable and reproducible is enough.

### 3. A Loom recording (8–10 minutes)
Record a screen-share walkthrough. Loom is free at loom.com. Drop the link
in your submission email.

**Cover this agenda in the recording, in this order:**

1. **(30 sec) Approach summary.** What did you decide to investigate, and why?
2. **(4–5 min) Walk through your findings.** For each one: what's the issue,
   how much money, a specific `claim_id` as an example, and how you
   verified it's real (not a false positive).
3. **(1–2 min) Show one piece of code, explain a decision you made.**
4. **(2–3 min) Answer these three questions out loud:**
   - Where did you spend the most time, and why?
   - What's one finding you're least sure about, and how would you verify
     it if you had another day?
   - **Which of your findings would you escalate to legal or compliance
     before sending a recovery request, and why?**

You don't need a script. Talking through your thinking honestly is what
we're looking for. If you got stuck, say so. If you used Claude for a
specific piece, say that too.

---

## Submission

Email the zip + Loom link to the address in your interview invite. Include
roughly how many hours you spent so we can calibrate.

---

## A few notes

- We are deliberately not telling you a target dollar amount. Whatever you
  find, find. If you think nothing is wrong, make that case.
- Sanity-check your own numbers. Do the totals add up? Would you stake your
  recommendation on them?
- This case is about judgment, not coverage. Three findings you can defend
  beats six you can't. Two findings you defend *and* a clearly-explained
  "do not recover" item beats four findings without context.
- If you have a clarifying question, email us — we may answer, or we may
  tell you to use your best judgment. (That's also a signal.)

Good luck.
