# RFPU: an AI that does exact radio-signal analysis inside its own "brain"

*Overview for stakeholders. No background in radio or AI is needed. Status as of 5 November 2026.*

---

## 1. The one-paragraph version

We took a modern AI language model (Qwen, a model similar in kind to ChatGPT) and made it analyse real radio signals, such as Wi-Fi, mobile-phone, TV broadcast and drone-controller transmissions, **using only its own internal machinery**. No outside calculator or software does the work for it. Every single step of the calculation is checked against an independent reference, so the results are **exact and provable**, not "AI guesses". The AI can then **ask itself** for an analysis in the middle of a conversation, get exact numbers back, and explain them in plain language. To our knowledge, nobody has shown this before.

---

## 2. Background, in plain words

### What is a "radio signal" and why analyse it?

Almost everything wireless (phones, Wi-Fi, TV, GPS, drones, car keys, aircraft) talks by sending invisible radio waves. A radio receiver records these waves as long lists of numbers, called **samples**. Analysing those numbers answers practical questions:
- Is anyone transmitting right now?
- How strong is the signal compared with the background noise?
- At what frequency, like a station on a dial?
- What kind of device is it: Wi-Fi, phone network, TV?

This work is called **signal processing**. It is normally done by specialised chips or conventional software that follow fixed mathematical recipes.

### What is a "language model"?

A language model is the kind of AI behind chatbots. Inside, it is a very large set of numbers (called **weights**) arranged in layers. Text goes in, the layers transform it step by step, and the model predicts the next word. Language models are good at reasoning in words, but they are **not** reliable calculators: ask one to do precise arithmetic and it may produce a confident wrong answer.

### The key idea we build on: a computer hidden inside the AI

Earlier this year, open-source researchers showed something surprising: you can **rewrite part of a language model's weights so that a small, real computer runs inside it**. The model's normal "predict the next word" machinery is repurposed so that each "word" it produces is actually one step of a tiny computer (a processor plus memory). Because the processor follows exact rules, its answers are exact.

This idea is called an in-weight computer. Examples are the projects Qwenputer and QwenPU; **we did not invent it, and we say so**.

What nobody had done is use such a hidden computer for **real-world signal processing**, or let the AI use it as a tool on real data. That is what we built.

---

## 3. What we actually built

Think of it in three layers.

1. **A toolbox of signal-analysis programs that run inside the AI**
   - We wrote **29 analysis programs**. They filter signals, find their frequencies, measure strength and noise, detect bursts of activity, demodulate (decode) simple signals, and recognise what kind of transmitter is present.
   - All of them run on the computer hidden in the AI's weights.
2. **A strict checking system**
   - Every step the AI's internal computer takes is compared, one by one, with an independent reference computer running the same program.
   - Every final answer is compared with a separately written calculation.
   - Only if **everything** matches do we count the result. Millions of steps have been checked this way, with zero mismatches.
3. **The AI using its own toolbox**
   - When you ask Qwen a question such as "Is there a burst of activity in this recording, and at what frequency?", it decides to call its built-in analysis tool. The hidden computer does the exact calculation and returns a report, and Qwen answers in plain English, quoting the exact numbers.
   - This runs on a single graphics card in a laptop.

We also **extended the hidden computer itself**. We designed and added three new "instructions" (built-in operations) specialised for radio work. One is for filtering, one for measuring power, and one for the core step of frequency analysis (the FFT). They made those tasks up to about 7× faster, without making the AI any bigger in its main structure: no extra layers.

---

## 4. What we achieved, in numbers that matter

| Question | Result | What it means in plain terms |
|---|---|---|
| Does it work on **real** recordings? | **132 out of 133** real recordings pass, from 5 independent sources (Wi-Fi/phone/TV, LoRa, drones, radio astronomy) | The previous version failed all of its 14 test recordings; this one passes those 14 and 132 of 133 in a much bigger test. The single miss is a near-tie between two almost identical peaks; the maths itself was right. |
| Does every tool hold up on all of them? | The full 8-step analysis chain and 10 separate tools are **exact on all 133 recordings** | Over 300 million calculation steps checked overnight, zero mismatches. |
| Is it exact? | Millions of steps checked, **zero errors** | Every step agrees with an independent reference computer. |
| How precise are the answers? | Most tools are accurate to about **1 part in 3,000 to 1 part in 25,000** | Comparable to good conventional digital equipment working with 16-bit numbers. Weak and strong signals are now handled equally well. |
| Can the AI use it in conversation? | **Yes.** In an automatically graded test of 96 questions, **99% of the numbers in the AI's answers came straight from the verified analysis**, and it picked the right tool every time | The few numbers that did not come from the verified analysis were the AI's own unit conversions; one was wrong by a factor of 1,000, and the automatic checker flagged it. The analysis now hands the AI those values directly. |
| How long does a question take? | About **30–50 seconds** on one laptop graphics card | Down from about 5 minutes this morning, through a faster but equally strict checking method that produces identical results. |
| Can it recognise what kind of transmitter it is hearing? | **96%** of recordings correctly identified as Wi-Fi, mobile (LTE) or TV (DVB-T) | On the higher-quality recordings; a small sample of 26 recordings, so indicative. The classifier itself runs inside the AI and is exactly verified. |
| Could we improve the hidden computer? | **3 new built-in operations** added, verified at every level including the saved AI model file | Up to **6.7× faster** filtering, proven with a pre-registered statistical test. |

---

## 5. Why this is unique, and why "first of its kind"

1. **Nobody had done signal processing inside an AI's weights before.** The handful of projects that hide computers inside language models run general-purpose programs and simple maths. A search of all of them (as of early November 2026) found **no radio, signal-processing or frequency-analysis work**.
2. **Existing "AI for radio" work keeps the maths outside the AI.** Today's systems pair a language model with ordinary software or specialised chips that do the analysis; the AI then reads the result and has to trust it. In our system the analysis happens **inside the AI's own weights**, and it is **checked step by step**.
3. **Exactness you can audit.** Most AI outputs cannot be proven correct. Ours can. Every number the AI reports comes from a computation that was checked step by step, and the full record is kept.
4. **The first radio-specific instructions added to a computer inside an AI.** We showed that you can design new built-in operations for this kind of hidden computer, test them, and compile them into the model without making it deeper.
5. **Honest, pre-registered science.**
   - We wrote down our success criteria **before** running each test and never moved the goalposts.
   - We published our failures as well: one precision target was first missed (69.2 against a goal of 70), and one idea for improving recognition did not help.
   - We then fixed the miss properly with a new design. That reached 77.9, against a new target of 75 that was also set in advance.

> **The careful claim:** *To our knowledge, this is the first demonstration of a language model performing exact, independently verified radio signal processing and recognition on real recordings within its own weights, and the first radio-specific instruction added to a computer compiled into a language model's weights.*

---

## 6. Why it matters for the public domain

- **Trustworthy AI for safety-critical sensing.** Radio monitoring matters for:
  - aviation and drone safety;
  - spectrum regulation (who is transmitting where);
  - emergency services;
  - cybersecurity (spotting rogue transmitters).

  An AI that can show *how* it got every number is far easier to trust and to audit than one that reasons opaquely.
- **A new way to combine AI with exact computation.** Instead of an AI calling outside software it cannot verify, the exact computation lives *inside* the model. Every step is recorded and checkable, so you get a full audit trail.
- **A blueprint beyond radio.** The same approach could apply to other sensor data (audio, vibration, medical signals, industrial sensors) and to any field where an AI must give exact, defensible numbers.
- **Openness and reproducibility.** Every result has a script, a log and a checksum, and a step-by-step guide lets others rerun it. Failures are kept on record.

---

## 7. What it is **not** (honest limits)

- **It is not fast.** It is roughly 100,000 to 1,000,000 times slower than a normal signal-processing chip. This is a proof of what is possible and of trustworthiness, not a real-time product.
- **"Exact" means exact to a defined 16-bit number format**, the same kind of format real digital equipment uses, not infinitely precise. We measure and publish the remaining tiny rounding error for every tool.
- **The AI's *interpretation* is not verified, only its numbers.** The numbers it reports are checked automatically (99% trace back to the verified analysis, and the rest are flagged). Its explanations are ordinary AI reasoning and can be wrong.
- **Evidence base:**
  - 133 real recordings for the analysis tools;
  - 26 recordings for transmitter recognition (still small);
  - 96 graded conversation questions;
  - one laptop.
- **One model size.** Only one size of this AI family (4 billion parameters) has the internal layout the hidden computer needs.
- **One checking mode needs the reference.** The fast checking method has the reference computer propose the steps and the AI's own layers confirm each one. It gives identical results to the slow method, but it is an audit mode, and we state that plainly.

---

## 8. What could be done next

**Near term**
- Publish the research paper; the draft is written.
- More recordings for transmitter recognition (the analysis tools are already tested on 133).
- Run the speed and accuracy studies on a second machine.

**Engineering**
- Fit the version with the new radio instructions onto a single graphics card together with the AI, so conversations also benefit from them.
- Add more specialised instructions.
- Make the checking even faster.

**Applications to explore**
- **Explainable spectrum monitoring:** an assistant that answers "what's on the air here?" with provable numbers.
- **Auditable drone and interference detection,** where every alert comes with a verifiable calculation.
- **Teaching and certification tools** that show exactly how each result was computed.
- **Other sensors:** audio, vibration, biomedical signals.

**Research directions**
- **Larger AI models,** which would require a different way of compiling the hidden computer.
- **Mixing exact and approximate reasoning,** where the AI's own judgement and the exact computation inform each other.
- **Formal guarantees** that go beyond step-by-step checking.

---

## 9. Where to find the details

| What | Where |
|---|---|
| Plain summary of all results | [README.md](README.md) |
| Draft research paper | [paper/PAPER_DRAFT.md](paper/PAPER_DRAFT.md) |
| Every experiment, including failures, in date order | [research/RESEARCH_LOG.md](research/RESEARCH_LOG.md) |
| Success criteria written before each test | [research/PROTOCOL.md](research/PROTOCOL.md) |
| How to reproduce everything | [REPRODUCE.md](REPRODUCE.md) |
| Comparison with related work | [research/NOVELTY_SEARCH.md](research/NOVELTY_SEARCH.md) |
