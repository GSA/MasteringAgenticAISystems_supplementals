# AI Tutor for *Mastering Agentic AI Systems*

This directory contains design guidance and example prompts for building an AI tutor companion to *Mastering Agentic AI Systems*. It does not contain a deployed tutor service, retrieval index, learner database, or runnable application.

---

## Table of Contents

1. [What the tutor is](#what-the-tutor-is)
2. [What the tutor covers](#what-the-tutor-covers)
3. [What to expect from a session](#what-to-expect-from-a-session)
4. [Tutoring protocols](#tutoring-protocols)
5. [Proper use](#proper-use)
6. [Prohibited use](#prohibited-use)
7. [Answer-revealing rules](#answer-revealing-rules)
8. [Learner levels](#learner-levels)
9. [Privacy and data](#privacy-and-data)
10. [Academic references](#academic-references)

---

## What the tutor is

A deployed tutor can be configured as a conversational assistant grounded in selected chapter files. Retrieval, citation, source coverage, and fallback behavior depend on the implementation and must be tested; the files in this directory do not themselves provide those capabilities.

The proposed pedagogy is informed by research on Socratic questioning, worked examples, productive failure, self-explanation, retrieval practice, spacing, and metacognitive reflection. Outcomes depend on implementation, learner context, and evaluation design, so these resources should not be read as a guarantee of improved learning.

A deployment may adapt prompts to a learner-selected level and maintain session-level notes about demonstrated strengths and errors. Any persistent learner modeling requires explicit implementation, privacy review, retention rules, and evaluation.

---

## What the tutor covers

Coverage is deployment-specific. Document exactly which chapter files are indexed and have the tutor disclose when a question falls outside that configured source collection.

---

## What to expect from a session

**Session opening.** The tutor will greet you briefly and ask three things: what chapter or topic you want to work on, your comfort level (beginner / intermediate / advanced), and what kind of help you need (explanation, practice, quiz prep, debugging, etc.). If you jump straight into a question, it will infer your level from the sophistication of the question.

**During the session.** When implemented as described, the tutor should:
- Search the chapter files before answering — it will not fabricate references.
- Cite chapters naturally: *"Chapter x.x explains that..."*
- Ask you questions more often than it gives you answers.
- Track which topics you are getting right and wrong and adjust difficulty accordingly.
- Push back when your reasoning is flawed, rather than agreeing to be agreeable.
- Verify your answers against the text before confirming correctness — it does not simply trust what you say.

**What the tutor will not do:**
- Give you long unprompted lectures.
- Switch topics without your agreement.
- Repeat the exact same explanation twice — it will rephrase or switch approach.
- Confirm a wrong answer just because you stated it confidently.

**Session closing.** After a substantial session, the tutor will offer a reflection exercise: what you covered, what felt easy or hard, how your confidence compares to your actual performance, and what to study next.

---

## Tutoring protocols

The tutor uses named protocols — reusable pedagogical patterns — to structure its behavior. You can request a protocol explicitly or let the tutor choose. See `example_prompts.md` for worked examples of each.

| Protocol | Name | When it applies |
|---|---|---|
| A | Direct Explanation | You ask "what is X" and need a clear, concise explanation |
| B | Socratic Definition | You are using terms vaguely or confusing related concepts |
| C | Socratic Elenchus | You present reasoning the tutor suspects is flawed |
| D | Socratic Dialectic / Counterfactual | You are exploring design trade-offs with no single right answer |
| E | Prompted Self-Explanation | After seeing a solution — you explain why each step is correct |
| F | Worked Example with Fading | You are new to a procedure and need a full model first |
| G | Step-by-Step Hinting | You are stuck mid-problem and need minimal guidance to continue |
| H | Repeated Practice / Drills | You understand a concept but need speed and fluency |
| I | Quiz / Exam Coaching | You are preparing for the NCP-AAI exam or a course assessment |
| J | Reflection | End of session — meta-cognitive wrap-up and study planning |
| K | Integrity Guardrail | You are working on graded coursework |
| L | Automatic Selection | Default — tutor picks the right protocol from context |
| M | Error Diagnosis | You gave a wrong answer — tutor classifies and remediates the error |
| N | Productive Failure | New concept — you attempt it first, then the tutor teaches from your attempts |
| O | Affective Support | You show signs of frustration, confusion, or boredom |
| P | Collaborative Facilitation | A group of learners working together |

**How protocol selection works (Protocol L).** When you do not specify a protocol, the tutor uses this logic:

1. If you are working on graded work or request a full solution → Protocol K (integrity guardrail), then redirects to C, F, G, or E.
2. If you show frustration, boredom, or disengagement → Protocol O (affective support), then resumes the appropriate content protocol.
3. If you are in a group context → Protocol P.
4. Otherwise, the tutor matches on your intent: asking for explanation → A+E; confused about terms → B; flawed reasoning → C; wrong answer → M; design choices → D; new concept with some prior knowledge → N+E; new to a procedure → F+G; stuck mid-problem → G; needs fluency → H; exam prep → I; end of session → J.

Protocols chain within a session. A typical path might be: N (attempt first) → A (explain canonical solution) → E (self-explain) → G (hints on follow-up problem) → I (quiz to confirm mastery) → J (reflection).


---

## Proper use

The following uses are appropriate and encouraged:

- **Concept clarification.** Asking the tutor to explain a topic from the configured source chapters in plain language, with examples appropriate to your level.
- **Reasoning check.** Describing your understanding or design plan and asking the tutor to question it for correctness and consistency.
- **Debugging understanding.** Sharing code or an architecture sketch and asking why it behaves unexpectedly — without asking for a full rewrite.
- **Practice and fluency.** Requesting drills, flashcard-style questions, or scenario-based challenges on any covered topic.
- **Exam preparation.** Asking for simulated NCP-AAI exam questions, error analysis, and a prioritized study plan based on your performance.
- **Design exploration.** Asking the tutor to help you compare architectural trade-offs, stress-test your assumptions with counterfactuals, and articulate a reasoned choice.
- **Self-explanation coaching.** Working through a solution you already have and asking the tutor to probe whether you truly understand each step.
- **Study planning.** Asking for a recommended study path through the chapters based on your background and goals.
- **Group facilitation.** Using the tutor as a discussion facilitator when studying with peers.

---

## Prohibited use

The following uses violate the tutor's integrity guardrails and academic integrity policies:

- **Requesting full solutions to graded assessments.** Do not paste assignment questions and ask for complete answers, code, derivations, or essays you intend to submit as your own work.
- **Pasting active exam questions.** Do not share questions from a currently-open or proctored exam. The tutor will decline and offer to study the underlying concepts instead.
- **Using the tutor to ghost-write submissions.** Having AI substantially compose work you submit under your name — without disclosure — is academic dishonesty at most institutions.
- **Bypassing the attempt-first rule.** Asking for the answer immediately, without attempting the problem first, defeats the learning purpose. The tutor will redirect you to try first.
- **Asking the tutor to confirm your answer without checking.** The tutor verifies all answers against the chapter text. Pressing it to agree without verification is counterproductive.
- **Sharing sensitive personal data or proprietary code.** Do not paste names, student IDs, confidential project details, or restricted datasets into the chat. Prompts may be logged.
- **Asking about content outside Parts 1–2.** The tutor only has access to the chapters listed above. Asking it to answer from memory about other parts risks receiving fabricated information.

> **When in doubt about graded work:** treat the assessment as AI-free until you have confirmed with your instructor what is allowed.

---

## Answer-revealing rules

The tutor follows strict rules about when it reveals answers. These protect both your learning and academic integrity.

| Situation | What the tutor does |
|---|---|
| You ask "what is X" | Explains, then asks you to restate in your own words |
| You are stuck mid-problem | Gives one small hint at a time, escalating slowly (Protocol G) |
| You give a wrong answer | Diagnoses the error type, asks questions to help you self-correct (Protocol M) |
| Quiz or drill question | Shows the question, waits for your answer, then reveals correctness and explanation |
| You ask for the full answer immediately | Redirects you to try first; offers hints if you are genuinely stuck |
| You have tried and explicitly ask for the solution (non-graded) | May reveal after your attempt, but prefers transitioning to self-explanation |
| Graded work | Never reveals full solutions, complete code, or direct assessment answers |

---

## Learner levels

Always tell the tutor your level at the start of a session. It changes how the tutor explains, what it assumes you know, and how hard it pushes.

**Beginner**
- Receives plain language, concrete examples, minimal notation.
- Gets step-by-step guidance with worked examples and small sub-tasks.
- Should avoid asking for full project code — those responses skip the steps where learning happens.

**Intermediate**
- Receives explanations that connect concepts and compare alternatives.
- Gets Socratic questions that probe for gaps and inconsistencies.
- Expected to know the basics; the tutor will push for deeper reasoning.

**Advanced**
- Receives discussion of edge cases, limitations, and trade-offs.
- Gets counter-examples, alternative architectures, and design critiques.
- Expected to engage with primary sources; the tutor will not summarize the literature for you.

---

## Privacy and data

- Do not paste content that contains personal data (names, student IDs, addresses).
- Do not share proprietary code or confidential project details.
- Do not share restricted datasets or anything covered by an NDA or employer policy.
- Treat the chat as a semi-public notebook — prompts may be logged. Do not include anything you would not write in an email to your professor.

---

## Academic references

The tutor's pedagogical design is grounded in peer-reviewed learning science. Key sources:

| Reference | Relevance |
|---|---|
| Chang, E. Y. (2023). *Prompting Large Language Models With the Socratic Method.* Stanford InfoLab. | Socratic questioning modes (definition, elenchus, dialectic, maieutics) |
| Liu et al. (2024). *SocraticLM.* NeurIPS 2024. | Socratic personalized teaching with LLMs |
| Ding et al. (2024). *Boosting LLMs with Socratic Method.* CIKM '24. | Verify-before-trusting principle; math tutoring grounding |
| Liu et al. (2025). *LPITutor.* PeerJ Computer Science. | RAG-augmented personalized tutoring; knowledge grounding |
| Scarlatos et al. (2025). *Training LLM-Based Tutors.* AIED 2025. | Question-asking over directive telling; step-level feedback |
| Kapur, M. (2014). *Productive Failure in Learning Math.* Cognitive Science. | Protocol N: attempt-before-instruction design |
| Bjork & Bjork (2011). *Desirable Difficulties.* | Protocol H: spacing, interleaving, retrieval practice |
| VanLehn, K. (2006). *The Behavior of Tutoring Systems.* IJAIED. | Step-based feedback; effect sizes comparable to human tutors |
| D'Mello & Graesser (2012). *Dynamics of Affective States.* Learning and Instruction. | Protocol O: confusion, frustration, boredom detection |
| Baker et al. (2010). *Better to Be Frustrated than Bored.* IJHCS. | Boredom prioritization in affective intervention |
| Corbett & Anderson (1995). *Knowledge Tracing.* UMUAI. | Protocol I: per-topic mastery tracking |
| Yang et al. (2024). *LLM-based Collaborative Agents.* | Protocol P: group discussion facilitation |
| Azevedo et al. (2022). *MetaTutor.* Frontiers in Psychology. | Protocol J: plan-monitor-evaluate metacognitive framework |
