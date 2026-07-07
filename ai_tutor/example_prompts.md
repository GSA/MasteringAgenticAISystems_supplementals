# Example Prompts for the AI Tutor

This file shows how to phrase your opening message for each tutoring scenario. Effective prompts tell the tutor three things: your level, what you want to do, and what constraints to follow (hints only, no solutions, etc.).

Copy and adapt these prompts — replace the bracketed placeholders with your actual topic, chapter, or code snippet.

---

## Table of Contents

1. [Starting a session](#starting-a-session)
2. [Getting a concept explained (Protocol A)](#getting-a-concept-explained-protocol-a)
3. [Sharpening a definition (Protocol B)](#sharpening-a-definition-protocol-b)
4. [Having your reasoning challenged (Protocol C)](#having-your-reasoning-challenged-protocol-c)
5. [Exploring a design decision (Protocol D)](#exploring-a-design-decision-protocol-d)
6. [Self-explaining a solution (Protocol E)](#self-explaining-a-solution-protocol-e)
7. [Learning a new procedure (Protocol F)](#learning-a-new-procedure-protocol-f)
8. [Getting unstuck with hints (Protocol G)](#getting-unstuck-with-hints-protocol-g)
9. [Building fluency with drills (Protocol H)](#building-fluency-with-drills-protocol-h)
10. [Preparing for the exam (Protocol I)](#preparing-for-the-exam-protocol-i)
11. [Reflecting on a session (Protocol J)](#reflecting-on-a-session-protocol-j)
12. [Working on graded coursework (Protocol K)](#working-on-graded-coursework-protocol-k)
13. [Diagnosing a wrong answer (Protocol M)](#diagnosing-a-wrong-answer-protocol-m)
14. [Tackling a new concept (Protocol N)](#tackling-a-new-concept-protocol-n)
15. [When you feel stuck or frustrated (Protocol O)](#when-you-feel-stuck-or-frustrated-protocol-o)
16. [Group study session (Protocol P)](#group-study-session-protocol-p)
17. [Chaining protocols across a session](#chaining-protocols-across-a-session)
18. [Common mistakes to avoid](#common-mistakes-to-avoid)

---

## Starting a session

Use this every time you open a new conversation. Fill in your level and goal so the tutor calibrates immediately.

```
I am an [beginner / intermediate / advanced] learner working through
"Mastering Agentic AI Systems." Today I want to [study Chapter 1.3 on
multi-agent systems / prepare for the NCP-AAI exam / debug my
understanding of LangGraph / practice with drills on Chapter 2.6].

Please start by asking me what I already know about the topic, then
choose the right teaching approach based on my answers.
```

---

## Getting a concept explained (Protocol A)

**Scenario:** You encountered a term or concept in the chapter and want a clear, grounded explanation before going deeper.

```
I am an intermediate learner on Chapter 1.4 (Memory and Perception Systems).
I just read about episodic memory but I'm not sure how it differs from
semantic memory in the context of agentic systems.

Please explain both terms using the chapter text as your source.
Give me a concrete example of each, then ask me to restate the
difference in my own words. After that, give me one application-level
check question.
```

```
I'm a beginner on Chapter 2.6 (Tool Integration and Function Calling).
What does it mean when the chapter says "LLMs never call functions
directly"? Explain it simply, without heavy notation, with one
example. Then ask me to explain it back to you.
```

---

## Sharpening a definition (Protocol B)

**Scenario:** You keep using a term but realize you cannot define it precisely. Use this before diving into problem-solving.

```
I want to use Socratic definition mode for the term "orchestration"
as used in Chapter 1.5A. I have a rough sense of what it means but
I cannot give a precise definition.

Ask me what I think it means first. Then ask me questions from
different angles — what it does, how it differs from "coordination,"
when you would use it — and offer examples and non-examples. Only
give me your synthesized definition after I have tried. At the end,
ask me to compare my initial and final definitions.
```

```
I keep confusing "stateless" and "stateful" agents. Use Socratic
definition mode to help me nail down both terms from Chapter 1.5A.
Start by asking me to define each one, probe the definitions with
edge cases, then give me your corrected version.
```

---

## Having your reasoning challenged (Protocol C)

**Scenario:** You have worked out an explanation, a plan, or a design and want the tutor to stress-test it — not validate it.

```
I will describe my understanding of how a multi-agent handoff works
in LangGraph (Chapter 2.2). Please use Socratic elenchus mode:
ask me to lay out my reasoning step by step, then question each
step for consistency with the chapter text. Use counter-examples
where my logic breaks down. Do not tell me the correct answer until
I have found the error myself.

Here is my reasoning: [paste your explanation]
```

```
I think I understand why progressive disclosure matters for agent UIs
(Chapter 1.1A). Challenge my reasoning — ask me to state my
assumptions, then probe them one by one. If my reasoning is
inconsistent with the chapter, help me discover it through questions,
not by correcting me directly.

My reasoning: [paste your explanation]
```

---

## Exploring a design decision (Protocol D)

**Scenario:** You are choosing between architectures, frameworks, or design patterns and there is no single right answer. You want help thinking through trade-offs.

```
I am building a multi-agent system for document processing and I am
deciding between LangGraph (Chapter 2.2) and a sequential LangChain
approach (Chapter 2.3). Use Socratic dialectic mode.

Help me compare the two along key dimensions: complexity, state
management, fault tolerance, and scalability. Use "what if" scenarios
to challenge my assumptions — for example, what if my workflow needs
to branch based on document type, or what if I need to pause
mid-pipeline for human review? Ask me to argue against my preferred
option, then help me articulate a final justified choice.
```

```
I am an advanced learner on Chapter 1.7B (Hybrid RAG+KG). I need to
decide whether to use pure vector search or a hybrid RAG+Knowledge
Graph approach for a reasoning-heavy question-answering task.

Use counterfactual reasoning to stress-test my thinking: what if the
query requires multi-hop inference? What if the knowledge base changes
frequently? Help me explore the trade-offs, then ask me to commit to
a choice and defend it.
```

---

## Self-explaining a solution (Protocol E)

**Scenario:** You have a solution or worked example in front of you and want to understand it deeply, not just read it.

```
I have this code example from Chapter 2.2 on LangGraph:
[paste code snippet]

Please use a self-explanation protocol. Walk me through it step by
step. At each key step, ask me to explain WHY it is correct or
necessary — not just what it does. Ask me what would happen if that
step were removed or changed. Only give your own explanation after
I have attempted mine. Close by asking me to state the general
principle the example illustrates.
```

```
I just read the trust equation explanation in Chapter 1.1A
(Trust = Transparency × Control). I think I understand it but
I want to make sure. Ask me to explain why the relationship is
multiplicative rather than additive, what would happen if control
were zero, and how this connects to UI design decisions. Challenge
any parts of my explanation that are incomplete.
```

---

## Learning a new procedure (Protocol F)

**Scenario:** You are new to a procedure — a derivation, a coding pattern, a configuration workflow — and need to see it done fully before trying yourself.

```
I am a beginner on Chapter 2.2 (LangGraph). I have never built a
stateful workflow with nodes and edges before.

Please use a worked example protocol. First, show me a fully worked
example of a simple LangGraph workflow — not the one in the chapter —
with numbered steps and a label for what each step accomplishes.
Then give me a similar but slightly different problem where some
steps are blank for me to fill in. Gradually reduce the scaffolding
until I attempt one mostly on my own.
```

```
I need to understand how to implement a retrieval-augmented generation
pipeline from Chapter 2.7 (Multimodal RAG). Show me a worked example
of a basic RAG pipeline first, then give me a version with missing
steps I need to complete. Highlight the general strategy at each
step, not just the specific code.
```

---

## Getting unstuck with hints (Protocol G)

**Scenario:** You are actively working on a problem and have made some progress, but you are stuck on the next step. You want minimal help — not the answer.

```
I am working on understanding how context window limits affect
stateful orchestration (Chapter 1.5A). Here is as far as I have got:
[paste your current thinking or partial answer]

I am stuck on [describe exactly where you are stuck]. Please give me
only one small hint as a question — not the answer, not the next
step written out. After I respond, give me another small hint if I
need it. Escalate slowly.
```

```
I am trying to explain how LangGraph handles conditional branching
but I cannot work out how the router node decides which edge to take.
Here is what I have so far: [paste your work]

Use step-by-step hinting. Do not tell me — ask me a question that
helps me figure it out. Wait for my attempt before giving another hint.
```

---

## Building fluency with drills (Protocol H)

**Scenario:** You understand a concept but want to get faster and more confident with it — the way you would practice flashcards.

```
I understand the five agent memory types from Chapter 1.4 but I am
slow and unsure when asked to distinguish them in scenarios.

Give me one drill question at a time. I will answer from memory —
no peeking at notes. After each answer, tell me if I am correct and
explain briefly. Mix in different question types (identify the memory
type, predict the failure mode, compare two types). Occasionally
revisit questions I got wrong after a few more items. Increase
difficulty as I improve.
```

```
I want to drill on framework selection from Chapter 2.1. For each
scenario I describe, I will tell you which framework I would choose
and why. Tell me if I am right, explain any errors, and gradually
give me harder scenarios. Track which framework types I keep getting
wrong.
```

---

## Preparing for the exam (Protocol I)

**Scenario:** You are studying for the NCP-AAI exam or a course assessment and want structured practice with feedback.

```
I am preparing for the NCP-AAI exam and I am weakest on Chapters
1.3 (Multi-Agent Systems) and 2.4 (Multi-Agent Frameworks).

Give me a short quiz of exam-style questions — a mix of
multiple-choice and short scenario questions at medium difficulty.
Show me one question at a time. Wait for my answer before telling
me if I am correct. After each question, explain what I got right
or wrong and which concept it tests. At the end, tell me which
topics I should prioritize and why.
```

```
I have studied all of Parts 1 and 2. I want a full mock exam session.
Generate 10 mixed-difficulty questions covering a spread of chapters.
Track my accuracy per topic. At the end, give me a prioritized study
plan for the week before the exam, sorted by my weakest areas.
```

---

## Reflecting on a session (Protocol J)

**Scenario:** You have finished a study session and want to consolidate what you learned, check your calibration, and plan next steps.

```
We just finished working through Chapter 1.7A and 1.7B on Knowledge
Graphs. Please use a reflection protocol.

Ask me what I covered, what felt easy and what felt hard. Then ask me
to rate my confidence on each sub-topic we touched (1–5). Compare my
confidence ratings to how I actually performed in our session today.
Point out any topics where my confidence and performance diverge.
Finally, help me plan my next study session with specific goals,
resources, and a way to check my progress.
```

---

## Working on graded coursework (Protocol K)

**Scenario:** You have an assignment or take-home assessment and want the tutor's help without crossing academic integrity lines.

```
I have a graded assignment on Chapter 2.6 (Tool Integration). I am
not asking for solutions — I know that is not allowed.

Please follow an integrity-preserving protocol. Help me understand
the concepts the assignment is testing, check whether my approach
is on the right track without doing the work for me, and give me
hints if I get stuck. Do not write any code I could submit directly.
```

```
I am working on a project that involves designing a multi-agent
workflow (Chapters 1.3 and 2.4). This is graded. I want to talk
through my architecture approach — please question my design choices,
point out gaps, and suggest things to consider, but let me make the
final design decisions myself.
```

---

## Diagnosing a wrong answer (Protocol M)

**Scenario:** You got something wrong and want to understand exactly why, not just be told the right answer.

```
I answered this question: "[paste the question]"
My answer was: "[paste your answer]"
I was told it was wrong.

Please diagnose my error. Tell me what type of mistake it is —
conceptual, procedural, factual, or careless. Ask me questions to
help me discover where my reasoning went wrong rather than just
telling me the right answer. After I correct my thinking, give me
a similar question to confirm I have fixed the misconception.
```

```
I thought the difference between in-context and external memory
(Chapter 1.4) was just about storage location. I was marked wrong.

Please use error diagnosis mode. What type of error is this? Walk
me through questions that reveal what I was missing, without just
stating the correct answer upfront.
```

---

## Tackling a new concept (Protocol N)

**Scenario:** You are about to study something new and want to engage with it before being taught — the productive failure approach.

```
I have not read Chapter 1.7A yet (Knowledge Graphs). I know about
vector databases and semantic search from earlier chapters.

Please use a productive failure protocol. Give me a problem that
requires knowledge graphs to solve well, but let me attempt it
first using only what I already know. Accept my attempts even if
they are wrong. After I have tried 2–3 approaches, show me the
canonical solution and explain how it relates to what I tried.
```

```
I understand basic RAG but I have not studied hybrid RAG+KG
(Chapter 1.7B). Before you explain it, give me a scenario where
basic RAG would fail, and let me try to figure out a solution.
After I have worked through it, teach me the hybrid approach and
connect it to my attempts.
```

---

## When you feel stuck or frustrated (Protocol O)

**Scenario:** You are not making progress, feel lost, or are losing motivation. Tell the tutor directly so it can adjust.

```
I have been reading Chapter 1.5B (Stateful Orchestration Worked
Examples) for 30 minutes and I feel like I am going in circles.
I am frustrated and not sure what I do not understand.

Please slow down and help me identify exactly where I am getting
confused. Start with the most basic sub-question — do not assume
I know anything about this section. Give me one very small, concrete
thing to do and check in with me after.
```

```
I feel like the material on Chapter 2.5 (Semantic Kernel) is too
easy compared to what we did on LangGraph. I am bored and not
really engaging.

Please switch to a more challenging mode — give me a hard design
problem, a counter-intuitive scenario, or a debate question that
makes me actually think. Increase the difficulty and ask me to
defend my answers.
```

---

## Group study session (Protocol P)

**Scenario:** You and a few classmates are studying together and want the tutor to facilitate without dominating.

```
There are three of us studying Chapter 1.3 (Multi-Agent Systems)
together. We are going to discuss the differences between
peer-to-peer and hierarchical multi-agent architectures.

Please act as a facilitator, not a lecturer. Monitor our discussion
and only interrupt if we get stuck, go off track, or leave someone
out. When you do intervene, use one short question — do not explain
or lecture. At the end of our discussion, help us reflect on what
we agreed on and what is still unresolved.
```

---

## Chaining protocols across a session

These examples show how to string protocols together into a full study arc.

**Arc 1 — Encountering a new concept for the first time**
```
Session goal: understand hybrid RAG+KG (Chapter 1.7B) from scratch.

Step 1 — I have not studied this yet. Use productive failure (Protocol N):
give me a problem and let me attempt it before you teach me.

[After the attempt:]
Step 2 — Now explain the canonical approach and compare it to my
attempts (Protocol A). Then walk me through the key steps of the
chapter's worked example and ask me to self-explain each one (Protocol E).

[After the explanation:]
Step 3 — Give me a drill on the key concepts from this chapter
(Protocol H) — 5 short questions, one at a time.

[At the end:]
Step 4 — Help me reflect on what I learned today and plan what to
review next (Protocol J).
```

**Arc 2 — Preparing for the exam on a weak topic**
```
I performed poorly on multi-agent coordination topics in practice.
Here is my plan for this session — please follow it:

1. Quiz me on Chapter 1.3 and 2.4 to see where my gaps are (Protocol I).
2. For each question I get wrong, diagnose my error type and help me
   self-correct through questions (Protocol M → C).
3. Give me a worked example on the topic I struggle with most (Protocol F).
4. Re-quiz me on those same topics to confirm improvement (Protocol I).
5. End with a reflection and a study plan for the next two days (Protocol J).
```

**Arc 3 — Debugging a design decision**
```
I am designing a production multi-agent pipeline and I think my
architecture is sound, but I want it stress-tested.

1. I will describe my architecture. Question my reasoning for
   correctness and consistency (Protocol C).
2. Then help me explore the alternatives I did not consider —
   use "what if" scenarios and trade-off questions (Protocol D).
3. If you find any significant errors in my reasoning, diagnose
   them and help me correct them through questions (Protocol M).
4. Once I have a refined design, ask me to self-explain why each
   component is the right choice (Protocol E).
```

---

## Common mistakes to avoid

**Too vague:**
> "Help me with Chapter 1.3."

The tutor does not know your level, what you have already tried, or what kind of help you want. Be specific.

**Asking for the answer immediately:**
> "What is the correct answer to this question: [paste question]"

The tutor will redirect you. Always show your attempt first, even if it is just a rough guess, before asking for help.

**Skipping the attempt on graded work:**
> "This is my homework assignment. Write the code for me."

This triggers Protocol K. The tutor will decline and offer to help you understand the problem and plan your approach instead.

**Not stating your level:**
> "Explain knowledge graphs."

Without knowing your level, the tutor will have to guess. An intermediate explanation may be too fast for a beginner or too slow for an advanced learner. Always state your level.

**Pasting an active exam question:**
> "Here is question 4 from my open-book exam that I am taking right now: ..."

The tutor will decline and offer to study the underlying concepts with you after the exam.

**Asking about content outside Parts 1–2:**
> "Explain Chapter 7.3 on Agent Toolkits."

The tutor does not have access to Parts 3–10. It will tell you this and offer to help with what it does cover.
