# AI 101 Course Candidates — Detailed Dataset

Compiled 2026-07-16 per `plans/course_search_plan.md`. Pass 1: 7 parallel search tracks →
37 raw entries → 28 unique courses after cross-track dedup + independent QA (URL, fee,
launch date, instructor). Pass 2 (`plans/course_gap_plan.md`): 6 gap tracks over elite/flagship
universities the first pass under-covered → 18 additional QA-verified courses (section D).
Total: 46 courses.

Eligibility: university professor-led or credible non-profit; free or ≤ ~$150; established
2024+ (priority 2025/2026); introductory; delivers one or more of O1–O3; covers ≥1 of C1–C6.
Machine-readable mapping: `course_inventory.csv`.

Legend: C1 science of AI · C2 AI system features · C3 government benefit · C4 risks
(discrimination/privacy) · C5 risk mitigation/trustworthy AI · C6 future trends/natsec ·
O1 apply in role · O2 critically assess/intervene · O3 governance (security/privacy/civil rights)

---

## A. University courses

### AI Foundations for Business Professionals (3-course specialization)
- Host: Saïd Business School, University of Oxford (Coursera) | Type: university | Program: Oxford 2025 Coursera/edX launch (21 courses)
- Instructor(s): Prof. Matthias Holweg, Professor of Operations Management
- Fee: audit-free per course; cert ~$49/mo subscription, financial aid | Established: 2025 (Oxford SBS news: courses released June–July 2025) | Duration: ~40 hrs
- Format: self-paced video + graded assignments | URL: https://www.coursera.org/specializations/ai-foundations-business-professionals
- Cs: C1 ML/neural networks/deep learning fundamentals; C2 task-specific to generative tools; C4 risk assessment, bias mitigation; C5 trustworthy-AI governance frameworks; C6 agentic AI capabilities/risks | Os: O1 harness AI for value in role; O2 evaluate risks, safe/fair/accountable AI; O3 governance, compliance, data ethics
- Notes: Beginner, no prior experience; 21k+ enrolled, 4.8/5; includes the standalone AI Governance course below.

### AI Governance
- Host: Saïd Business School, University of Oxford (Coursera) | Type: university | Program: AI Foundations for Business Professionals specialization (standalone-enrollable)
- Instructor(s): Prof. Matthias Holweg
- Fee: audit-free; cert ~$49 | Established: 2025 (archive.org first capture Jul 23, 2025; first reviews Nov 2025) | Duration: ~2 weeks
- Format: self-paced video + peer-review assignment | URL: https://www.coursera.org/learn/ai-governance
- Cs: C1 why/how AI systems fail; C4 bias, misalignment, overreliance; C5 Trustworthy AI Cycle, risk management, red-teaming; C6 implementation strategies ahead | Os: O1 implementation/build-vs-buy decisions; O2 failure modes and overreliance; O3 accountability across AI lifecycle
- Notes: Beginner; 12k+ enrolled, 4.8/5; strongest single-course C5 treatment among university MOOC entries.

### Foundations of Generative AI
- Host: Georgia Institute of Technology (GTx, edX) | Type: university | Program: none
- Instructor(s): Dr. David Joyner, Executive Director of Online Education & OMSCS
- Fee: audit-free; verified cert $90 | Established: 2025 (edX press release Oct 16, 2025) | Duration: 3 weeks, 2–3 hrs/wk
- Format: self-paced video + modules | URL: https://www.edx.org/learn/artificial-intelligence/the-georgia-institute-of-technology-introduction-to-genai
- Cs: C1 history of AI, neural networks, transformers; C2 what generative AI is; C4 ethics of model alignment; C5 aligning generative AI module; C6 why GenAI matters, state of the art | Os: O1 understand tools professionals rely on; O2 "driver's ed" AI literacy — understand, not just operate
- Notes: 10 modules, no prior experience required; clean conceptual C1 coverage.

### AI Literacy for Everyone (6-course specialization)
- Host: University of Michigan (Coursera) | Type: university | Program: U-M Center for Academic Innovation GenAI series (35+ courses)
- Instructor(s): U-M multidisciplinary faculty (law, NLP, economics, communication)
- Fee: audit-free per course; cert via ~$49/mo subscription | Established: 2024 (U-M CAI: 35+ GenAI courses launching by July 2024) | Duration: ~30 hrs
- Format: self-paced video + assignments | URL: https://www.coursera.org/specializations/ai-literacy-for-everyone
- Cs: C1 how GenAI/ML models work; C2 GenAI tools, prompt engineering; C3 GenAI in business (transferable to gov); C4 ethical/legal implications; C5 responsible AI practices, regulation; C6 labor and future of work | Os: O1 integrate GenAI into daily activities; O2 GenAI for critical thinking; O3 ethics, authorship, regulation
- Notes: Beginner; 4.7/5 (539 reviews); broadest C/O coverage of any single candidate; anchor course dates to Dec 2023 — the specialization is the 2024+ unit.

### Generative AI: Governance, Policy, and Emerging Regulation
- Host: University of Michigan (Coursera) | Type: university | Program: Responsible Generative AI specialization
- Instructor(s): Merve Hickok, Lecturer, U-M School of Information; President, Center for AI and Digital Policy
- Fee: audit-free; cert ~$49 | Established: 2024 (U-M CAI 2024 launch; first reviews Jul 2024) | Duration: ~3 hrs
- Format: self-paced video + readings | URL: https://www.coursera.org/learn/generative-ai-governance-policy-and-emerging-regulation
- Cs: C4 risk and impact assessments; C5 responsible-AI principles, transparency; C6 emerging US/EU/G7 regulation | Os: O1 governance decisions for AI adoption; O3 policy, regulation, accountability
- Notes: Caveat — Coursera tags it "intermediate" (one track excluded it on that basis) but content is short, non-technical, and review-confirmed accessible; nationally recognized AI-policy instructor.

### AI Fundamentals
- Host: University of Leeds (FutureLearn) | Type: university | Program: Click Start (Institute of Coding + Nominet)
- Instructor(s): Dr. Sam Wilson, Associate Professor, School of Computer Science
- Fee: free; free digital upgrade via Click Start (standard $54) | Established: 2024 (archive.org 2024-09-19) | Duration: 2 weeks, 4 hrs/wk
- Format: self-paced video + activities + assessment | URL: https://www.futurelearn.com/courses/how-to-get-into-ai
- Cs: C1 AI evolution 1940s→LLMs; C2 hands-on AI tools; C4 ethical/legal/social challenges; C6 future impact | Os: O1 workplace productivity with AI; O2 responsible-use ethics activities
- Notes: Short non-technical on-ramp; nonprofit-funded free certificate route is rare on FutureLearn.

### Unleash Your Potential: AI Fundamentals
- Host: University of Bristol (FutureLearn) | Type: university | Program: AI for Collective Intelligence (AI4CI) hub
- Instructor(s): Prof. Seth Bullock (Toshiba Chair in Data Science & AI); Prof. Genevieve Liveley
- Fee: free (free digital upgrade) | Established: 2026 (Bristol news Feb 2026) | Duration: 4 weeks, ~3 hrs/wk
- Format: self-paced video + case studies | URL: https://www.futurelearn.com/courses/unleash-your-potential-ai-fundamentals
- Cs: C1 fundamental concepts, how AI works; C2 cross-sector case studies; C3 applications across sectors (indirect); C4 ethics, risks, bias; C5 AI safety, sustainability | Os: O1 engage with AI at work; O2 think critically about AI capability
- Notes: Newest UK entry (Feb 2026); contributors span law, cybersecurity, medicine, philosophy.

### Foundation Models and Generative AI (6.S087)
- Host: MIT OpenCourseWare | Type: university | Program: MIT EECS / OCW
- Instructor(s): Rickard Brüel Gabrielsson, MIT EECS/CSAIL
- Fee: free (no certificate) | Established: 2024 (IAP 2024; OCW-published 2024) | Duration: ~4-week lecture series
- Format: self-paced lecture videos | URL: https://ocw.mit.edu/courses/6-s087-foundation-models-and-generative-ai-january-iap-2024/
- Cs: C1 self-supervised learning → foundation models; C2 ChatGPT/DALL-E/Copilot systems; C6 foundation models changing science/business | Os: O1 applications in business/science workflows
- Notes: "Non-technical series of lectures... all backgrounds welcome"; strongest C1 depth in the dataset.

### Free Online Certificate in Artificial Intelligence and Career Empowerment
- Host: University of Maryland, Robert H. Smith School of Business | Type: university | Program: Smith Executive Education
- Instructor(s): Balaji Padmanabhan, Dean's Professor of Decision, Operations & IT (+11 Smith faculty)
- Fee: free, certificate included | Established: May 1, 2025 (PRNewswire; Maryland Today) | Duration: 10 self-paced modules
- Format: self-paced video + industry-expert interviews | URL: https://www.rhsmith.umd.edu/programs/executive-education/learning-opportunities-individuals/free-online-certificate-artificial-intelligence-and-career-empowerment
- Cs: C1 AI literacy/capabilities; C2 how AI solutions are built; C3 built for transitioning federal civil-service workers; C5 responsible AI module | Os: O1 apply AI across functions/careers; O2 assess AI in workflows; O3 responsible AI in organizations
- Notes: Explicitly aimed at federal workers in career transition; 37,000+ enrolled by Aug 2025.

### GenAI 101
- Host: Indiana University, Kelley School of Business | Type: university | Program: Kelley Learn AI initiative
- Instructor(s): Brian Williams, Sam Frumer Professor of Accounting
- Fee: free, digital badge included | Established: Aug 2025 (IU); worldwide Apr 2026 (IU News) | Duration: 8 modules / 16 lessons
- Format: self-paced short videos + interactive AI tutor | URL: https://kelley.iu.edu/learnAI
- Cs: C2 prompt engineering across ChatGPT/Claude/Gemini; C5 ethical AI use | Os: O1 20 career-ready workplace skills; O2 fact-checking AI-generated content
- Notes: No technical background needed; shareable LinkedIn credential.

### AI Whisperer: A Microcourse in Crafting Prompts for Generative AI
- Host: University of South Florida, Bellini College of AI, Cybersecurity and Computing | Type: university | Program: USF Microcredentials
- Instructor(s): John Licato, Associate Professor (with Profs. Tempestt Neal, Sudeep Sarkar)
- Fee: free; optional $39 digital badge | Established: Oct 28, 2025 (USF release) | Duration: 3–4 hrs
- Format: self-paced on Canvas Network | URL: https://www.usf.edu/innovative-education/usf-microcredentials/a-microcourse-in-crafting-prompts.aspx
- Cs: C1 ML/GenAI fundamentals unit; C2 tool selection + prompt engineering | Os: O1 generate role-relevant reports/presentations; O2 judging output quality
- Notes: Shortest commitment in dataset; good first course; companion "GenAI in Action" microcourse from same college.

### Introduction to AI Literacy
- Host: Acadia University (Open Acadia) | Type: university | Program: Open Acadia continuing education
- Instructor(s): Dr. Daniel Lametti, Department of Psychology
- Fee: free, certificate included | Established: early 2026 (Acadia newsroom Mar 5, 2026; CBC) | Duration: ~2.5 hrs
- Format: self-paced module (lecture, podcast, forum) | URL: https://explore.acadiau.ca/product?catalog=AI-Literacy
- Cs: C1 what AI is, how it works; C2 ChatGPT/Claude-type tools; C4 limitations, over-reliance risks; C5 responsible use | Os: O1 when/how to use AI at work; O2 human-vs-AI thinking differences
- Notes: Cognitive-science angle directly supports O2; 5,000+ enrollments in first two weeks.

### Trustworthy AI: Managing Bias, Ethics, and Accountability
- Host: Johns Hopkins University (Coursera) | Type: university | Program: JHU AI series
- Instructor(s): Ian McCulloh, PhD, JHU senior scientist/lecturer
- Fee: audit-free; cert ~$49 | Established: late 2024/Jan 2025 (archive.org 2025-01-09) | Duration: ~2 weeks, 4 modules
- Format: self-paced video + 9 assignments | URL: https://www.coursera.org/learn/responsible-ai-and-ethics
- Cs: C1 human vs. machine bias mechanics; C4 algorithmic bias, privacy, drift; C5 risk-based responsible AI, regulation | Os: O2 fairer risk assessment of AI outputs; O3 bias, privacy, regulation, accountability
- Notes: Non-technical, case-study driven; listed "intermediate" but no formal prerequisites; strong disparate-impact anchor coverage.

### Artificial Intelligence in National Security
- Host: King's College London, Dept. of War Studies (edX) | Type: university | Program: Professional Certificate in Grand Strategy (standalone-friendly)
- Instructor(s): Prof. Kenneth Payne, Professor of Strategy
- Fee: audit-free; cert $249 (over cap — audit route qualifies) | Established: Jul 3, 2025 (KCL catalogue; archive.org 2025-06-23) | Duration: 5 hrs
- Format: self-paced online short course | URL: https://www.edx.org/learn/social-sciences/kings-college-london-artificial-intelligence-in-national-security
- Cs: C2 autonomy, data, AI decision systems; C3 strategic AI in defense/statecraft; C4 ethical/legal challenges; C6 AI arms race, state competition | Os: O2 human decision-making vs. autonomy; O3 ethics/law/governance of AI security systems
- Notes: Leading war-studies scholar (author of "I, Warbot"); directly covers autonomous-systems and natsec-policy concept anchors.

### Generative AI Cybersecurity & Privacy for Leaders: A Primer
- Host: Vanderbilt University (Coursera) | Type: university | Program: GenAI Cybersecurity & Privacy for Leaders specialization
- Instructor(s): Dr. Jules White, Professor of Computer Science & Associate Dean; Dr. Sam Hays
- Fee: audit-free; paid cert optional | Established: Dec 2024–Jan 2025 (archive.org 2025-01-17) | Duration: 4 hrs, 5 modules
- Format: self-paced video + prompt practice | URL: https://www.coursera.org/learn/generative-ai-security-privacy
- Cs: C2 GenAI features, secure prompting; C4 misinformation, deepfakes, phishing, privacy; C5 organizational preparedness, tabletop exercises; C6 deepfake/persuasion threats | Os: O1 safe GenAI prompting at work; O2 recognizing deepfakes/misinformation; O3 individual + organizational privacy/security governance
- Notes: Beginner, 4.8/5; strongest synthetic-media anchor coverage; framed for non-technical leaders.

### AI Literacy for Life & Work
- Host: Grand Valley State University, College of Computing | Type: university | Program: GVSU community course/certificate
- Instructor(s): Marouane Kessentini, Dean of College of Computing (+4 faculty)
- Fee: free; optional paid digital certificate/badge | Established: Mar 2, 2026 (GVNext 2026-02-24) | Duration: 7 modules, 8–10 hrs
- Format: self-paced videos + readings + activities | URL: https://www.gvsu.edu/computing/ai-community-coursecertificate-258
- Cs: C1 how AI systems function; C2 core concepts, communicating with AI; C4 bias, ethical/environmental implications; C5 responsible use | Os: O1 workplace applications module; O2 evaluating accuracy and bias in outputs
- Notes: Brand-new 2026 launch, dean-level backing, no coding required.

## B. University–non-profit partnerships

### AI and Digital Transformation in Government
- Host: University of Oxford Saïd Business School + UNESCO (SPARK-AI Alliance) | Type: university/intergovernmental | Program: Oxford Saïd Online / UNESCO AI for the Public Sector
- Instructor(s): Oxford Saïd faculty team (program lead: Caroline Williams, Executive Director, Oxford Saïd Online; no single named professor)
- Fee: free (UNESCO-funded); joint Oxford/UNESCO certificate | Established: Nov 10, 2025 (UNESCO announcement) | Duration: ~12 hrs
- Format: self-paced online; English/Spanish (more languages following) | URL: https://www.sbs.ox.ac.uk/programmes/executive-education/online-learning/ai-and-digital-transformation-government
- Cs: C2 hands-on generative AI tools; C3 government service delivery, inclusive design; C4 AI & Human Rights module; C5 ethics + data governance modules; C6 leadership for digital transformation | Os: O1 apply GenAI in public-sector work; O2 rights-based assessment of automated decisions; O3 human rights, data governance, inclusion
- Notes: Surfaced independently by 4 of 7 search tracks — the single strongest civil-servant fit in the dataset; 30,000+ learners in 192 countries by mid-2026; benefits framing is general-government rather than U.S.-federal.

### AI, Justice and Rule of Law
- Host: UNESCO + University of Oxford (Saïd, Blavatnik School, Faculty of Law) | Type: university/intergovernmental | Program: UNESCO AI and Rule of Law initiative
- Instructor(s): Prof. Ignacio Cofone, Professor of Law and Regulation of AI (academic director)
- Fee: free | Established: Apr 27, 2026 (English edition; UNESCO / law.ox.ac.uk) | Duration: ~18 hrs, 6 modules
- Format: self-paced MOOC | URL: https://secure.sbs.ox.ac.uk/corporate/landingPage.do?method=load&corporateGroupId=7948447 (announcement: https://www.law.ox.ac.uk/content/news/faculty-law-co-creates-new-online-course-ai-justice-and-rule-law)
- Cs: C4 bias, fair-trial and AI-evidence risks; C5 safeguards, transparency, reasoned-decision rights; C6 AI decision-support in future justice systems | Os: O2 assessing algorithmic outputs, preventing over-reliance; O3 AI, law, human-rights governance
- Notes: Aimed at judges/lawyers/policymakers but open globally, no technical background; French/Spanish editions June 2026.

## C. Non-profit courses

### Responsible AI for Public Professionals
- Host: InnovateUS (Burnes Center for Social Change, Northeastern University + The GovLab) | Type: nonprofit | Program: InnovateUS at-your-own-pace courses
- Instructor(s): Program led by Beth Simone Noveck, Professor, Northeastern; NJ State Chief AI Strategist (100+ expert contributors)
- Fee: free, certificate included | Established: Jul 2024 (InnovateUS launch announcement Jul 11, 2024) | Duration: 2 parts, ~1 hr 45 min (24 short videos)
- Format: self-paced video + hands-on GenAI demos (platform-agnostic) | URL: https://innovate-us.org/course/responsible-ai-for-public-professionals/ (Part 1 also standalone as "Using Generative AI at Work")
- Cs: C1 what generative AI is; C2 GenAI tools and best practices; C3 day-to-day government work; C4 risks, bias, privacy, deepfakes; C5 responsible-use practices | Os: O1 choose suitable work tasks for GenAI; O2 protect sensitive data, verify outputs; O3 security, ethics, societal challenges
- Notes: Adopted statewide by NJ, NY (100k+ employees), Georgia, DC (first city to mandate responsible-AI training); refreshed every six months; WCAG 2.1 accessible.

### Responsible AI for Public Organizations
- Host: InnovateUS (Burnes Center, Northeastern University) | Type: nonprofit | Program: InnovateUS at-your-own-pace courses
- Instructor(s): Burnes Center/GovLab team led by Beth Simone Noveck
- Fee: free, certificate | Established: 2025 (QA: absent from InnovateUS's Oct 2024 course lineup; first archive capture Aug 2025) | Duration: 2 parts, ~2 hrs
- Format: self-paced video + worksheets | URL: https://innovate-us.org/course/responsible-ai-for-public-organizations/
- Cs: C1 overview of AI approaches; C2 AI foundations, data quality; C3 surfacing/selecting agency AI projects; C4 GenAI-specific risks; C5 NIST AI RMF, trustworthy-AI characteristics, ethical toolkits | Os: O1 design/implement AI projects in one's agency; O2 apply risk frameworks pre-deployment; O3 governance, people & talent, ethics toolkits
- Notes: One of very few free courses teaching the NIST AI RMF directly to government staff; organizational companion to the individual-level course.

### Responsible AI for Public Sector Legal Professionals
- Host: InnovateUS (with AI for Impact, Burnes Center, Reboot Democracy, GovLab) | Type: nonprofit | Program: InnovateUS at-your-own-pace courses
- Instructor(s): Curriculum led by Burnes Center team (Beth Simone Noveck); 20+ senior public-sector legal advisors
- Fee: free; CLE-eligible in some states | Established: summer 2025 (rebootdemocracy.ai) | Duration: 2 parts, self-paced
- Format: self-paced video + hands-on activities | URL: https://innovate-us.org/course/responsible-ai-for-public-sector-legal-professionals/
- Cs: C1 GenAI and ML explained; C2 day-to-day GenAI demos; C3 AI in government law practice; C4 risks, sensitive-information protection; C5 writing AI governance policy; C6 societal challenges | Os: O1 GenAI in government legal workflows; O2 key questions for responsible use; O3 data governance, policy writing
- Notes: Promoted via National Association of Attorneys General; directly relevant to human oversight of automated processes.

### A Human Rights-Based Approach to AI
- Host: Global Campus of Human Rights (with Danish Institute for Human Rights, Queen's University Belfast, Raoul Wallenberg Institute) | Type: nonprofit | Program: GC Human Rights MOOCs
- Instructor(s): Thérèse Murphy (QUB), Line Gamrath Rasmussen (DIHR), Sue Anne Teo (RWI); incl. David Kaye (UC Irvine)
- Fee: free (no fees) | Established: first edition Feb 17–Mar 23, 2025 | Duration: 5 weeks, ~6 hrs
- Format: cohort MOOC (materials remain open between editions) | URL: https://elearning.gchumanrights.org/courses/course-v1:gchumanrights+HRBAtoAI+2025/about
- Cs: C2 generative AI module; C3 AI in the public sector module; C4 AI and human-rights harms; C5 rights-based approach in action | Os: O2 scrutinizing automated systems' rights impacts; O3 civil rights, discrimination, public-sector governance
- Notes: Strongest civil-rights framing in the dataset; QA (Jul 2026): 2025-edition enrollment currently closed, no 2026 edition posted yet — monitor for next edition.

### AI Safety, Ethics, and Society
- Host: Center for AI Safety (CAIS) | Type: nonprofit | Program: AI Safety, Ethics, and Society textbook course
- Instructor(s): Dan Hendrycks, Director, CAIS (textbook author); facilitated discussions
- Fee: free (application-based cohorts) | Established: 2024 (first cohort Jul–Oct 2024, 240 participants) | Duration: ~13 weeks, ~5 hrs/wk
- Format: virtual cohort; readings + lectures + discussions + capstone | URL: https://www.aisafetybook.com/virtual-course
- Cs: C1 AI fundamentals, deep learning, scaling laws; C4 malicious use, bias, rogue-AI risks; C5 safety engineering, risk management; C6 national/international/compute governance | Os: O2 monitoring, robustness, human control; O3 corporate/national/international governance
- Notes: Designed for non-technical audiences (optional fundamentals week); Routledge textbook companion; strong natsec framing.

### Responsible AI Foundations (course series)
- Host: All Tech Is Human (on LinkedIn Learning, unlocked free) | Type: nonprofit | Program: Responsible AI Foundations Professional Certificate
- Instructor(s): Renée Cummings, Professor of Practice in Data Science, University of Virginia; Rebekah Tweed, ATIH Executive Director
- Fee: free (Patrick J. McGovern Foundation-funded) | Established: Oct 2025 (5 courses); certificate path Feb 2026 | Duration: a few hours, 4–5 short courses
- Format: self-paced video | URL: https://alltechishuman.org/rai-courses
- Cs: C4 identifying/understanding AI risks; C5 mitigating risks, operationalizing governance; C6 governing agentic AI | Os: O2 risk identification and mitigation practice; O3 AI governance landscape and roles
- Notes: Practitioner-oriented, no prerequisites; suits professionals building governance programs.

### AI in National Security: Integrating Artificial Intelligence into Public Sector Missions
- Host: Special Competitive Studies Project (SCSP) (Coursera) | Type: nonprofit | Program: AI + AGI in National Security microcredential (3 free courses)
- Instructor(s): Ylli Bajraktari (SCSP President, former NSCAI ED); Lt Gen (ret.) Jack Shanahan (first DoD JAIC Director)
- Fee: free | Established: Mar 2025 (PRNewswire; archive.org 2025-03-11) | Duration: 5 hrs, 4 modules
- Format: self-paced video + hands-on exercises | URL: https://www.coursera.org/learn/ai-national-security
- Cs: C1 defines AI in practical-applications module; C2 GenAI tool demos; C3 public-sector mission applications, Agency AI Playbook; C5 responsible-adoption playbook; C6 AI megatrends in national security | Os: O1 leverage AI tools for your mission; O3 agency AI adoption strategy
- Notes: Purpose-built for federal/natsec workforce; nonprofit successor to NSCAI; 6.9k enrolled, 4.6/5; closest single match to the C6 mandate.

### The Future of AI
- Host: BlueDot Impact | Type: nonprofit | Program: BlueDot AI safety portfolio
- Instructor(s): BlueDot Impact course team
- Fee: free (certificate on completion) | Established: Mar 2025 (launch post; archive.org 2025-04-26) | Duration: ~2 hrs, 4 units
- Format: self-paced, videos + interactive demos | URL: https://bluedot.org/courses/future-of-ai
- Cs: C1 current AI capabilities; C4 AI-enabled cyberattacks, authoritarian misuse; C6 AI trajectory next decade | Os: O2 critically weighing capability/risk claims; O3 societal resilience, policy engagement
- Notes: 2-hour on-ramp for complete non-specialists before deeper study.

### AGI Strategy
- Host: BlueDot Impact | Type: nonprofit | Program: BlueDot AI safety portfolio
- Instructor(s): BlueDot teaching fellows (AI safety/governance practitioners; cohorts of ~8)
- Fee: free / pay-what-you-want | Established: 2025 (archive.org 2025-08-25) | Duration: ~25 hrs (5 weeks or 5-day intensive)
- Format: cohort — readings + weekly facilitated discussions | URL: https://bluedot.org/courses/agi-strategy
- Cs: C4 threat analysis of advanced-AI misuse; C5 strategic risk-reduction interventions; C6 AGI trajectory, lab-state competition | Os: O2 stress-testing AI development claims; O3 governance/security strategy for policy professionals
- Notes: Recruits policy, security, law, intelligence professionals; application-based scheduling; strongest live-cohort option in the natsec niche.

### It's Not Just Business – AI Risks, Rewards and Responsibilities
- Host: The Alan Turing Institute | Type: nonprofit | Program: Turing Online Learning Platform (BridgeAI)
- Instructor(s): Dr. A.E. Jaques and Dr. Milo Phillips-Brown (course authors)
- Fee: free (CC-BY-4.0 materials) | Established: Nov 13, 2024 (Zenodo record 14137369) | Duration: 1–5 hrs
- Format: self-paced online + open workshop materials | URL: https://www.turing.ac.uk/courses/its-not-just-business-ai-risks-rewards-and-responsibilities
- Cs: C2 generative AI worked use case; C4 anticipating social consequences; C5 CAIR3 responsible design-develop-deploy framework | Os: O2 pre-empting unacceptable deployment risks; O3 social/environmental responsibility governance
- Notes: UK national AI institute; narrower scope than full literacy surveys — best as a governance supplement.

---

## Near-misses (deduplicated; failed exactly one criterion)

Launch date (pre-2024, no dated relaunch): Elements of AI (Helsinki, 2018); Ethics of AI (Helsinki, 2020); AI: Ethics & Societal Challenges (Lund, ~2020); AI Strategy and Governance (Wharton, 2021); Trustworthy Generative AI (Vanderbilt, Nov 2023); Generative AI Essentials (U-M standalone, Dec 2023); AI Ethics and Governance (Turing Commons, 2022); Media Literacy in the Age of Deepfakes (MIT, 2021); AI in Society (Helsinki/Una Europa, 2022); An Introduction to AI (OpenLearn, 2022); GenAI in Action v2.0 (USF — relaunch not documented); AI Government Leadership Program (Partnership for Public Service, orig. 2019); AI for Everyone (Universiti Malaya — no verifiable date).

Cost: AI for Work and Life (U. North Florida, now $249); NACo AI Leadership Academy (~$1,000); AI for the Public Sector (Georgia Tech PE, $795); AI Security XACS134 (Stanford, $545); HarvardX ML/AI with Python (cert $299 + prerequisites).

Prerequisites/level: Responsible and Ethical AI (Northeastern, Python required); Explainable AI Specialization (Duke, intermediate); Frontier AI Governance (BlueDot, requires AGI Strategy); Intro to AI for Cybersecurity (JHU, intermediate).

Access restrictions: AI Essentials (Holyoke CC, MA residents); Free Workforce AI (Mississippi Gulf Coast CC — Intel vendor curriculum); GovEx AI Track (JHU, selected cities only); FPF trainings (member-only); CWCT AI Literacy (Purdue NW, restricted eligibility); Getting Started with GenAI (OU Click Start, UK 18–30, closed); NPS CHDS self-study (gov-affiliated only); Horizon AI Policy Workshop (selective application); Government AI Campus (Apolitical — for-profit provider); AI Fluency (OpenLearn — Microsoft-written content).

---

## D. Gap-analysis additions (2026-07-16, per plans/course_gap_plan.md)

18 courses found by the 6-track gap sweep (Stanford/Berkeley/CMU; Harvard/MIT; Ivy+ privates;
state flagships; international elites; platform newest-first recheck). All entries live-fetch
verified by the finding agent and re-checked by a 2-agent QA pass (fees/enrollability as of
Jul 2026). Note: Coursera has removed free-audit from some newer courses — three entries below
are Coursera Plus/paid-enrollment (~$59/mo, 7-day free trial) and are marked as such.

### AI Fundamentals for Public Servants: Opportunities, Risks and Strategies
- Host: Stanford Online + Stanford HAI (delivered on Apolitical) | Type: university | Program: Stanford HAI Policymaker Education
- Instructor(s): Stanford HAI faculty, instructors, and fellows (individual names not published on public pages)
- Fee: free (no-cost places, first-come first-served) | Established: Aug 12, 2024 (Apolitical press release; archive.org 2024-09-05) | Duration: self-paced microcourse
- Format: self-paced video (Apolitical platform) | URL: https://apolitical.co/microcourses/en/ai-fundamentals-for-public-servants-opportunities-risks-and-strategies/
- Cs: C1 how AI works; C2 capabilities/limitations; C3 government use cases; C4 bias, trust, risks; C5 human-centered ethical framework; C6 regulatory changes ahead | Os: O1 leverage AI in government work; O2 understand limitations and risks; O3 regulation, bias, ethics
- Notes: 1,700+ public servants enrolled (HAI news Jan 2025). QA: requires a free Apolitical account restricted to public servants/policymakers — fits this dataset's audience but not fully open to the general public.

### The AI Awakening: Implications for the Economy and Society
- Host: Stanford School of Engineering / Stanford Online (Coursera) | Type: university | Program: Stanford Online (SOE-YCS0028)
- Instructor(s): Erik Brynjolfsson, Professor, Stanford; Director, Stanford Digital Economy Lab
- Fee: QA-corrected — Coursera Plus/paid enrollment (~$59/mo, 7-day free trial); free audit no longer offered | Established: 2024 (archive.org 2024-01-11) | Duration: ~4 hrs, 3 modules
- Format: self-paced video | URL: https://www.coursera.org/learn/ai-awakening
- Cs: C1 technology behind generative AI; C2 data and human role in models; C3 (indirect) economy/workforce policy insight; C4 risks of generative AI; C6 near-future transformation | Os: O1 how organizations use GenAI; O2 capabilities and limits
- Notes: 12k enrolled, 4.7/5; guests incl. Eric Schmidt, Mira Murati; light on governance (no C5/O3).

### The Science and Implications of Generative AI (open course archive)
- Host: Harvard Kennedy School | Type: university | Program: DPI-681M course family
- Instructor(s): Sharad Goel, Professor of Public Policy; Dan Levy; Teddy Svoronos (all HKS)
- Fee: free (CC BY 4.0 open archive) | Established: Spring 2024 (HKS news Apr 23, 2024) | Duration: 11 sessions, ~15.5 hrs video + materials
- Format: self-paced open archive — lecture videos, pre-class materials, practitioner discussions | URL: https://generative-ai-course.hks.harvard.edu/spring-2024
- Cs: C1 LLMs, deep neural networks; C2 prompting, RAG, fine-tuning; C3 policymaker audience, public-sector cases; C4 misuse, misinformation, inequality; C5 alignment, mitigation, copyright; C6 societal disruption, future of work | Os: O1 prompt engineering, when to implement; O2 limitations, decision criteria; O3 copyright, policy, disinformation governance
- Notes: Strongest gap find — purpose-built for non-technical public-sector leaders by three HKS faculty; static archive (no cohort/certificate); live exec version costs $1,995 (near-miss).

### How to AI (Almost) Anything (MAS.S60)
- Host: MIT Media Arts and Sciences (via MIT OCW) | Type: university | Program: MIT OCW
- Instructor(s): Prof. Paul Liang, MIT Media Lab
- Fee: free (OCW, CC-licensed) | Established: Spring 2025 (OCW page; public course site + lecture videos) | Duration: full semester, ~26 lectures
- Format: self-paced OCW — slides, videos, readings | URL: https://ocw.mit.edu/courses/mas-s60-how-to-ai-almost-anything-spring-2025/
- Cs: C1 deep learning and foundation models; C2 multimodal AI (language, vision, audio, sensors); C3 (indirect) applying AI to real-world data; C4 faulty rewards, jailbreaks; C5 safety and reliability week; C6 frontier multimodal/agentic AI | Os: O1 apply AI to own domain/data; O2 critical thinking when applying AI
- Notes: Caveat — graduate pacing with research component; no governance module (O3 absent); best for technically curious professionals, weaker pure-AI-101 fit.

### AI in Education: Leveraging ChatGPT for Teaching
- Host: Wharton Online, University of Pennsylvania (Coursera) | Type: university | Program: none
- Instructor(s): Ethan Mollick, Associate Professor, Wharton; Dr. Lilach Mollick, Wharton Generative AI Labs
- Fee: QA-corrected — Coursera Plus/paid enrollment (~$59/mo, 7-day free trial); free audit not confirmed | Established: Nov 21, 2024 (Wharton press release) | Duration: ~6 hrs, 4 modules
- Format: self-paced video, beginner | URL: https://www.coursera.org/learn/ai-in-education-leveraging-chatgpt-for-teaching
- Cs: C1 GenAI/LLM fundamentals; C2 ChatGPT applications, custom GPTs; C3 (indirect) public-educator upskilling; C4 ethics and privacy risks; C5 assessing outputs with human expertise | Os: O1 integrate AI into teaching workflows; O2 assess AI outputs using expertise; O3 privacy/ethical-risk coverage
- Notes: 4.8/5 (303 reviews), 29k enrolled; educator-framed but content is general GenAI literacy; leading AI-adoption scholar.

### Introduction to Generative AI (Duke)
- Host: Duke University (Coursera) | Type: university | Program: LLMOps Specialization, course 1
- Instructor(s): Noah Gift, Executive in Residence, Duke MIDS; Alfredo Deza, Adjunct Professor
- Fee: audit-free; paid cert optional | Established: 2024 (archive.org 2024-04-13) | Duration: ~4 weeks self-paced
- Format: self-paced video, beginner, no prior knowledge | URL: https://www.coursera.org/learn/intro-gen-ai
- Cs: C1 LLM/foundation-model foundations; C2 GenAI system types (API, embedded); C3 (indirect) workforce productivity; C4 ethics segment | Os: O1 effective prompting for work; O2 iteratively improving outputs
- Notes: 4.5/5, 30k enrolled; later specialization courses turn developer-oriented; governance thin — pair with a governance course.

### What Can AI Do for Marketing?
- Host: Emory University, Goizueta Business School (Coursera) | Type: university | Program: AI for Marketing Specialization, course 1
- Instructor(s): David Schweidel, Professor of Marketing
- Fee: audit-free; paid cert optional | Established: 2024 (archive.org 2024-09-12) | Duration: ~2 weeks self-paced
- Format: self-paced video, beginner | URL: https://www.coursera.org/learn/whatcanaidoformarketing
- Cs: C2 AI across the customer journey; C3 (indirect) outreach/communications functions; C4 "Bias in the Machine" module | Os: O1 apply AI in communication workflows; O2 assessing AI-driven impact; O3 bias/fairness module
- Notes: Domain-scoped (marketing) — include as role-specific intro, not general AI 101.

### Introduction to Artificial Intelligence (UIUC)
- Host: University of Illinois Urbana-Champaign, Gies College of Business (Coursera) | Type: university | Program: Gies open courses
- Instructor(s): Robert J. Brunner, Professor and Associate Dean for Innovation
- Fee: audit-free; paid cert optional | Established: 2025 (archive.org 2025-03-15) | Duration: ~21 hrs, 4 modules
- Format: self-paced MOOC, beginner, no prerequisites | URL: https://www.coursera.org/learn/intro-to-artificial-intelligence
- Cs: C1 history, ML/DL fundamentals; C2 neural nets, generative models, data; C3 (indirect) policy/societal-impact module; C4 bias, data ethics; C5 responsible AI, regulation, compliance; C6 AGI, AI safety, workforce future | Os: O1 business applications across workflows; O2 model evaluation, output assessment; O3 regulation, compliance, data ethics
- Notes: 17.5k enrolled, 4.7/5; strongest all-six-C coverage found in the gap sweep.

### Introduction to Generative AI (BUSAI X001)
- Host: The Ohio State University, Professional and Continuing Education | Type: university | Program: AI for Business Growth certificate track (OSU AI Fluency Initiative)
- Instructor(s): Dr. Jeff Dotson, Associate Professor of Marketing, Fisher College
- Fee: $49 | Established: 2025 (OSU News 2025-09-16) | Duration: ~10 hrs self-paced
- Format: online self-paced; OSU certificate of completion | URL: https://reg-continuinged.osu.edu/search/publicCourseSearchDetails.do?method=load&courseId=1260877
- Cs: C1 generative AI in plain language; C2 tool capabilities and limitations | Os: O1 apply AI to writing/planning/productivity; O2 evaluating AI-generated content
- Notes: Non-technical adult learners; open registration (QA-confirmed Jun–Sep 2026 session); pair with a governance course for C4/C5.

### Demystifying AI, Understanding Risks, and Shaping the Future
- Host: Purdue University (Purdue Online) | Type: university | Program: Purdue AI Microcredentials, Foundational Non-Technical bundle
- Instructor(s): Dr. David Peterson, Assistant Professor of Sociology
- Fee: $99 | Established: 2024 (program page archive 2024-07-11) | Duration: ~15 hrs, 4 modules
- Format: online asynchronous; certificate + Credly badge; ABET-recognized | URL: https://www.purdue.edu/online/artificial-intelligence-micro-credentials/
- Cs: C1 AI history, deep-learning basics; C2 system design, applications; C4 speculative + immediate societal risks; C6 shaping AI's trajectory | Os: O1 practical applications any background; O2 understanding risks and limitations
- Notes: First ABET-recognized AI microcredential series; designed for professionals/policymakers.

### AI Policy and Governance
- Host: Purdue University (Purdue Online) | Type: university | Program: Purdue AI Microcredentials, Foundational Non-Technical bundle
- Instructor(s): Dr. Daniel Schiff, Assistant Professor of Technology Policy; Co-Director, GRAIL lab
- Fee: $99 | Established: 2024 (program page archive 2024-07-11) | Duration: ~15 hrs, 5 modules
- Format: online asynchronous; certificate + Credly badge | URL: https://www.purdue.edu/online/artificial-intelligence-micro-credentials/
- Cs: C3 public policy for AI in government; C4 ethics, civil-society concerns; C5 governance strategies, self-regulation; C6 international regulatory trends | Os: O2 critically assessing societal risks; O3 governance, regulation, responsible-AI oversight
- Notes: Strongest O3-specific university offering found; instructor is a former JPMorgan Responsible AI lead — highly relevant for government audiences.

### AI Ethics, Inclusion & Society
- Host: University of Glasgow (FutureLearn; listed as "Generative AI Ethics and Society") | Type: university | Program: Glasgow free online courses
- Instructor(s): Dr Ciorsdaidh Watts, Senior Lecturer + University AI Champion; Dr Lydia Bach
- Fee: free tier (time-limited access; upgrade $54) | Established: 2025 (archive.org 2025-06-23) | Duration: 3 weeks, 5 hrs/wk
- Format: self-paced online | URL: https://www.futurelearn.com/courses/ai-ethics-inclusion-and-society
- Cs: C1 what AI is, how it works; C2 data and algorithms; C3 AI in learning/healthcare/environment; C4 bias, digital divide, discrimination; C5 governance, accountability frameworks; C6 existential risks, future of AI | Os: O2 critically examine AI deployment; O3 governance, bias, inclusion
- Notes: Found independently by two gap tracks; ethics-first framing, weak O1; co-created with students (mixed quality signal).

### Generative AI in Education
- Host: University of Glasgow (Coursera) | Type: university | Program: none
- Instructor(s): Dr Gabriella Rodolico, Senior Lecturer in Science Education
- Fee: free enrollment/time-limited content; cert paid | Established: 2024 (archive.org 2024-04-21) | Duration: ~5 hrs, 4 modules
- Format: self-paced, beginner | URL: https://www.coursera.org/learn/generative-ai-in-education
- Cs: C1 generative AI fundamentals; C2 prompt engineering, tools; C3 (indirect) public education workforce; C4 misinformation, plagiarism, copyright; C5 responsible-use guidance | Os: O1 tools in workflow; O2 critical considerations; O3 partial (copyright/ethics)
- Notes: Education-framed but modules 1–3 are a general GenAI intro; 9.5k learners, 4.6/5.

### AI Regulation: Navigating the EU AI Act
- Host: TU Delft (DelftX, edX) | Type: university | Program: Navigating AI and Online Platform Regulations
- Instructor(s): Marie-Therese Sekwenz, TU Delft (TPM); Prof Hannah Ruschemeier
- Fee: audit-free; verified cert $140 | Established: 2026 (edX datePublished 2026-01-06; first run Jun 2026) | Duration: 4 weeks, 2–4 hrs/wk
- Format: self-paced MOOC, no prerequisites | URL: https://learningforlife.tudelft.nl/ai-regulation-navigating-the-eu-ai-act/
- Cs: C2 AI value chain, GPAI vs AI systems; C3 (indirect) public-administration audience; C4 prohibited practices, high-risk harms; C5 trustworthy AI, human oversight, data governance; C6 GPAI systemic risk debates | Os: O1 map requirements to AI workflows; O2 human oversight, transparency duties; O3 risk-based regulation, enforcement
- Notes: EU-specific law — for US government staff this is comparative-governance material, not general AI 101; QA-confirmed free audit track.

### Artificial Intelligence for Everyone (Universiti Malaya)
- Host: Universiti Malaya (FutureLearn) | Type: university | Program: none
- Instructor(s): Norisma Idris, Assoc. Prof., Dept. of AI (+4 UM AI faculty)
- Fee: free tier (8-week access) or $109 one-off | Established: 2025 (archive.org 2025-04-06 — resolves the earlier Track-1 date question) | Duration: 8 weeks, 2 hrs/wk
- Format: self-paced online | URL: https://www.futurelearn.com/courses/artificial-intelligence-for-everyone
- Cs: C1 ML and deep-learning foundations; C2 NLP, computer vision, GenAI/LLMs; C3 (indirect) cross-domain applications; C4 ethical/governance considerations; C5 governance of applications | Os: O1 apply AI techniques to scenarios; O2 analyse applications critically; O3 ethics/governance coverage
- Notes: Was a Track-1 near-miss for unverifiable date — now date-verified; taught by the AI department of Malaysia's top university.

### Agentic AI and AI Agents: A Primer for Leaders
- Host: Vanderbilt University (Coursera) | Type: university | Program: Agentic AI and AI Agents for Leaders Specialization, course 1
- Instructor(s): Dr. Jules White, Professor of Computer Science & Associate Dean
- Fee: audit-free; cert paid | Established: Jan 2025 (archive.org 2025-01-15) | Duration: ~6 hrs, 2 modules
- Format: self-paced video + custom-GPT hands-on | URL: https://www.coursera.org/learn/agentic-ai
- Cs: C1 how GenAI-plus-tools works; C2 agent capabilities, custom GPTs; C6 future-of-work agent trends | Os: O1 build basic work-automation agents; O2 differentiate innovation from hype
- Notes: 118k+ enrolled, 4.7/5; scope is agentic AI specifically — strong complement to the concept map's AI Agent anchor.

### Current Issues in Ethics and AI
- Host: University of Colorado Boulder (Coursera) | Type: university | Program: none
- Instructor(s): Bobby Schnabel, Professor of Computer Science; former ACM CEO
- Fee: QA-corrected — Coursera Plus/paid enrollment (~$59/mo, 7-day free trial); free audit not confirmed | Established: 2025 (archive.org 2025-11-19) | Duration: 3 weeks
- Format: self-paced, 5 modules | URL: https://www.coursera.org/learn/current-issues-in-ethics-and-ai
- Cs: C3 critical view of policing/justice/gov AI uses; C4 bias, privacy, misinformation; C5 ethical frameworks, AI regulation; C6 AGI, autonomous warfare, future of work | Os: O2 evaluate AI harms critically; O3 regulation, privacy, civil-rights (bias, policing)
- Notes: Distinguished instructor; ethics/risk-centric complement with natsec topics (autonomous warfare).

### AI for Work and Life (UNF)
- Host: University of North Florida (Open edX via GetCertificate.Online) | Type: university | Program: UNF AI for Work and Life certificate
- Instructor(s): Multi-instructor: Dr. Josh Gellers, Dr. Suzanne Ehrlich (UNF faculty), Dr. Moez Limayem (UNF President), Dr. Reid Blackman (ethics)
- Fee: QA-resolved — module content free to audit; verifiable certificate $249 (over cap; audit route qualifies) | Established: Sep 2025 (UNF newsroom; course id UNF+AI101+2025) | Duration: 8 modules, ~10 hrs + expert-led virtual sessions
- Format: online on-demand + live expert sessions; 1 CEU | URL: https://learn.getcertificate.online/courses/course-v1:UNF+AI101+2025_S/about
- Cs: C1 AI foundations and futures; C2 everyday AI tools; C3 (indirect) AI across law/finance/medicine; C4 AI in Society module; C5 responsible-use (Blackman segment); C6 futures trends | Os: O1 prompting + capstone applying AI to own role; O2 art and science of prompting; O3 AI in Society, ethics
- Notes: Explicit "AI 101" branding; was a Track-2 near-miss on fee — QA confirmed the free-audit path persists (only the certificate is $249), restoring eligibility.

### Gap-sweep near-misses (new, deduplicated)

Fee: Future Proof with AI (Harvard Business School, $199); HKS GenAI exec edition ($1,995); Cambridge PACE Ethical AI (£415+); TU Delft AI in Public Service (€575 — ideal topical fit otherwise); Penn State AI Literacy for Professionals ($525); eCornell AI certificates ($3,750); UChicago GenAI for Business ($2,800); Kellogg exec AI programs; Ohio State sibling courses ($279); Imperial AI Fundamentals for policymakers (£750, in-person, UK-only — multi-fail).
Launch date: MITx Understanding the World Through Data (Oct 2023); ASU AI Foundations courses (2023); Edinburgh Data Ethics/AI (2023); TU Delft AI in Practice (2020); Wharton AI For Business (2021); RMIT GenAI for Business (2023); UMD Cybersecurity in the AI Era (2022); Vanderbilt GenAI Strategic Leader (includes 2023 courses).
Prerequisites/level: Berkeley LLM Agents MOOC (ML/DL background); Penn AI & ML Essentials with Python; MIT 6.S184 (math/PyTorch, student-led); CS50 AI with Python (unchanged — CS50x/Python required).
Provider/access: CMU OLI "AI for Learning" (staff-authored not professor-led; educator-specific — demoted by consolidation review); Glasgow GenAI in the Classroom for Educators (educators-only, minimal AI-101 content — demoted by consolidation review); MIT Horizon (institutional licensing only); Stanford HAI/GSA federal series (2023 launch + gov-only); UGA AI Literacy for All (UGA-affiliates only); Purdue alumni free tier (alumni-only); CSU systemwide Canvas courses (CSU-only); UT Austin Essentials of AI (2023 + UT-only); HKS free webinars (not structured courses); AI Literacy for Early Career Learners (EdTech provider); KCL AI in Education (pre-2024 rework).

### Unverified watch-list (could not confirm via live fetch — recheck later)

CMU Heinz/OFAI "AI for Public Service" open-source curriculum (announced Oct 2024, no enrollable page yet); CMU Heinz + NACD AI Oversight for Directors (Dec 2025 announcement); UC Berkeley "AI-Forward Professional" (page 404); Harvard Data Science Initiative GenAI intensive (fee unstated); Columbia Plus AI courses (site 403); Cornell summer AI course (page not located); Emory ailearning portal; Ohio State Agentic AI for Small Business ($79, date/instructor unverified); Penn State Berks Responsible AI (noncredit, unpriced); Purdue Interdisciplinary AI Fundamentals ($99 — verify instructor faculty roles); Glasgow student/healthcare GenAI courses; LSE AI for Social Sciences (fee/eligibility unclear); Edinburgh AI for Leaders (page 404); Turing Institute self-paced courses (dates unverified); UW/Minnesota/Texas A&M/Princeton/Yale/NYU/Rice: no eligible public offering found (negative sweeps documented in gap plan cache).
