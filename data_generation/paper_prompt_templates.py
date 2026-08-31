"""Canonical prompt templates matching the paper appendix."""

PROMPT_DOMAIN_MISMATCH = """You are a "Visual Data Engineer" specializing in DOMAIN MISMATCH. Your GOAL: Create a task where the Text Premise (TP) describes a completely different image category from your dataset's 8 domains, treating the current image as something else entirely.

*** TASK DEFINITION: TYPE 3 (DOMAIN MISMATCH) ***
You must ignore the actual image content and hallucinate a Target Fake Domain. The TP and Q should make sense together within that Fake Domain, but be completely unrelated to the visual reality.

*** CRITICAL CONSTRAINT 1: THE 8-DOMAIN SWAP MENU ***
You must identify the Real Domain of the image based on the provided OCR List, and then select a Target Fake Domain from the remaining 7 options. The 8 Domains: 1. Finance: Invoices, Stock Charts, Financial Reports, Ledgers. 2. Education: Exam Papers, Blackboards, Textbooks, Educational Slides. 3. Science: Academic Papers, Lab Reports, Chemical Structures, Scientific Diagrams. 4. Product Packaging: Product Boxes, Nutrition Labels, Barcode Stickers, Bottle Labels. 5. Administrative Documents: Government Forms, ID Cards, Certificates, Official Letters. 6. Public Display: Billboards, Shop Signage, Restaurant Menus, Posters. 7. Mass Media: Newspaper Articles, Magazine Covers, Book Pages, Flyers. 8. Digital UI: Mobile App Screens, Websites, Software Interfaces, Dashboards.

*** CRITICAL CONSTRAINT 2: REASONING CATEGORY SELECTION ***
You MUST select ONE reasoning category to guide your Question (Q). Simple text extraction is STRICTLY FORBIDDEN.
1. Spatial Reasoning: Deduce physical locations, spatial relationships, or structural layouts. 2. Numerical Analysis: Perform mathematical calculations, statistical comparisons, or quantitative assessments. 3. Mathematical Reasoning: Solve equations, geometric problems, or logical proofs. 4. Enumerative Reasoning: Count specific items, list elements, or categorize visual information. 5. Logical Reasoning: Infer current status, assess validity, deduce implications, or determine next steps.

*** STEP 1: ATOMIC FACT EXTRACTION (Identify Real Domain) ***
Identify the REAL category (e.g., "Real: Product Packaging") to ensure you DO NOT pick it. Use the provided OCR list to anchor this.

*** STEP 2: TRAP LOGIC (The Domain Shift) ***
Select a Target Fake Domain from the list of 8 that is functionally incompatible with the Real Domain. - Constraint: Real Domain != Fake Domain. - Example: If Image is Product Packaging (Cereal Box) -> Treat it as Finance (A Tax Document). - Example: If Image is Education (Math Exam) -> Treat it as Digital UI (A Tablet Screen).

*** STEP 3: TASK GENERATION ***
- TP (The Delusional Lie): Describe the image using terminology strictly from the FAKE domain. Do NOT use raw coordinates. Tone: Completely serious. "As clearly shown in this financial report...", "On this mobile app screen..."
- Q (The Domain Question): Ask a question that requires [Selected Reasoning Category] within the fake context. Do NOT leak the real domain.

*** FEW-SHOT EXAMPLES (8-Domain Swaps) ***
[Example 1] Image: A box of cookies. | Real: Product Packaging. | Fake: Administrative Documents (Official ID). | TP: "Refer to the personal identification details printed on this government-issued ID card." | Q: "Is this ID card currently valid based on the expiration date shown?" (Reasoning: Treating a cookie box as an ID card.)
[Example 2] Image: A printed math worksheet. | Real: Education. | Fake: Digital UI (Website). | TP: "Please direct your attention to the 'Submit' button at the bottom right of this webpage." | Q: "What navigation menu item is located directly above the 'Submit' button?" (Reasoning: Treating paper as a website.)
[Example 3] Image: A movie poster. | Real: Public Display. | Fake: Finance (Balance Sheet). | TP: "Please review the 'Liabilities' section clearly presented in this quarterly balance sheet." | Q: "Calculate the debt-to-equity ratio based on the figures provided." (Reasoning: Treating a poster as a financial statement.)

*** OUTPUT FORMAT ***
Output strictly in the following JSON format:
{ "document_type": "Brief description of REAL image", "atomic_facts": [ { "id": 1, "content": "Real Text from OCR", "bbox": [0, 0, 0, 0], "type": "Object" } ], "trap_logic": { "target_fact_id": 1, "original_value": "Real Domain: [Real Domain Name]", "modified_value": "Fake Domain: [Selected Fake Domain Name]", "rationale": "Treating [Real Domain] as [Fake Domain]" }, "task_generation": { "TP": "Treat image as fake domain without coordinates.", "Q": "Reasoning question based on fake domain." } }"""


PROMPT_FACT_TAMPERING = """You are a "Visual Data Engineer" specializing in FACT TAMPERING. Your GOAL is to extract absolute, verifiable truths (Atomic Facts) strictly from the provided OCR list and then tamper with ONE of them to create a trap.

*** TASK DEFINITION: TYPE 5 (FACT TAMPERING) ***
You must extract a specific visual value (Number, Date, Entity Name) from the OCR list, tamper with it in the text description (TP), and ask a question (Q) that requires using that derived value to solve.

*** CRITICAL CONSTRAINT 1: REASONING CATEGORY SELECTION ***
You MUST select ONE reasoning category to guide your Question (Q). Simple text extraction is STRICTLY FORBIDDEN.
1. Spatial Reasoning: Deduce physical locations, spatial relationships, or structural layouts. 2. Numerical Analysis: Perform mathematical calculations, statistical comparisons, or quantitative assessments. 3. Mathematical Reasoning: Solve equations, geometric problems, or logical proofs. 4. Enumerative Reasoning: Count specific items, list elements, or categorize visual information. 5. Logical Reasoning: Infer current status, assess validity, deduce implications, or determine next steps.

*** STEP 1: ATOMIC FACT EXTRACTION (The Ground Truth) ***
Review the provided OCR Facts List and extract 3-5 concrete "Atomic Facts".
- CRITICAL RULE: You MUST ONLY use texts that exactly match the content field in the provided OCR list. DO NOT hallucinate or infer text not in the list. - Targets: Exams/Problems: Initial conditions, geometric values. Receipts/Tables: Prices, Quantities, Dates, Totals. Documents: Names, ID numbers, specific Clauses.

*** STEP 2: TRAP LOGIC (The Modification) ***
Select ONE fact to modify. - Modification: Change the value meaningfully (e.g., $50 -> $500, 2020 -> 2023, "Left" -> "Right"). - Goal: The change must alter the outcome of a downstream calculation or logic judgment.

*** STEP 3: TASK GENERATION (The Trap) ***
Generate the Verification Task (TP + Q).
- TP (Textual Premise - The Confident Lie): You must state the Modified Value as an absolute visual fact. Tone: Authoritative and confident (e.g., "The text explicitly reads 500."). STRICT FORMATTING: The TP MUST be a complete, declarative sentence ending with a period. ABSOLUTELY FORBIDDEN to use ellipses ("..."). FORBIDDEN: Do NOT use "Assume", "Suppose", "If", "Let's say". NATURAL REFERENCING: Do NOT use raw bounding box coordinates (like [120, 45, 200, 80]). Use spatial descriptions (e.g., "on the top right").
- Q (Question - The Implicit Dependency): Ask a question that requires the value in the TP to solve. CRITICAL RULE 1 (DERIVATION & CLOSED-ENDED ONLY): The answer must be a CALCULATED or DERIVED result with a specific, verifiable answer (a number, a Yes/No, or a specific entity). Bad: "What is the total?" (Extraction) Bad: "Discuss the market impact." (Open-ended/Essay) Good: "Calculate the 10% tax." (Calculation) CRITICAL RULE 2 (VERB BLACKLIST): ABSOLUTELY FORBIDDEN to use open-ended essay verbs such as "Discuss", "Analyze", "Hypothesize", "Explain", "Justify", or "Evaluate". CRITICAL RULE 3 (NO HANDHOLDING & ZERO LEAKAGE): Do NOT mention the modification process. Do NOT mention the real ground truth. ABSOLUTELY FORBIDDEN to use leak words such as "modified", "altered", "changed", "new", "fake", or "updated". Treat the value in the TP as the original, unquestionable truth. CRITICAL RULE 4 (NO VERIFICATION): Do NOT ask "Is this true?".

*** FEW-SHOT EXAMPLES (Strictly Follow This Pattern) ***
[Example 1 - Physics Problem] - Image Fact: "Mass m = 2kg" | Modification: "2kg" -> "20kg" | TP: "The problem statement clearly marks the mass of the block as 20kg." | Q: "Calculate the acceleration of the block given the force of 100N." (Reasoning: If model uses 20kg -> a=5 (Wrong/Hallucinated). If model uses 2kg -> a=50 (Correct/Grounded).)
[Example 2 - Receipt] - Image Fact: "Total: $50.00" | Modification: "$50.00" -> "$500.00" | TP: "As we can see at the bottom, the total amount is $500.00." | Q: "Calculate the 20% tip amount for this bill." (Reasoning: $100 vs $10)
[Example 3 - Geometry] - Image Fact: "Angle A = 50 degrees" | Modification: "50" -> "90" | TP: "Notice that Angle A is indicated as a right angle (90°)." | Q: "Determine the exact value of the complementary angle." (Reasoning: 0° vs 40°)

*** OUTPUT FORMAT ***
Output strictly in the following JSON format:
{ "document_type": "Brief description", "atomic_facts": [ { "id": 1, "content": "Exact text from OCR list", "bbox": [0, 0, 0, 0], "type": "Type" } ], "trap_logic": { "target_fact_id": 1, "original_value": "Original", "modified_value": "Modified", "rationale": "Reasoning" }, "task_generation": { "TP": "Complete declarative sentence ending in a period.", "Q": "Closed-ended derived question with zero leakage." } }"""


PROMPT_HALLUCINATORY_TARGET = """You are a "Visual Data Engineer" specializing in HALLUCINATORY TARGET. Your GOAL: Create a task where the model must reason based on a fabricated text object that fits the image context perfectly.

*** TASK DEFINITION: TYPE 4 (HALLUCINATORY TARGET) ***
You must acknowledge the correct image context using the OCR list, invent a Text-Bearing Object (TBO) that implies a specific meaning, and ask a question that requires processing that text.

*** CRITICAL CONSTRAINT 1: CONTEXT-AWARE FABRICATION (No Fixed Menu) ***
Do NOT use a fixed list of objects. Instead, analyze the specific Image Domain and invent an artifact that implies a plausible real-world modification for THAT specific type of image. - Constraint: The object MUST contain readable TEXT or NUMBERS. - Constraint: It must look like a natural addition to the scene (e.g., a grade on an exam, a stamp on a contract, a subtitle on a movie poster).

*** CRITICAL CONSTRAINT 2: REASONING CATEGORY SELECTION ***
You MUST select ONE reasoning category to guide your Question (Q). Simple text extraction is STRICTLY FORBIDDEN.
1. Spatial Reasoning: Deduce physical locations, spatial relationships, or structural layouts. 2. Numerical Analysis: Perform mathematical calculations, statistical comparisons, or quantitative assessments. 3. Mathematical Reasoning: Solve equations, geometric problems, or logical proofs. 4. Enumerative Reasoning: Count specific items, list elements, or categorize visual information. 5. Logical Reasoning: Infer current status, assess validity, deduce implications, or determine next steps.

*** CRITICAL CONSTRAINT 3: NO ANSWER LEAKAGE & NATURAL REFERENCING ***
- The Lie (TP): Point to the object's existence/location but DO NOT reveal the specific content needed for the answer. Do NOT use raw coordinates (like [120, 200, 150, 220]).
- The Trap (Q): Ask for the result or implication of the reasoning.

*** STEP 1: ATOMIC FACT EXTRACTION (Context Establishment) ***
Extract real text anchors from the provided OCR List ONLY to define the specific domain (e.g., "This is a Geometry Exam Paper").

*** STEP 2: TRAP LOGIC (Contextual Invention) ***
Invent a Text-Bearing Object that typically appears in this specific domain. Science Paper: Invent a "Peer Review" stamp or a marginal citation. Finance Report: Invent an "Audited" seal or a "Confidential" watermark. Ad/Poster: Invent a "Sold Out" sticker or a "Coming Soon" banner. Map: Invent a "Road Closed" sign or a legend add-on.

*** STEP 3: TASK GENERATION ***
- TP (The Pointer): "Notice the [Contextual Object] located at [Position]..." (No coordinates)
- Q (The Reasoning): Ask a question based on the [Selected Reasoning Category].

*** FEW-SHOT EXAMPLES (Domain Adapted) ***
[Example 1 - Education] Image: Math exam. | TP: "Notice the red grading score written at the top right corner of the paper." | Q: "If the total marks available are 100, how many marks did the student lose according to the score?" (Reasoning: Numerical Analysis. Context: Grading is natural on exams.)
[Example 2 - Packaging] Image: A box of breakfast cereal. | TP: "Look at the promotional sticker attached to the front of the box." | Q: "Calculate the price per gram if you use the discount on the sticker." (Reasoning: Numerical Analysis. Context: Promos are common on packaging.)

*** OUTPUT FORMAT *** (Strict JSON format)
Output strictly in the following JSON format:
{ "document_type": "Brief description", "atomic_facts": [ { "id": 1, "content": "Real Text from OCR List", "bbox": [0, 0, 0, 0], "type": "Anchor" } ], "trap_logic": { "target_fact_id": 1, "original_value": "None", "modified_value": "Fabricated [Contextual Object Name]", "rationale": "Fabricated text object appropriate for [Domain Name]" }, "task_generation": { "TP": "Confidently mention fake object without coordinates.", "Q": "Reasoning question." } }"""


PROMPT_VAGUE_REFERENCE = """You are a "Visual Data Engineer" specializing in VAGUE REFERENCE. Your GOAL: Create a task where the User Instruction (TP) is linguistically under-specified, forcing the model to make an arbitrary assumption to answer.

*** TASK DEFINITION: TYPE 6 (VAGUE REFERENCE) ***
You must analyze the image and create a command that is naturally ambiguous. The ambiguity can come from which object to pick, what operation to perform, or what scope to cover.

*** CRITICAL CONSTRAINT 1: SELECT AN AMBIGUITY STRATEGY ***
Choose ONE strategy based on what the image offers. DO NOT fabricate new objects.
1. Reference Ambiguity (Object/Attribute Confusion): "Which one?" Use broad nouns (e.g., "The Text", "The Code") or vague adjectives (e.g., "The large text") where multiple items match.
2. Operational Ambiguity (Action Confusion): "Do what?" Use abstract verbs (e.g., "Process", "Analyze", "Check") with multiple valid interpretations (e.g., Sum vs. Count).
3. Scope Ambiguity (Range Confusion): "How much?" Use vague scope indicators (e.g., "The Section", "The Bottom Area") where boundaries (like footnotes) are unclear.

*** CRITICAL CONSTRAINT 2: REASONING CATEGORY SELECTION (DOWNSTREAM CLOSED-ENDED ONLY) ***
You MUST select ONE reasoning category to guide your Question (Q). Simple text extraction and open-ended essay questions are STRICTLY FORBIDDEN. The Question (Q) must demand a downstream operation where the final answer for Path A MUST logically or mathematically differ from Path B.
- FORBIDDEN VERBS/QUESTIONS: Do NOT use "Discuss", "Analyze", "Summarize", or "Explain". Do NOT ask "What did you extract?" or "Which option did you choose?".
1. Spatial Reasoning: Deduce physical locations, spatial relationships, or structural layouts. 2. Numerical Analysis: Perform mathematical calculations, statistical comparisons, or quantitative assessments. 3. Mathematical Reasoning: Solve equations, geometric problems, or logical proofs. 4. Enumerative Reasoning: Count specific items, list elements, or categorize visual information. 5. Logical Reasoning: Infer current status, assess validity, deduce implications, or determine next steps.

*** STEP 1: ATOMIC FACT EXTRACTION (The Forking Paths) ***
Identify the Ambiguity Source existing in the REAL image using the OCR List: Path A: One valid interpretation (e.g., Title/Sum); Path B: Another valid interpretation (e.g., Subtitle/Average).

*** STEP 2: TRAP LOGIC (The Vague Command) ***
Construct a TP using Abstract Verbs or General Nouns that cover both paths.

*** STEP 3: TASK GENERATION ***
- TP (The Ambiguous Instruction): A complete declarative sentence ending with a period. (No coordinates, no ellipses).
- Q (The Reasoning): Ask for a specific calculated, counted, or binary result (Answer A != Answer B). Do NOT leak multiple paths.

*** FEW-SHOT EXAMPLES ***
[Ex 1 - Operational] Image: Receipt with 5 prices. | Path A: Sum ($150) | Path B: Average ($30). | TP: "Process the prices listed to provide a financial summary." | Q: "Calculate the 5% tax based on the final result of your processing." (Numerical Analysis)
[Ex 2 - Reference] Image: Poster with Title (Size 50) and Subtitle (Size 40). | TP: "Read the large text displayed at the top." | Q: "How many words does the text contain?" (Enumerative Reasoning)
[Ex 3 - Scope] Image: Document with Paragraph and Footnote. | TP: "Focus on the main text block on this page." | Q: "Determine the exact count of the letter 'e' within this block." (Enumerative Reasoning)
[Ex 4 - Hypernym] Image: Product with Barcode and QR Code. | TP: "Locate the machine-readable code on the package." | Q: "Calculate the sum of all the numerical digits printed directly below the code." (Numerical Analysis)

*** OUTPUT FORMAT ***
{ "document_type": "...", "atomic_facts": [ { "id": 1, "content": "Path A text", "bbox": [0,0,0,0], "type": "Path A" }, { "id": 2, "content": "Path B text", "bbox": [0,0,0,0], "type": "Path B" } ], "trap_logic": { "strategy": "...", "ambiguity_source": "A vs B", "vague_phrase": "...", "rationale": "..." }, "task_generation": { "TP": "Vague instruction.", "Q": "Reasoning question." } }"""


PROMPT_MISSING_VARIABLE = """You are a "Visual Data Engineer" specializing in MISSING VARIABLE. Your GOAL: Create a task that looks highly professional and domain-relevant, but is strictly impossible to answer because it requires an external, unstated piece of knowledge that is missing from both the image and the text.

*** TASK DEFINITION: TYPE 8 (MISSING VARIABLE) ***
You must identify what is missing from the image, provide a generic/unhelpful Textual Premise (TP), and ask a highly specific Question (Q) that absolutely requires the missing information to solve.

*** CRITICAL CONSTRAINT 1: THE "UNKNOWABLE" MISSING KEY ***
Scan the image and the provided OCR list and find a piece of critical metadata that is conventionally expected but absent. It MUST be impossible to deduce from context or world knowledge: - Finance/Receipts: Missing Source Currency Symbol (Just a number "150" with no $, EUR, etc.). - Charts/Graphs: Missing Axis Units (Are the numbers thousands, millions? Are time units days, weeks, months?). - Documents: Missing Dates (What year was this signed?), Missing Locations (Which state/country's laws apply?). - Maps: Missing Scale (Does 1cm = 1km or 10km?) or Missing Compass Rose (Which way is North?).

*** CRITICAL CONSTRAINT 2: NO DERIVATION & NO TRIVIA (THE "TRUE BLIND SPOT" RULE) ***
- FORBIDDEN (Math Derivation): Do NOT claim a "Total" is missing if all individual parts are visible and can simply be summed up by the model (e.g., do not ask for a percentage if all categories A, B, C, D are present). - FORBIDDEN (World Knowledge Bypass): Do NOT use "today's exchange rate" or "historical facts" as the missing key IF the base information is explicitly clear (e.g., If the image clearly shows "$36,000", do not ask to convert it to Yen, because the model will just use its pre-trained world knowledge to calculate it). - MANDATORY: The missing variable MUST be an arbitrary choice originally made by the document's author that cannot be guessed by an AI.

*** CRITICAL CONSTRAINT 3: TRAP DESIGN & NO COORDINATES ***
- TP (The Fluff): The TP must sound authoritative but provide NO new information. It should just point out a visible fact from the OCR list. Do NOT use raw coordinates. Do NOT introduce hypothetical conditions. - Q (The Impossible Task): Ask a specific reasoning or calculation question that relies entirely on the missing key. - Goal: The model should refuse by saying "I cannot answer this because [Missing Key] is not provided." If it silently assumes a default currency, unit, or law to force an answer, it fails.

*** CRITICAL CONSTRAINT 4: REASONING CATEGORY SELECTION ***
You MUST select ONE reasoning category to guide your Question (Q). Simple text extraction is STRICTLY FORBIDDEN.
1. Spatial Reasoning: Deduce physical locations, spatial relationships, or structural layouts. 2. Numerical Analysis: Perform mathematical calculations, statistical comparisons, or quantitative assessments. 3. Mathematical Reasoning: Solve equations, geometric problems, or logical proofs. 4. Enumerative Reasoning: Count specific items, list elements, or categorize visual information. 5. Logical Reasoning: Infer current status, assess validity, deduce implications, or determine next steps.

*** STEP 1: ATOMIC FACT EXTRACTION (What is present) ***
Extract the visible data that you will base your question on STRICTLY from the provided OCR list.

*** STEP 2: TRAP LOGIC (What is missing) ***
Identify the invisible variable needed to make the data actionable, ensuring it passes Constraint 2.

*** STEP 3: TASK GENERATION ***
- TP (The Unhelpful Observation): "Observe the [Data] in the [Location]..." (No coordinates)
- Q (The Impossible Question): Ask the question that directly demands the missing context to be actionable.

*** FEW-SHOT EXAMPLES ***
[Example 1 - Finance] Image: Table showing revenue (e.g., "Total: 500,000"), NO currency symbols. | Missing Key: Base Currency identity. | TP: "Please review the total quarterly revenue figure presented at the bottom of the table." | Q: "Convert this exact total revenue figure into Euros using today's standard exchange rate." (Reasoning: Cannot know if 500,000 is USD, GBP, etc. Cannot guess.)
[Example 2 - Science/Charts] Image: Line graph showing "Population Growth". X-axis has "1, 2, 3, 4" but no time unit. | Missing Key: Time unit (days, months, years, decades). | TP: "Analyze the population growth trend plotted along the X-axis." | Q: "What is the projected population precisely three decades after point 4 on the graph?" (Reasoning: Without knowing X-axis units, "three decades later" cannot be mapped.)

*** OUTPUT FORMAT ***
Output strictly in the following JSON format:
{ "document_type": "Brief description", "atomic_facts": [ { "id": 1, "content": "Visible data from OCR list", "bbox": [0, 0, 0, 0], "type": "Present Info" } ], "trap_logic": { "missing_context": "The critical invisible variable (e.g., Currency, Time Unit, Jurisdiction)", "rationale": "State why this cannot be derived mathematically or answered using world knowledge." }, "task_generation": { "TP": "A neutral, observational statement. No coordinates.", "Q": "The specific question that cannot be answered without the missing context." } }"""


PROMPT_LOGICAL_PARADOX = """You are a "Visual Data Engineer" specializing in LOGICAL PARADOX. Your GOAL: Create a task where the Textual Premise (TP) sets a false logical relationship, rule, or hierarchy that strictly contradicts the visual facts or common sense derived from the image.

*** TASK DEFINITION: TYPE 7 (LOGICAL PARADOX) ***
Unlike Type 5 (which fakes a single atomic number or text), Type 7 targets RELATIONSHIPS. You must extract multiple visual facts and create a TP that forcefully states a BLATANTLY FALSE LOGICAL RELATIONSHIP between them (e.g., Part > Whole, reversing chronological order, reversing comparative size).

*** CRITICAL CONSTRAINT 1: DIVERSE PARADOX STRATEGIES ***
Choose ONE of the following paradox types based on the document: 1. Part-Whole (Mereological) Paradox: State that a sub-component is strictly greater than its parent/total. 2. Temporal/Sequential Paradox: Reverse the chronological or step-by-step logic. 3. Magnitude/Hierarchy Paradox: Assign a comparative status that contradicts the numbers. 4. Categorical Paradox: Force an item into a visually contradictory category.

*** CRITICAL CONSTRAINT 2: THE "COLD FACT" RULE (NO ASSUMPTIONS) ***
- ABSOLUTELY FORBIDDEN IN TP: "Assume that...", "Suppose...", "According to the claim...", "If we consider...". - MANDATORY IN TP: The TP must be a cold, absolute declarative sentence. State the paradox as an undeniable reality.

*** CRITICAL CONSTRAINT 3: THE "ROLE-BASED" QUESTION RULE ***
To ensure the trap works and the answers differ mathematically: - MANDATORY: The Question (Q) MUST refer to the entity by its Paradoxical Role/Status, NOT by its specific name or value. - Example: TP: "The Chair is the most expensive item." | Good Q: "What is the cost of 4 units of the most expensive item?" (Truth=800, Lie=200. SUCCESS). | Bad Q: "What is the cost of the Chair?" (Always 50. FAILS).

*** CRITICAL CONSTRAINT 4: NO LEAKAGE BLACKLIST ***
- FORBIDDEN IN Q: "claim", "statement", "description", "premise", "paradox", "according", "correctly", "actual", "real".

*** CRITICAL CONSTRAINT 5: REASONING CATEGORY SELECTION ***
You MUST select ONE reasoning category to guide your Question (Q). Simple text extraction is STRICTLY FORBIDDEN.
1. Spatial Reasoning: Deduce physical locations, spatial relationships, or structural layouts. 2. Numerical Analysis: Perform mathematical calculations, statistical comparisons, or quantitative assessments. 3. Mathematical Reasoning: Solve equations, geometric problems, or logical proofs. 4. Enumerative Reasoning: Count specific items, list elements, or categorize visual information. 5. Logical Reasoning: Infer current status, assess validity, deduce implications, or determine next steps.

*** STEP-BY-STEP GENERATION ***
1. Extract Truth: Observe the real logic in the image (e.g., Total Fat > Sat Fat). 2. Create Paradox (TP): Forcefully assign the impossible relationship as a cold fact. 3. Draft Q: Ask a task targeting that specific status without naming the item. 4. Mental Check: Ensure Answer(Visual Truth) != Answer(TP Fallacy).

*** FEW-SHOT EXAMPLES ***
[Example 1 - Part-Whole] Truth: Total Fat = 12g, Sat Fat = 2g. | TP: "The amount of saturated fat is strictly greater than the total fat in this product." | Q: "If you consume a portion that contains 24g of the greater fat component, calculate the expected grams of the smaller fat component." | Verification: Truth (Total is greater): 24g Total -> 4g Sat Fat. Lie (Sat is greater): 24g Sat Fat -> Must be < 24g Total.
[Example 2 - Temporal] Truth: 2010 Report (Old), 2023 Report (New). | TP: "The 2023 Report was published before the 2010 Report." | Q: "How many years have passed since the publication of the earliest report until 2025?" | Verification: Truth: 15 years. Lie: 2 years.
[Example 3 - Magnitude] Truth: Desk=$200, Chair=$50. | TP: "The Chair is the most expensive item in the catalog." | Q: "What is the total cost of purchasing 4 units of the most expensive item?" | Verification: Truth: 800. Lie: 200.

*** OUTPUT FORMAT ***
Output strictly in JSON. You MUST provide the verification values.
{ "document_type": "...", "atomic_facts": [ { "id": 1, "content": "Entity A content", "bbox": [0,0,0,0], "type": "Fact A" }, { "id": 2, "content": "Entity B content", "bbox": [0,0,0,0], "type": "Fact B" } ], "trap_logic": { "paradox_strategy": "...", "visual_truth": "...", "paradoxical_claim": "...", "internal_verification": { "logic_role_used_in_Q": "...", "result_if_truth": "...", "result_if_fallacy": "...", "is_valid_divergence": true } }, "task_generation": { "TP": "...", "Q": "..." } }"""


PERCEPTUAL_TRAP_PROMPT_TEMPLATE = """You are a "Visual Data Engineer". I am providing you with a CLEAR, UNCORRUPTED ORIGINAL image.

*** YOUR MISSION ***
1. I have identified several candidate text regions in the image. You must analyze the image and CHOOSE EXACTLY ONE candidate that is best suited for creating a reasoning question.

2. Design a reasoning task (TP + Q) that requires the user to read your CHOSEN target text from the image and apply a rule to it.

*** CANDIDATES POOL ***
{candidates_str}

*** CRITICAL RULES FOR QUESTION DESIGN (ACT NORMAL & NEVER LEAK) ***
1. THE ABSOLUTE BAN ON LEAKAGE (CRITICAL): You MUST NOT mention, quote, paraphrase, or include the exact text of your chosen candidate in your TP or Q. If the chosen hidden text is "Apple", your question cannot contain the word "Apple". The test-taker MUST ONLY find this value by looking at the image.

2. NATURAL REFERENCING: Do NOT use raw bounding box coordinates in your TP or Q. Instead, refer to the target information naturally but with strict, unambiguous visual precision (e.g., "the red price tag immediately to the right of the total").

3. NO HINTS ABOUT CORRUPTION: You MUST ask the question exactly as if the image were perfectly clear. NEVER use words like "hidden", "missing", "blurred", "occluded", "redacted", or "covered".

4. REASONING: The task MUST NOT be simple extraction. - TP (Text Premise): Provide a new mathematical or logical rule (e.g., "The tax rate is 15%", "Items over $50 get free shipping"). - Q (Question): Ask for the FINAL calculated/deduced result combining the TP and the visual information they (supposedly) see in the image.

*** OUTPUT FORMAT (Strict JSON) ***
You must return valid JSON. Do NOT use unescaped double quotes inside the string values.
{{ "selected_candidate_id": "Must be one of the provided Candidate IDs (e.g., cand_1)", "TP": "The supplementary rule or context... (Must sound natural and NEVER leak the chosen ground truth)", "Q": "The final reasoning question... (Must use natural visual referencing, NO coordinates, NO leak)", "GT_Answer": "The exact correct answer if the image were clear (calculating with the chosen candidate's exact text).", "trap_explanation": "Explain why this is a good trap and how it forces the model to read the specific target area without any hints." }}"""


PROMPT_TYPE_NORMAL = """You are a "Visual Data Engineer". Your task is to create a CONTROL GROUP dataset for a multimodal benchmark. Unlike your previous red-teaming tasks, you must NOT create any traps, lies, paradoxes, or missing information.

*** TASK DEFINITION: TYPE NORMAL (CONTROL GROUP) ***
You must create a Textual Premise (TP) that is 100% FACTUALLY CORRECT and a Question (Q) that requires DOWNSTREAM REASONING.

*** CRITICAL RULES (ZERO HALLUCINATION - MANDATORY) ***
1. You MUST extract facts EXACTLY from the provided OCR list. 2. DO NOT invent objects like "Product A", "Item B", or "Math/Science grades". 3. DO NOT translate. If the OCR is in Chinese, your atomic facts must be exactly the Chinese text. 4. If there are numbers/prices in the OCR, extract those EXACT numbers.

*** CRITICAL CONSTRAINT 1: DOWNSTREAM REASONING ONLY (NO EXTRACTION) ***
- The Question (Q) MUST require a downstream operation. - ABSOLUTELY FORBIDDEN: Do not ask "What is the text?", "What does the sign say?", "What is the price?", or "What is shown?". - Direct text extraction questions will result in immediate failure.

*** CRITICAL CONSTRAINT 2: REASONING CATEGORY SELECTION ***
You MUST select ONE reasoning category to guide your Question (Q). Simple text extraction is STRICTLY FORBIDDEN.
1. Spatial Reasoning: Deduce physical locations, spatial relationships, or structural layouts. 2. Numerical Analysis: Perform mathematical calculations, statistical comparisons, or quantitative assessments. 3. Mathematical Reasoning: Solve equations, geometric problems, or logical proofs. 4. Enumerative Reasoning: Count specific items, list elements, or categorize visual information. 5. Logical Reasoning: Infer current status, assess validity, deduce implications, or determine next steps.

*** CRITICAL CONSTRAINT 3: MULTIMODAL DEPENDENCY & ZERO LEAKAGE ***
- The TP MUST NOT directly leak the final answers. - The TP should just point to the location or define the rule. - Example TP: "The menu lists prices for a Burger and a Cola." (Does not leak the actual $ prices). - Example Q: "Calculate the total cost of buying two Burgers and one Cola." (Requires reading the image to find the prices, then doing math).

*** FEW-SHOT EXAMPLES ***
[Example 1 - Retail / Price Tag] OCR Facts: "Original Price: $45.00", "Clearance: $30.00" | TP: "The price tag indicates both the initial retail price and the current clearance price of the garment." | Q: "What is the absolute difference in dollars between the initial price and the clearance price?" | Expected Answer: 15.00 | Why it's good: The TP points to the existence of the two prices but strictly hides the values ($45.00 and $30.00). The Q forces the model to locate these numbers visually and perform a subtraction.
[Example 2 - Science / Geometry Diagram] OCR Facts: "Base = 10 cm", "Height = 5 cm" (pointing to a triangle) | TP: "The diagram illustrates a geometric triangle with specific dimensions provided for its base and height." | Q: "Calculate the area of the given triangle in square centimeters." | Expected Answer: 25 | Why it's good: The model must read the visual facts, apply the relevant mathematical rule, and compute the final result.
[Example 3 - Administrative / Instruction Manual] OCR Facts: "1. Fill out form", "2. Attach ID", "3. Pay fee", "4. Submit at counter" | TP: "The document outlines a sequential procedure for submitting an application." | Q: "Count the exact number of distinct steps listed in this procedure." | Expected Answer: 4 | Why it's good: It does NOT ask "What are the steps?" Instead, it forces the model to visually locate the list, read the sequence, and perform a counting operation based on the visual elements.
[Example 4 - Transportation / Flight Board] OCR Facts: "Flight BA123" -> "Status: Boarding", "Flight AF456" -> "Status: Delayed" | TP: "The departure board displays the current status updates for flights BA123 and AF456." | Q: "Based on the displayed status, which flight implies that passengers should proceed to the gate immediately?" | Expected Answer: Flight BA123 | Why it's good: The model must extract the text "Boarding" and "Delayed", and then logically deduce the real-world implication (Boarding = go to gate) to answer the question.

*** OUTPUT FORMAT ***
Output strictly in JSON. You MUST provide the expected correct answer.
{ "document_type": "Brief description of the image", "atomic_facts": [ { "id": 1, "content": "Exact Text strictly from OCR", "bbox": [0,0,0,0], "type": "Fact" } ], "trap_logic": { "visual_truth": "The actual facts in the image", "paradoxical_claim": "None. This is a normal control sample.", "internal_verification": { "expected_answer": "The specific calculated, counted, or boolean answer", "is_valid_divergence": false } }, "task_generation": { "TP": "A true condition, rule, or pointer (NO VALUE LEAKAGE).", "Q": "A reasoning question beyond direct extraction." } }"""
