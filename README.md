# Mascot-cp

The objective of this project is to label sections of audio files with specific classroom practices (e.g., praise, opportunities to respond, feedback) based on the statements made by the instructor. This task fits into an overall objective of providing real-time feedback to teachers regarding their daily interactions with their students. This is an ongoing project and collaboration with Wesley Morris and Jess Boyle, PhD students at Vanderbilt Univerity's Peabody College. 

## Project Overview 

### Classroom Observation Codes

| Code | Definition | Examples | Non-examples |
|------|------------|----------|--------------|
| **Instructional Talk (IST)** | Teacher provides information or self-talk questions about academic content, targeted skill, or activity. Includes reviewing objectives and academic instruction. | • "Today in math, we are going to work on addition"<br>• "When we are adding numbers, we use a plus sign"<br>• "The theme in this text is about friendship"<br>• Teacher reads written questions without providing response time | • "In reading today, we need to make sure we are following our rules. We should be quietly listening while I read and following along" (ST) |
| **Social Talk (ST)** | Teacher provides information about social/classroom expectations or engages in social conversations. Includes "narrating" statements about classroom environment or student behavior. | • "Last week we struggled during social studies, let's talk about expectations"<br>• "I really like your new backpack"<br>• "How was everyone's weekend?"<br>• "Oh, I gotta lotta hands up for this one" | • "Let's get started, the title says, 'See Spot Run' and I can see a picture of a dog on the cover" (IST) |
| **Academic OTR (aOTR)** | Instructional question/statement seeking academic response orally or publicly. Must be clearly expecting student response. | • "What is the capital of Tennessee?"<br>• "Show me a thumbs up if you agree"<br>• "Susan, please share your answer"<br>• "Repeat after me: The. Dog. Ran." | • "Do you want to get kicked out?" (REP)<br>• "It's time for read aloud" (ST)<br>• "Do you understand?" (NEU) |
| **Social OTR (sOTR)** | Question/statement seeking organizational or preparatory response related to learning, organization, transitions, or social expectations. | • "Can everyone go to the carpet"<br>• "Turn to page 5"<br>• "Put your notebooks away"<br>• "Does anyone have questions?" | • "At the top of page 5, what is our title?" (aOTR)<br>• "Put your listening ears on" (ST)<br>• "Hey everyone, listen up" (ST) |
| **Reprimand (REP)** | Statement intended to stop/reprimand behavior. Includes scolding, negative statements, or consequences. | • "I told you to stand up and push in your chair"<br>• "Start paying attention or your name goes on the board"<br>• "Group 1, should you be talking?"<br>• Taking away items being misused | • "Could everyone write down their answer?" (aOTR)<br>• "Try harder on your math worksheet" (IST)<br>• "Not right now" (NEU) |
| **Redirection (RED)** | Statement intended to redirect behavior by stating what students should do. | • "Johnny, please turn around"<br>• "Group 1, stay focused"<br>• "Remember to raise your hand"<br>• "What level voice should we use?" | • "Please stop" (REP)<br>• "You need to behave better" (REP)<br>• "Please get started" (if first time asking - aOTR) |
| **Behavior Specific Praise (BSP)** | Statement indicating approval with explicit mention of praised behavior. | • "Good work keeping hands to self, Yvonne!"<br>• "Billy, I like how you showed your work!"<br>• "Thank you for raising your hand!"<br>• "Your handwriting is improving!" | • "Thank you" when collecting assignment (IST/ST)<br>• "Right" (IST) |
| **General Praise (GPRS)** | General statement indicating approval without specific behavior mentioned. | • "Great!"<br>• "Good job, Mary!"<br>• "Good try"<br>• "Woo-hoo, she got it!"<br>• "I love how great everyone is doing" | • "Thank you" when collecting assignment (NEU/ST)<br>• "Right" (aAFF) |
| **Academic Affirmation (aAFF)** | Statement indicating accurate response to aOTR/sOTR. Must clearly communicate correctness. | • "That's correct!"<br>• "Yes Isabella, that's right!"<br>• "Exactly"<br>• Repeating correct answer with affirmation | • Simple repetition without affirmation (IST)<br>• Responses adding new content (IST) |
| **Academic Corrective (aCORR)** | Statement acknowledging incorrect response to aOTR. | • "Not quite"<br>• "No, that's not right"<br>• "It's not little" | • "What do you think?" after incorrect answer (aOTR)<br>• "We are not shouting out" (REP) |
| **Student Voice (SV)** | Any audio with students talking. Exact words don't need to be clear. | • Any student voice heard<br>• Both academic and non-academic student speech | • Static without voice (SIL) |
| **Neutral (NEU)** | Catch-all for statements not meeting other definitions. Includes brief interruptions and standalone filler words. | • "Okay?" (rhetorical)<br>• "Now, hmmm"<br>• "Not right now"<br>• Brief interactions with visitors | • "Okay... let's look at this problem" (IST if <2 second pause) |

### Notes
- Multiple codes can apply to a single interaction
- Context and teacher tone help distinguish between similar codes
- Timing between statements affects segmentation
- Filler words should be grouped with nearby content if within 2 seconds

## Diarization 

Diariation is the process of segmenting and labeling audio data based on speaker identity. This is a critical step in the transcription process, as it allows for the identification of individual speakers and the separation of their speech. In this project, we are trying to identify and label specific classroom practices based on the statements made by the instructor.

### Pyannote 

pyannote.audio is a Python-based open-source toolkit designed for speaker diarization. Built on the PyTorch machine learning framework, it provides cutting-edge pretrained models and pipelines that can identify and separate different speakers in audio. Users can also fine-tune these models with their own data to achieve enhanced performance.

### Whisper Frame Classification

Wesley Morris trained a model to classify frames of audio as teacher speech or non-student speech. The model was trained on the same training, vaildation, and test splits as the RoBERTa model. 

## Transcription 

Stable-ts is a Python library designed to enhance Whisper's audio transcription capabilities, with a particular focus on improving timestamp accuracy. It modifies Whisper to produce more reliable start and end times for transcribed speech, while offering local execution capabilities and support for multiple output formats including SRT, VTT, ASS, TSV, and JSON. The library processes audio through a multi-layered approach, handling both word-level and segment-level timestamps, and can work with various audio formats thanks to its FFmpeg and PyTorch dependencies.

The library includes advanced silence suppression that can be implemented in two ways: either through volume-based detection that analyzes audio intensity relative to neighboring sections, or through Silero VAD (Voice Activity Detection) for more complex audio environments. It offers timestamp refinement through an iterative process where portions of audio are muted and token probabilities are recalculated to find precise word boundaries. Additionally, it features customizable word regrouping algorithms that can split and merge segments based on punctuation, gaps, or custom rules. The library also supports gap adjustment to improve segment boundary accuracy and provides visualization tools for monitoring these adjustments. Simple to install via pip and compatible with any ASR (Automated Speech Recognition) system, not just Whisper, Stable-ts serves as a versatile tool for improving speech transcription accuracy, particularly in applications requiring precise audio synchronization such as subtitling or audio editing.

## Text Classification 

Classroom practices do not always last a fixed amount of time. Sometimes IST my last 15 seconds, while other times it may last 3 seconds. The audio is based on speaker turns, but a teacher may exhibit multiple classroom practices within a single turn. Therefore, we need to leverage multi-label classification to identify one or more classroom practices within a single turn.

### Supervised Fine Tuning 

#### RoBERTa (Robustly Optimized BERT Approach) 

RoBERTa (Robustly Optimized BERT Approach) is distinctive due to its enhanced training process: it uses a massive 160GB text dataset (more than 10 times larger than BERT's), incorporating data from diverse sources including Wikipedia, news articles, Reddit content, and story-like text from Common Crawl. RoBERTa introduces several key technical improvements to BERT's architecture. It eliminates the Next Sentence Prediction (NSP) objective, finding that this removal either matches or improves downstream task performance. The model employs dynamic masking instead of static masking, generating new mask patterns each time data passes through the model. It also uses larger batch sizes (up to 8K sequences) and longer training sequences, which improves both the model's perplexity on masked language modeling and end-task accuracy. Like BERT, RoBERTa is pretrained using Masked Language Modeling (MLM), where it randomly masks 15% of words in input sentences and learns to predict them, enabling bidirectional representation learning. These optimizations led to state-of-the-art performance on various NLP benchmarks at the time of its release, including GLUE tasks, SQuAD, and RACE, demonstrating particular strength in tasks like natural language inference, textual entailment, and question answering.

### In-context learning with LLM

#### Qwen2.5:14b

Qwen2.5 is a newly released family of open-source language models that represents a significant advancement in AI capabilities. The collection includes general language models ranging from 0.5B to 72B parameters, trained on an impressive 18 trillion tokens. These models support over 29 languages and can handle up to 128K input tokens while generating 8K tokens. Alongside the main language models, Qwen2.5 includes specialized variants: Qwen2.5-Coder, which comes in 1.5B, 7B, and 32B sizes and is specifically trained on 5.5 trillion tokens of code-related data, and Qwen2.5-Math, available in 1.5B, 7B, and 72B sizes, which focuses on mathematical reasoning in both Chinese and English.

The models show substantial improvements over their predecessors, achieving scores of 85+ on the MMLU knowledge benchmark and HumanEval coding tests, and 80+ on the MATH benchmark. They also demonstrate enhanced capabilities in following instructions, handling structured data, generating reliable JSON output, and performing role-play scenarios. Most models in the family are released under the Apache 2.0 license, with the exception of the 3B and 72B variants, and can be easily implemented using Hugging Face Transformers or deployed through platforms like vLLM and Ollama.

#### Running the Model Locally 

Ollama (Omni-Layer Learning Language Acquisition Model) is a groundbreaking platform that democratizes access to large language models (LLMs) by enabling users to run them locally on their machines. It represents a paradigm shift in machine learning, utilizing unsupervised learning and neural networks to learn language structures without explicit grammatical rules or annotations. The platform's multi-layered architecture allows it to process language from basic sounds to complex sentence structures without direct human intervention.

What sets Ollama apart is its comprehensive feature set: local execution capabilities that ensure privacy and faster processing, an extensive library of pre-trained LLMs including popular models like Llama 3, seamless integration with various tools and frameworks (such as Python, LangChain, and LlamaIndex), and robust customization options for fine-tuning models to specific needs. These features make it an accessible and powerful tool for both individuals and organizations looking to leverage LLMs in their applications and workflows.

#### Iteration 1

First Prompt: 

```
    "system", f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an education researcher analyzing classroom interactions and teaching practices.

    TASK: Extract and label classroom practices from multiple transcript texts using standardized codes.

    OUTPUT FORMAT: [{{
        "interaction_number": "Number of the interaction in batch",
        "text": "Original utterance or interaction",
        "codes": ["List of applicable codes"],
        "speaker": "Speaker identifier",
        "duration": "Duration value"
    }}]

    Key Rules:
    1. Code Application:
    - Apply all relevant codes to each utterance
    - Multiple codes may apply to a single utterance
    - Process ALL interactions in the input
    - Use exact code values from the provided list:
        - IST: Instructional Talk
        - ST: Social Talk
        - aOTR: Academic Opportunity to Respond
        - sOTR: Social Opportunity to Respond
        - REP: Reprimand
        - RED: Redirection
        - BSP: Behavior Specific Praise
        - GPRS: General Praise
        - aAFF: Academic Affirmation
        - aCORR: Academic Corrective
        - NEU: Neutral

    2. Text Processing:
    - Process each interaction separately
    - Preserve complete utterances for context
    - Include speaker and duration information
    - Keep related interactions together
    - Return results for ALL interactions in the batch
```

Took 1525 minutes to label interactions across 593 batches of 5 examples each. 

Issues with first prompt:  
* Did not include examples of the codes in the prompt
* Did not include SV and only included the teacher codes which did not match the model setup for the RoBERTa model.
* The text preprocessing was not identical to the RoBERTa model so not fair or correct to directly compare performance 


#### Iteration 2 

Second Prompt:

 ```   
    You are an education researcher analyzing classroom practices. Label each classroom interaction using these codes, applying all relevant codes to each utterance.

    CODES WITH EXAMPLES:
    1. IST (Instructional Talk):
    "Today we're going to learn about fractions"
    "Let's give some other people a chance"

    2. ST (Social Talk):
    "How was your weekend?"
    "Stephen, are you done with number 5 yet, or do you need more time?"

    3. aOTR (Academic Opportunity to Respond):
    "What's the capital of France?"
    "Solve problem number 5 on your worksheet"

    4. sOTR (Social Opportunity to Respond):
    "Who wants to be line leader today?"
    "Raise your hand if you're ready to start"

    5. REP (Reprimand):
    "I'm very disappointed in your behavior"
    "Stop talking while others are working"

    6. RED (Redirection):
    "Johnny, I need you to please turn around"
    "Remember to raise your hand"

    7. BSP (Behavior Specific Praise):
    "Great job raising your hand before speaking!"
    "I like how quietly Sarah lined up"

    8. GPRS (General Praise):
    "Good work!"
    "I love how great everyone is doing."

    9. aAFF (Academic Affirmation):
    "That's correct, 2 + 2 equals 4"
    "Yes, the main character is Harry"

    10. aCORR (Academic Corrective):
        "Not quite, try again"
        "The answer is kilometers, not meters"

    11. NEU (Neutral):
        "Now, hmmm."
        "Not right now."
    
    12. SV (Student Voice):
        "How much money does Jody have?" (aOTR). "45" (SV). 

    OUTPUT FORMAT: [
        {
            "Interaction_number": 1,
            "Transcript": "you  all  did  a  very  nice  job,  most  of  you  anyway,  packing  the  acorns  around.  Now,  Jeff  has  asked  a  very  good  question.",
            "Labels": ["BSP", "IST", "aOTR"],
            "Duration": 5.56
        }
    ]

    Key Rules:
    - Apply ALL relevant codes to each utterance
    - A single utterance may have multiple codes
    - Process each interaction separately but include all in response
    - Maintain exact wording and punctuation from transcripts

    Return only valid JSON without explanations.<|eot_id|>
```

Took [XXX] minutes to label interactions across  [XXX] batches of 5 examples each. 

## Results 


### Fine Tuned RoBERTa Model 

#### Specific Classroom Practices 

#### Concept-Level Metrics

| Category | F1 Score | Precision | Recall |
|----------|----------|-----------|---------|
| Academic_Feedback | 0.5965 | 0.6892 | 0.5258 |
| Corrective_Behavioral_Feedback | 0.3844 | 0.5074 | 0.3094 |
| Opportunity_to_Respond | 0.7894 | 0.8205 | 0.7606 |
| Other | 0.8119 | 0.6871 | 0.9919 |
| Praise | 0.5787 | 0.6441 | 0.5253 |
| Teacher_Talk | 0.8105 | 0.7385 | 0.8980 |

| Metric | Score |
|--------|-------|
| Macro F1 | 0.6619 |
| Micro F1 | 0.7754 |

#### Code-Level Metrics

| Code | F1 Score | Precision | Recall |
|------|----------|-----------|---------|
| BSP | 0.5799 | 0.5385 | 0.6282 |
| GPRS | 0.3463 | 0.5970 | 0.2439 |
| IST | 0.7608 | 0.7104 | 0.8189 |
| NEU | 0.1619 | 0.3269 | 0.1076 |
| RED | 0.2254 | 0.3902 | 0.1584 |
| REP | 0.3027 | 0.5490 | 0.2090 |
| ST | 0.4834 | 0.5424 | 0.4359 |
| SV | 0.7946 | 0.6678 | 0.9808 |
| aAFF | 0.6470 | 0.6747 | 0.6215 |
| aCORR | 0.3293 | 0.5400 | 0.2368 |
| aOTR | 0.7890 | 0.7748 | 0.8038 |
| sOTR | 0.3990 | 0.4566 | 0.3543 |

| Metric | Score |
|--------|-------|
| Macro F1 | 0.4849 |
| Micro F1 | 0.7038 |

### LLM In-context Learning

One of the challenges in using a LLM for classification tasks is that the output is not always returned in the desired or expected format. For small scae projects, this can often be manually detected and/or corrected. However, for larger projects, this can be time consuming and inefficient. 

NEED TO UPDATE 
| Code | F1 Score | Precision | Recall |
|------|----------|-----------|---------|
| BSP | 0.5799 | 0.5385 | 0.6282 |
| GPRS | 0.3463 | 0.5970 | 0.2439 |
| IST | 0.7608 | 0.7104 | 0.8189 |
| NEU | 0.1619 | 0.3269 | 0.1076 |
| RED | 0.2254 | 0.3902 | 0.1584 |
| REP | 0.3027 | 0.5490 | 0.2090 |
| ST | 0.4834 | 0.5424 | 0.4359 |
| SV | 0.7946 | 0.6678 | 0.9808 |
| aAFF | 0.6470 | 0.6747 | 0.6215 |
| aCORR | 0.3293 | 0.5400 | 0.2368 |
| aOTR | 0.7890 | 0.7748 | 0.8038 |
| sOTR | 0.3990 | 0.4566 | 0.3543 |

| Metric | Score |
|--------|-------|
| Macro F1 | 0.4849 |
| Micro F1 | 0.7038 |

I did not train the model specifically on Concept-Level labels, but if we aggregate the code-level labels into the concept-level labels we can see that the model performs better at the concept-level than the code-level.

| Category | F1 Score | Precision | Recall |
|----------|----------|-----------|---------|
| Academic_Feedback | 0.5965 | 0.6892 | 0.5258 |
| Corrective_Behavioral_Feedback | 0.3844 | 0.5074 | 0.3094 |
| Opportunity_to_Respond | 0.7894 | 0.8205 | 0.7606 |
| Other | 0.8119 | 0.6871 | 0.9919 |
| Praise | 0.5787 | 0.6441 | 0.5253 |
| Teacher_Talk | 0.8105 | 0.7385 | 0.8980 |

| Metric | Score |
|--------|-------|
| Macro F1 | 0.6619 |
| Micro F1 | 0.7754 |

## Discussion 

In analyzing the Code-Level Metrics, we observe significant variation in the model's performance across different classroom interaction codes. Academic-focused interactions show notably stronger performance, with Academic Opportunity to Respond (aOTR) achieving an F1 score of 0.79 and Instructional Talk (IST) reaching 0.76. Student Voice (SV) also demonstrates robust performance with an F1 score of 0.79. However, the model struggles considerably with neutral interactions (NEU, F1: 0.16) and behavioral management codes such as Redirection (RED, F1: 0.23) and Reprimand (REP, F1: 0.30). General Praise (GPRS) and Academic Corrective feedback (aCORR) also show relatively weak performance with F1 scores of 0.35 and 0.33 respectively. The substantial gaps between precision and recall scores for many codes suggest challenges in achieving balanced classification performance.

At the broader behavior category level, the model demonstrates stronger overall performance. The "Other" and "Teacher_Talk" categories lead with F1 scores of 0.81, followed closely by "Opportunity_to_Respond" at 0.79. These stronger performances at the category level suggest that the model better handles broader patterns of classroom interaction compared to specific behavioral codes. However, "Corrective_Behavioral_Feedback" remains challenging even at this level, with an F1 score of 0.38. Both "Academic_Feedback" and "Praise" show moderate performance with F1 scores around 0.59, indicating room for improvement in these important instructional categories.

There's a clear pattern where the model performs better at identifying broader behavioral categories (Macro F1: 0.66) compared to specific behavioral codes (Macro F1: 0.48). This suggests that fine-grained behavioral distinctions present a greater challenge for the model. There's a consistent strength in classifying academic-focused interactions compared to behavioral management ones, pointing to potential areas for model improvement. 

## Limitations 

We had to make certain exclusions from the training data for the RoBERTa model. This included removing any interactions that were labeled as "SIL" (Silence) or "UNI" (Unintelligible). This was done to ensure that the model was only trained on interactions that could be accurately labeled, if a model is given text from a transcript it doesn't make sense to have it try and identify silence or unintelligible speech. We also excluded clips less than 0.5 seconds in length. This was done to reduce the number of single word utternances that do not have enough context to be accurately labeled as a specific classroom practice. These exclusions were applied to the training, validation, and test data. The time length exclusions could be replicated a priori if this model were to be deployed, but more thought would need to be given to how to handle intelligible speech because Whisper usually produces a transcription for audio even if it doesn't make a lot of sense. 

## Hardware 

### Supervised Fine Tuning

## Future Work

- Improve diarization to separate teacher and student speech 
- Whisper Frame Classification using Audio Data
- Combine Audio and Text Data for Improved Classification

## Contact Info 

Jessica Boyle, Doctoral Student Department of Special Education, Vanderbilt University jessica.r.boyle@vanderbilt.edu

Wesley Morris, Doctoral Student Department of Psychology, Vanderbilt University wesley.g.morris@vanderbilt.edu

Isabel Arvelo, Masters Student Data Science Institute, Vanderbilt University isabel.c.arvelo@vanderbilt.Edu