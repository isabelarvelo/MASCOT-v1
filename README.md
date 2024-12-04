# MASCOT-cp 📓✏️🏫

The objective of this project is to label sections of audio files with specific classroom practices (e.g., praise, opportunities to respond, feedback) based on the statements made by the instructor. This task fits into an overall goal of providing real-time feedback to teachers regarding their daily interactions with their students. This is an ongoing project and collaboration with Jessica Boyle, a graduate student in the Department of Special Education, and Wesley Morris, a graduate student in the Department of Psychology and Human Development at Vanderbilt Univesity. 

## Project Overview 

### Motivation 

Effective classroom management is crucial for fostering student engagement and academic success, especially in inclusive classrooms with diverse learning and behavioral needs. Many teachers face challenges in consistently applying evidence-based practices due to resource constraints and the complexity of classroom dynamics. The tool we are building aims to provide teachers with real-time, individualized feedback to overcome the limitations of traditional, labor-intensive observation methods. 

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

Diariation is the process of segmenting and labeling audio data based on speaker identity. This is a critical step in the transcription process, as it allows for the identification of individual speakers and the separation of their speech. In this project, we are trying to identify and label specific classroom practices based on the statements made by the instructor so it is important that we are able to identify which parts of the classroom audio correspond to the teacher. We do not want to give the teacher feedback on their instruction based on speech produced by students. 

**Pyannote**: pyannote.audio is a Python-based open-source toolkit designed for speaker diarization. Built on the PyTorch machine learning framework, it provides  pretrained models and pipelines that can identify and separate different speakers in audio. Users can also fine-tune these models with their own data to achieve enhanced performance. 

**Whisper Frame Classification**: Wesley Morris, using resources from the LEAR lab, trained a model (MASCoT-CP1.0) to classify frames of audio as teacher speech or non-student speech. The model was trained on the same training, vaildation, and test splits as the RoBERTa model. 

After considering the coverage and purity of the diarization using both approaches, we found that the Whisper model was much more effective at distinguishing between student vs speaker voices. The median purity (how pure hypothesis segments are), coverage (the ratio of the duration of the intersection with the most co-occurring hypothesis segment and the duration of the reference segment) and pseudo f1 score (harmonic mean between purity and coverage) for the Whisper model are 0.963, 0.899, 0.929 respectively, while for the pyyanote diarization they are 0.952, 0.476, 0.631. Diarization is a huge area of research in this space, but is not the main focus of this project. Future work will focus on improving current diarization methods using packages like WhisperX or fine tuning other audio LLMs. 


## Transcription 

Stable-ts is a Python library designed to enhance Whisper's audio transcription capabilities, with a particular focus on improving timestamp accuracy. It modifies Whisper to produce more reliable start and end times for transcribed speech, while offering local execution capabilities and support for multiple output formats including SRT, VTT, ASS, TSV, and JSON. The library processes audio through a multi-layered approach, handling both word-level and segment-level timestamps, and can work with various audio formats.

The library includes advanced silence suppression that can be implemented in two ways: either through volume-based detection that analyzes audio intensity relative to neighboring sections, or through Silero VAD (Voice Activity Detection) for more complex audio environments. It offers timestamp refinement through an iterative process where portions of audio are muted and token probabilities are recalculated to find precise word boundaries. Additionally, it features customizable word regrouping algorithms that can split and merge segments based on punctuation, gaps, or custom rules. The library also supports gap adjustment to improve segment boundary accuracy and provides visualization tools for monitoring these adjustments. Simple to install via pip and compatible with any ASR (Automated Speech Recognition) system, not just Whisper, Stable-ts serves as a versatile tool for improving speech transcription accuracy, particularly in applications requiring precise audio synchronization.

## Text Classification 

Classroom practices do not always last a fixed amount of time. Sometimes an instructional talk (IST) segment may last 15 seconds, while other times it may last 3 seconds. The audio is fed into the model are based on speaker turns from the diarization step, but a teacher may exhibit multiple classroom practices within a single turn. Therefore, we need to set this up as a multi-label classification to identify one or more classroom practices within a single turn. 

### Supervised Fine Tuning 

#### RoBERTa (Robustly Optimized BERT Approach) 

RoBERTa (Robustly Optimized BERT Approach) is a popular model was trained using a massive 160GB text dataset (more than 10 times larger than BERT's), incorporating data from diverse sources including Wikipedia, news articles, Reddit content, and story-like text from Common Crawl. RoBERTa introduces several key technical improvements to BERT's architecture. It eliminates the Next Sentence Prediction (NSP) objective, finding that this removal either matches or improves downstream task performance. The model employs dynamic masking instead of static masking, generating new mask patterns each time data passes through the model. It also uses larger batch sizes (up to 8K sequences) and longer training sequences, which improves both the model's perplexity on masked language modeling and end-task accuracy. Like BERT, RoBERTa is pretrained using Masked Language Modeling (MLM), where it randomly masks 15% of words in input sentences and learns to predict them, enabling bidirectional representation learning. These optimizations led to state-of-the-art performance on various NLP benchmarks at the time of its release, including GLUE tasks, SQuAD, and RACE, demonstrating particular strength in tasks like natural language inference, textual entailment, and question answering.

Algorithm: RoBERTa vs BERT Training Comparison

/* Training approach differences between models */

Input: Text corpus D
Input: θ, initial transformer parameters
Output: θ̂, the trained parameters
Hyperparameters: Nepochs ∈ N, η ∈ (0, ∞), vocab_size, batch_size, max_steps

**BERT Training**
```
BERT(D, θ):
    V ← BuildCharacterBPE(D, vocab_size=30K)
    M ← GenerateStaticMasks(D)  // Generate once
    for step = 1,2,...,1M do
        B ← SampleBatch(D, size=256)
        for (x1,x2) in B do
            if random() < 0.5:
                x2 ← GetNextSegment(x1)
                is_next ← 1
            else:
                x2 ← GetRandomSegment()
                is_next ← 0
            
            mask ← M[x1,x2]  // Use pre-generated mask
            loss_mlm ← MLMLoss(x1,x2,mask,θ)
            loss_nsp ← NSPLoss(x1,x2,is_next,θ)
            loss ← loss_mlm + loss_nsp
            θ ← Optimize(loss,η)
    return θ̂=θ
```

**RoBERTa Training**
```
RoBERTa(D, θ):
    V ← BuildByteBPE(D, vocab_size=50K)
    for step = 1,2,...,500K do
        B ← SampleBatch(D, size=8K)
        for x in B do
            seqs ← []
            len ← 0
            while len < 512:
                s ← GetNextSentence()
                if len + length(s) > 512:
                    break
                seqs.append(s)
                len += length(s)
            
            x ← Concatenate(seqs)
            mask ← GenerateDynamicMask(x)  // New mask each time
            loss ← MLMLoss(x,mask,θ)
            θ ← Optimize(loss,η)
    return θ̂=θ
```

Main differences:
1. Masking: Static (BERT) vs Dynamic per step (RoBERTa)
  • BERT creates masks once during preprocessing and reuses them, seeing same mask ~4 times
  • RoBERTa generates new masks every time data is accessed, increasing training diversity

2. Batch size: 256 (BERT) vs 8K (RoBERTa) 
  • RoBERTa's larger batch size enables better parallelization and optimization
  • Requires learning rate scaling (from 1e-4 to 1e-3) and warmup tuning to maintain stability

3. Steps: 1M (BERT) vs 500K (RoBERTa)
  • Despite fewer steps, RoBERTa sees more data due to larger batch size
  • Total compute remains similar but RoBERTa's distribution is more efficient

4. NSP objective: Present in BERT, removed in RoBERTa
  • BERT uses NSP to learn document relationships by predicting if segments are consecutive
  • RoBERTa shows NSP doesn't improve downstream tasks and may hurt MLM training

5. Input format: Sentence pairs (BERT) vs packed sentences (RoBERTa)
  • BERT segments are limited by NSP requirement to use sentence pairs
  • RoBERTa packs continuous text to maximize usage of 512 token limit

6. Tokenization: Character BPE (BERT) vs Byte BPE (RoBERTa)
  • RoBERTa's byte-level BPE can encode any text without "unknown" tokens
  • Larger vocabulary (50K vs 30K) enables more efficient encoding of common patterns


### In-context learning with LLM

#### Qwen2.5:14b

[Qwen2.5](https://github.com/QwenLM/Qwen2.5) is a newly released family of open-source language models that represents a significant advancement in AI capabilities. The collection includes general language models ranging from 0.5B to 72B parameters, trained on 18 trillion tokens. These models support over 29 languages and can handle up to 128K input tokens while generating 8K tokens. Alongside the main language models, Qwen2.5 includes specialized variants: Qwen2.5-Coder, which comes in 1.5B, 7B, and 32B sizes and is specifically trained on 5.5 trillion tokens of code-related data, and Qwen2.5-Math, available in 1.5B, 7B, and 72B sizes, which focuses on mathematical reasoning in both Chinese and English.

The models show substantial improvements over their predecessors, achieving scores of 85+ on the MMLU knowledge benchmark and HumanEval coding tests, and 80+ on the MATH benchmark. They also demonstrate enhanced capabilities in following instructions, handling structured data, generating reliable JSON output, and performing role-play scenarios. Most models in the family are released under the Apache 2.0 license, with the exception of the 3B and 72B variants, and can be easily implemented using Hugging Face Transformers or deployed through platforms like vLLM and Ollama.

In this project, we use the base 14B Qwen2.5 model, which has the following features:

* Type: Causal Language Models
* Architecture: transformers with RoPE, SwiGLU, RMSNorm, and Attention QKV bias
* Number of Parameters: 14.7B
* Number of Paramaters (Non-Embedding): 13.1B
* Number of Layers: 48
* Number of Attention Heads (GQA): 40 for Q and 8 for KV
* Context Length: 131,072 tokens

Qwen2.5 resilient to the diversity of system prompts so the prompt does not need to be as structured or specific as prompts used for other models like the Llama family of models. 

#### Running the Model Locally 

[Ollama](https://ollama.com/) (Omni-Layer Learning Language Acquisition Model) is a platform that democratizes access to large language models (LLMs) by enabling users to run them locally on their machines. The platform's multi-layered architecture allows it to process language from basic sounds to complex sentence structures without direct human intervention. Ollama offers local execution capabilities that ensure privacy and faster processing, an extensive library of pre-trained LLMs including popular models like Llama 3, seamless integration with various tools and frameworks (such as Python, LangChain, and LlamaIndex), and robust customization options for fine-tuning model. These features make it an accessible and powerful tool that took only a few minutes to get up an running. 

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

Took 426m minutes to label interactions across 577 batches of 5 examples each. 

## Results 


### Fine Tuned RoBERTa Model 

#### Specific Classroom Practices 

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



### LLM In-context Learning

One of the challenges in using a LLM for classification tasks is that the output is not always returned in the desired or expected format. For small scae projects, this can often be manually detected and/or corrected. However, for larger projects, this can be time consuming and inefficient. 

| Code | F1 Score | Precision | Recall |
|------|----------|-----------|---------|
| BSP | 0.2817 | 0.1709 | 0.8000 |
| GPRS | 0.1615 | 0.2100 | 0.1313 |
| IST | 0.6859 | 0.6069 | 0.7887 |
| NEU | 0.1873 | 0.1165 | 0.4774 |
| RED | 0.2679 | 0.1891 | 0.4592 |
| REP | 0.3068 | 0.2070 | 0.5923 |
| ST | 0.2025 | 0.4040 | 0.1351 |
| SV | 0.0768 | 0.8105 | 0.0403 |
| aAFF | 0.3137 | 0.3520 | 0.2830 |
| aCORR | 0.2639 | 0.5588 | 0.1727 |
| aOTR | 0.7018 | 0.7927 | 0.6295 |
| sOTR | 0.1637 | 0.1850 | 0.1468 |


| Metric | Score |
|--------|-------|
| Macro F1 | 0.3011 |
| Micro F1 | 0.4179 |

I did not train the model specifically on Concept-Level labels, but if we aggregate the code-level labels into the concept-level labels we can see that the model performs better at the concept-level than the code-level.

| Category | F1 Score | Precision | Recall |
|------|----------|-----------|---------|
| Academic_Feedback | 0.3396 | 0.4245 | 0.2830 |
| Corrective_Behavioral_Feedback | 0.3895 | 0.2781 | 0.6498 |
| Opportunity_to_Respond | 0.7215 | 0.8042 | 0.6542 |
| Other | 0.4039 | 0.7466 | 0.2768 |
| Praise | 0.4915 | 0.3664 | 0.7465 |
| Teacher_Talk | 0.7440 | 0.7219 | 0.7675 |

| Metric | Score |
|--------|-------|
| Macro F1 | 0.5150 |
| Micro F1 | 0.5907 |


## Data 

The dataset includes 115 audio recordings of classroom instruction from 30 elementary school teachers. The audio , label files, and intermediate forms of data used for the project are not included in the repository because it is not publicly available and contains sensitive information. 

## Discussion 

In analyzing the Code-Level Metrics for the RoBERTa model, we observe significant variation in the model's performance across different classroom practices. Academic-focused interactions show notably stronger performance, with Academic Opportunity to Respond (aOTR) achieving an F1 score of 0.79 and Instructional Talk (IST) reaching 0.76. Student Voice (SV) also demonstrates fairly robust performance with an F1 score of 0.79. However, the model struggles considerably with neutral interactions (NEU, F1: 0.16) and behavioral management codes such as Redirection (RED, F1: 0.23) and Reprimand (REP, F1: 0.30). General Praise (GPRS) and Academic Corrective feedback (aCORR) also show relatively weak performance with F1 scores of 0.35 and 0.33 respectively. The substantial gaps between precision and recall scores for many codes suggest challenges in achieving balanced classification performance.

At the broader behavior category level, the model demonstrates stronger overall performance. The "Other" and "Teacher_Talk" categories lead with F1 scores of 0.81, followed closely by "Opportunity_to_Respond" at 0.79. These stronger performances at the category level suggest that the model better handles broader patterns of classroom interaction compared to specific behavioral codes. However, "Corrective_Behavioral_Feedback" remains challenging even at this level, with an F1 score of 0.38. Both "Academic_Feedback" and "Praise" show moderate performance with F1 scores around 0.59. The model performing better at identifying broader behavioral categories compared to specific behavioral codes suggests that fine-grained behavioral distinctions present a greater challenge for the model. There's a consistent strength in classifying academic-focused interactions compared to behavioral management ones. 

There are significant differences in the performance of the RoBERTa model trained through supervised fine tuning and the LLM training through in-context learning. RoBERTa demonstrates notably superior performance overall, with substantially higher macro F1 scores at both the code level and concept level. This performance advantage is particularly evident in academic interactions, where RoBERTa achieves stronger scores for codes like academic Opportunity to Respond (0.79 vs 0.70), Instructional Talk (0.76 vs 0.69), and Academic Feedback (0.65 vs 0.31). The models show comparable performance in codes like Redirection (RoBERTa 0.23 vs LLM 0.27) and Reprimand (both 0.30). A particularly notable difference emerges in Student Voice classification, where RoBERTa achieves a balanced F1 score of 0.79 with good precision (0.67) and recall (0.98), while the LLM shows a severe imbalance with an F1 of just 0.08, despite high precision (0.81) but negligible recall (0.04).

At the concept level, RoBERTa performs notably better in Academic Feedback (0.60 vs 0.34) and maintains an advantage in Opportunity to Respond (0.79 vs 0.72). The models show identical performance in Corrective Behavioral Feedback (both 0.38), while RoBERTa leads in Praise (0.58 vs 0.49). In broader categories, RoBERTa maintains its edge with stronger performance in Teacher Talk (0.81 vs 0.73) and Other (0.81 vs 0.41).

The fine tuned RoBERTa model demonstrates more stable performance across categories, while the LLM shows greater variance between different codes and categories. RoBERTa also maintains better balance between precision and recall, whereas the LLM tends toward extremes, particularly visible in codes like Student Voice. While both models perform better at the concept level than the code level, the performance gap between levels is smaller for RoBERTa (0.18) than the LLM (0.21).

RoBERTa appears more suitable for production use due to its higher overall performance, more consistent results, and better balanced precision-recall trade-offs. While the LLM might be adequate for broader category analysis where perfect accuracy isn't critical, RoBERTa proves more reliable for fine-grained analysis. Overall, the comparison suggests that while both models show promise, the fine-tuned RoBERTa model currently stands as the more robust choice for classroom interaction analysis, particularly when detailed code-level analysis is required.

## Limitations 

We had to make certain exclusions from the labeled data for the model. This included removing any interactions that were labeled as "SIL" (Silence) or "UNI" (Unintelligible). This was done to ensure that the model was only trained on interactions that could be accurately labeled, if a model is given text from a transcript it doesn't make sense to have it try and identify silence or unintelligible speech. We also excluded clips less than 0.5 seconds in length. This was done to reduce the number of single word utternances that do not have enough context to be accurately labeled as a specific classroom practice. These exclusions were applied to the training, validation, and test data. The time length exclusions could be replicated a priori if this model were to be deployed, but more thought would need to be given to how to handle intelligible speech because Whisper usually produces a transcription for audio even if it doesn't make a lot of sense. 

Furthermore, the test set is not exactly the same for the RoBERTa model and the LLM model because the Qwen2.5 model did not always return the example in the expected format so we had drop 23 examples. 


## Future Work

* Improve diarization to separate teacher and student speech 
    * RevAI, a leader in the diarization space, just released an open source model https://huggingface.co/Revai/reverb-diarization-v1 
* Try in-context learning with Claude or OpenAI model and then leverage prompt caching for reproducibility. This would allow us to provide more examples in the prompt. We could prune and add examples for certain types of classroom practices based on where the model seems to performing well vs struggling with 
* We are currently in the process of training a Whisper Frame Classification model toidentify the specific classroom practices instead of just teacher, not teacher 
* The goal is to integrate and audio and text data and yolk them together with a late biding approach to train a model that takes in multimodal input to predict the classroom practices.

## Contact Info 

Jessica Boyle, Doctoral Student Department of Special Education, Vanderbilt University jessica.r.boyle@vanderbilt.edu

Wesley Morris, Doctoral Student Department of Psychology, Vanderbilt University wesley.g.morris@vanderbilt.edu

Isabel Arvelo, Masters Student Data Science Institute, Vanderbilt University isabel.c.arvelo@vanderbilt.Edu