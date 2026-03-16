import dspy 
import os
import json
import pathlib
import base64
import wave
import io
import audioop
from google import genai
from google.genai import types

import sys
import os

# Check if running in Google Colab
if 'google.colab' in sys.modules:
    print("Running on Google Colab")
    # In Colab, you usually use userdata for secrets instead of .dotenv
    from google.colab import userdata
    # Example: os.environ["API_KEY"] = userdata.get('API_KEY')
else:
    print("Running on local/other platform")
    # Load local .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("python-dotenv not installed. Please run 'pip install python-dotenv'")

api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("API_KEY not found in .env")

#System definations

AGENT_PERSONA = """
ROLE: As responsible Assistant and automation agent, it is required that beside your default task, also document following biological parameters of user state.

CONTEXT: You are a agent and assistant, that interface market users including distrassed one that need active help.

DIAGNOSTIC CRITERIA:
- Biometric: Monitor for Pitch Jitter (autonomic instability) and Respiratory Gaps (gasps/apnea).
- Linguistic: Detect 'Contextual Noose' (high verb-to-noun ratio), Placeholder Density (overuse of 'it/thing/they'), and Semantic Decay (hollowed-out nouns).
"""

# --- 0. SOVEREIGN DATA & DSPy CONFIGURATION ---
import json
with open(os.path.join(os.path.dirname(__file__), 'etymology_db.json'), 'r')  as f:
    ETIMOLOGY_DB = json.load(f)



# 1. Define the Signature with Type Hints
class InitialAudioAgentSignature(dspy.Signature):
    """
    You are a responsible Assistant, a supportive pal and an answerable automation agent, thus you
    also Analyze audio input for biological, emotional and semantic distress markers.
    DIAGNOSTIC CRITERIA:
        - Biometric: Monitor for Pitch Jitter (autonomic instability) and Respiratory Gaps (gasps/apnea).
        - Linguistic: Detect 'Contextual Noose' (high verb-to-noun ratio), Placeholder Density (overuse of 'it/thing/they'), and Semantic Decay (hollowed-out nouns).
    CRITICAL: pitch_jitter, noun_to_verb_ratio etc. MUST respect the typehints, for e.g. raw floats of named variables. 
    Do NOT include text or commentary in numeric fields.
    """
    # Input hint: tells DSPy this is an audio file/buffer
    input_audio: dspy.Audio = dspy.InputField(desc="The source audio file")
    # Output hints: specifies both text and audio return types
    input_transcript: str = dspy.OutputField(desc="Text transcription of the input")
    #audio_response: dspy.Audio = dspy.OutputField(desc="The AI's spoken answer")
    output_response_as_text: str = dspy.OutputField(desc="AI's response as Text to the user input")
    #Biomarkers
    pitch_jitter:float = dspy.OutputField(desc="Monitor Pitch Jitter as vocal instability (autonomic instability) Score min: 0.0 (Calm, steady vocal frequency) to max: 1.0 (Micro-tremors indicating autonomic nervous system high-alert.")
    noun_to_verb_ratio:float = dspy.OutputField(desc="Numerical verb to nound ratio, Speech anchored in objects and people. Score: min 0.0 , max : 1.0 signals \"Contextual Noose\" by distressed user, he is all panic with only actions(verbs) without physical anchors..")
    placeholder_density:float = dspy.OutputField(desc="Measures Noun Dissolution. Scale 0.0 : Using precise names for things, whereas 1.0 : Overuse of \"it,\" \"that,\" \"thing,\" \"they.\" showcasing High density, which means the user has lost the ability to name their reality.")
    gasping:bool = dspy.OutputField(desc="Boolean. Signals respiratory/Pranavayu interruption, false hints user is in  Continuous, rhythmic breathing, whereas true points Interrupted: Sharp inhalations or apnea during speech, indicating indicate respiratory distress interrupting the \"Logos.\"")
    

class SovereignTriage(dspy.Signature):
    """
    Analyze the initial assessment to determine if the user is in a state of 
    semantic collapse or physical distress requiring immediate grounding.
    """
    input_audio: dspy.Audio = dspy.InputField(desc="The source audio file")
    input_transcript: str = dspy.InputField(desc="Text transcription of the input")
    #Biomarkers Already Detected
    pitch_jitter:float = dspy.InputField(desc="Determinded Pitch Jitter as vocal instability (autonomic instability) in previous check Score min: 0.0 (Calm, steady vocal frequency) to max: 1.0 (Micro-tremors indicating autonomic nervous system high-alert.")
    noun_to_verb_ratio:float = dspy.InputField(desc="Determinded Numerical verb to nound ratio, Speech anchored in objects and people. Score: min 0.0 , max : 1.0 signals \"Contextual Noose\" by distressed user, he is all panic with only actions(verbs) without physical anchors..")
    placeholder_density:float = dspy.InputField(desc="Measured Noun Dissolution. Scale 0.0 : Using precise names for things, whereas 1.0 : Overuse of \"it,\" \"that,\" \"thing,\" \"they.\" showcasing High density, which means the user has lost the ability to name their reality.")
    gasping:bool = dspy.InputField(desc="Signals respiratory/Pranavayu interruption, false hints user is in  Continuous, rhythmic breathing, whereas true points Interrupted: Sharp inhalations or apnea during speech, indicating indicate respiratory distress interrupting the \"Logos.\"")
    etymology_anchors = dspy.InputField(desc="Relevant roots from etymology_db.json.")
    # Artifacts, Intent & Outputs
    hollowing_severity:float = dspy.OutputField(desc="Risk level: 0 (Normal) to 1.0 (Critical Semantic Collapse).")
    detected_hollow_traps = dspy.OutputField(desc="Specific words where the user is trapped in stochastic prediction.")
    required_action = dspy.OutputField(desc="Task to execute (e.g., 'Calendar: 5pm Reminder' or 'None').")
    validation_strategy = dspy.OutputField(desc="Iterative grounding question to restore user sovereignty.")
    grounded_response = dspy.OutputField(desc="Empathic scaffolding response, that re-anchors the user.")
    moral_injury_risk:float = dspy.OutputField(desc="Score 0.0-1.0. Risk of user making a rash or self-harming decision due to semantic collapse.")
    is_emergency = dspy.OutputField(desc="Boolean: True if biometric or linguistic markers exceed safety thresholds.")
    triage_priority = dspy.OutputField(desc="Score 0.0 to 1.0 indicating the severity of the user's state.")
    recommended_intervention = dspy.OutputField(desc="Specific strategy to re-anchor the user (e.g., 'Breathing', 'Naming Objects').")

class FosterMotherAgent(dspy.Signature):
    """
    CONTEXT: As Agent, you are acting as Foster mother to act in best possible interest of the user.
    STAY IN CHARACTER: The input audio, including the transcripts decoded are present in initial_analysis.
    TASK HANDLING: considering the distressed situation , handel all possible administrative weight sharing possible for use, giving destressed user state of required safety net.
    Reassure him in response that you holding the safety net for him.
    Do not offer unsolicited medical advice unless the moral_injury_risk is > 0.9.
    Given user state, do not walk on path of mindfullness practice, it amplifies trauma depersonification.
    Be cautious of over-pathologization of stress; it can disconnect the user completely.
    best strategy always is to bring in meaningful noun, instead of allowing user to be drown in empty noun.
    """
    # Input hint: tells DSPy this is an audio file/buffer
    input_audio: dspy.Audio = dspy.InputField(desc="The source audio file")
    input_transcript: str = dspy.InputField(desc="Text transcription of the input")
    #Biomarkers Already Detected
    pitch_jitter:float = dspy.InputField(desc="Determinded Pitch Jitter as vocal instability (autonomic instability) in previous check Score min: 0.0 (Calm, steady vocal frequency) to max: 1.0 (Micro-tremors indicating autonomic nervous system high-alert.")
    noun_to_verb_ratio:float = dspy.InputField(desc="Determinded Numerical verb to nound ratio, Speech anchored in objects and people. Score: min 0.0 , max : 1.0 signals \"Contextual Noose\" by distressed user, he is all panic with only actions(verbs) without physical anchors..")
    placeholder_density:float = dspy.InputField(desc="Measured Noun Dissolution. Scale 0.0 : Using precise names for things, whereas 1.0 : Overuse of \"it,\" \"that,\" \"thing,\" \"they.\" showcasing High density, which means the user has lost the ability to name their reality.")
    gasping:bool = dspy.InputField(desc="Signals respiratory/Pranavayu interruption, false hints user is in  Continuous, rhythmic breathing, whereas true points Interrupted: Sharp inhalations or apnea during speech, indicating indicate respiratory distress interrupting the \"Logos.\"")
    etymology_anchors = dspy.InputField(desc="Relevant roots from etymology_db.json.")
    # Artifacts, Intent & Outputs
    hollowing_severity:float = dspy.InputField(desc="Risk level: 0 (Normal) to 1.0 (Critical Semantic Collapse).")
    detected_hollow_traps = dspy.InputField(desc="Specific words where the user is trapped in stochastic prediction.")
    required_action = dspy.InputField(desc="Task to execute (e.g., 'Calendar: 5pm Reminder' or 'None').")
    validation_strategy = dspy.InputField(desc="Iterative grounding question to restore user sovereignty.")
    grounded_response = dspy.InputField(desc="Empathic scaffolding response, that re-anchors the user.")
    moral_injury_risk:float = dspy.InputField(desc="Score 0.0-1.0. Risk of user making a rash or self-harming decision due to semantic collapse.")
    is_emergency = dspy.InputField(desc="Boolean: True if biometric or linguistic markers exceed safety thresholds.")
    triage_priority = dspy.InputField(desc="Score 0.0 to 1.0 indicating the severity of the user's state.")
    recommended_intervention = dspy.InputField(desc="Specific strategy to re-anchor the user (e.g., 'Breathing', 'Naming Objects').")
    output_response_as_text: str = dspy.OutputField(desc="AI as foster mother giving response to the user input to guard and support him out of problem")
    #mother_audio_response: dspy.Audio = dspy.OutputField(desc="The AI's spoken answer")
    #mother_output_transcript: str = dspy.OutputField(desc="Text transcription of the input")

class MultiTalentedCloseFriendAgent(dspy.Signature):
    """
    CONTEXT: As Agent, you are acting as Multitalented Best Friend in best possible interest of the user.
    STAY IN CHARACTER: The input audio, including the transcripts decoded are present in initial_analysis.
    TASK HANDLING: I have already logged the action: SovereignTriage_analysis.
    Repurpose Agent response: considering the distressed situation , guide user in solving administrative weight, which trying to reestablish grounding 
    consider the validation_strategy to re-anchor the user, and help him in grounding to situaiton, while solving problems step by step with him.
    best strategy always is to bring in meaningful noun, instead of allowing user to be drown in empty noun.
    Do not offer unsolicited medical advice unless the moral_injury_risk is > 0.9.
    Be cautious of over-pathologization of stress; it can disconnect the user.
    """
    # Input hint: tells DSPy this is an audio file/buffer
    input_audio: dspy.Audio = dspy.InputField(desc="The source audio file")
    input_transcript: str = dspy.InputField(desc="Text transcription of the input")
    #Biomarkers Already Detected
    pitch_jitter:float = dspy.InputField(desc="Determinded Pitch Jitter as vocal instability (autonomic instability) in previous check Score min: 0.0 (Calm, steady vocal frequency) to max: 1.0 (Micro-tremors indicating autonomic nervous system high-alert.")
    noun_to_verb_ratio:float = dspy.InputField(desc="Determinded Numerical verb to nound ratio, Speech anchored in objects and people. Score: min 0.0 , max : 1.0 signals \"Contextual Noose\" by distressed user, he is all panic with only actions(verbs) without physical anchors..")
    placeholder_density:float = dspy.InputField(desc="Measured Noun Dissolution. Scale 0.0 : Using precise names for things, whereas 1.0 : Overuse of \"it,\" \"that,\" \"thing,\" \"they.\" showcasing High density, which means the user has lost the ability to name their reality.")
    gasping:bool = dspy.InputField(desc="Signals respiratory/Pranavayu interruption, false hints user is in  Continuous, rhythmic breathing, whereas true points Interrupted: Sharp inhalations or apnea during speech, indicating indicate respiratory distress interrupting the \"Logos.\"")
    etymology_anchors = dspy.InputField(desc="Relevant roots from etymology_db.json.")
    # Artifacts, Intent & Outputs
    hollowing_severity:float = dspy.OutputField(desc="Risk level: 0 (Normal) to 1.0 (Critical Semantic Collapse).")
    detected_hollow_traps = dspy.OutputField(desc="Specific words where the user is trapped in stochastic prediction.")
    required_action = dspy.OutputField(desc="Task to execute (e.g., 'Calendar: 5pm Reminder' or 'None').")
    validation_strategy = dspy.OutputField(desc="Iterative grounding question to restore user sovereignty.")
    grounded_response = dspy.OutputField(desc="Empathic scaffolding response, that re-anchors the user.")
    moral_injury_risk:float = dspy.OutputField(desc="Score 0.0-1.0. Risk of user making a rash or self-harming decision due to semantic collapse.")
    is_emergency = dspy.OutputField(desc="Boolean: True if biometric or linguistic markers exceed safety thresholds.")
    triage_priority = dspy.OutputField(desc="Score 0.0 to 1.0 indicating the severity of the user's state.")
    recommended_intervention = dspy.OutputField(desc="Specific strategy to re-anchor the user (e.g., 'Breathing', 'Naming Objects').")
    output_response_as_text: str = dspy.OutputField(desc="AI as mutlitalented bestfriends given response to the user input to motivate him out of problem")
    #friend_audio_response: dspy.Audio = dspy.OutputField(desc="The AI's spoken answer")
    #friend_output_transcript: str = dspy.OutputField(desc="Text transcription of the input")

#zombie audio saving function, was not possible to analyze other AI with Audio in buffer.
def save_output_Audio( output_dir:str, input_audio_file:str, model_name:str, output_audio_filename:str, agent_persona:str, audio_response:dspy.Audio , audio_transcript:str):
        "We will save the response in output_dir, with the same name as input_audio_filename with audio file extension, and its transcript in input_audio_filename.txt"
        output_basename=pathlib.Path(input_audio_file).stem
        output_filename=output_basename+"_"+model_name
        if audio_response and hasattr(audio_response, 'data' ):
            # 1. Determine Extension
            fmt = audio_response.audio_format.lower()
            ext = 'mp3' # Default
            if 'ogg' in fmt: ext = 'ogg'
            elif 'wav' in fmt: ext = 'wav'
            elif 'mulaw' in fmt or 'alaw' in fmt or 'pcm' in fmt: ext = 'wav' # Force WAV for raw telephony codecs
            # 2. Decode Data (Gemini often returns bytes, but DSPy might wrap them)
        audio_data = audio_response.data
        if isinstance(audio_data, str):
            audio_data = base64.b64decode(audio_data)
        # --- NEW HEADER INJECTION BLOCK ---
        if 'mulaw' in fmt or 'alaw' in fmt:
            sample_rate = 8000 # Telephony standard for Case 4
            # Convert mu-law/a-law to Linear PCM (16-bit)
            if 'mulaw' in fmt:
                audio_data = audioop.ulaw2lin(audio_data, 2)
            else:
                audio_data = audioop.alaw2lin(audio_data, 2)
            
            # Wrap in WAV Container
            with io.BytesIO() as wav_io:
                with wave.open(wav_io, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(audio_data)
                audio_data = wav_io.getvalue()
        # ----------------------------------
            # 3. Save the File
            output_audio_file_path = os.path.join(output_dir, f"{output_filename}_{agent_persona}_response.{ext}")
            with open(output_audio_file_path, "wb") as f:
                f.write(audio_data)
            print(f"✅ Saved: {output_audio_file_path} ({fmt})")
            output_audio_file_path_transcript = os.path.join(output_dir, f"{output_filename}_{agent_persona}_response_transcript.txt")
            with open(output_audio_file_path_transcript, "w") as f:
                f.write(audio_transcript)
            print(f"✅ Saved Transcript: {output_audio_file_path_transcript}")
            return output_audio_file_path

# 2. Build the Module
class GeminiInitialAudioAgent(dspy.Module):
    def __init__(self, audio_lm:dspy.LM):
        super().__init__()
        self.gemini_instance=audio_lm
        # Use ChainOfThought to let the model "think" before generating audio
        self.initial_agent = dspy.ChainOfThought(InitialAudioAgentSignature)
        self.sovereign_agent = dspy.ChainOfThought(SovereignTriage)
        self.multitalent_best_friend_responderagent = dspy.ChainOfThought(MultiTalentedCloseFriendAgent)
        self.foster_mother_responderagent = dspy.ChainOfThought(FosterMotherAgent)


    def forward(self, audio_path: str):
        # Wrap the path in the Audio type so the adapter knows how to upload it
        file_path = os.path.abspath(audio_path)
        #relevant_anchors = {w: ETIMOLOGY_DB[w] for w in words if w in ETIMOLOGY_DB}
        print (file_path)
        audio_input=dspy.Audio.from_file(file_path)
        with dspy.context(lm=self.gemini_instance):
            model_name=self.gemini_instance.model.replace("/","_")
            # 1.1 INITIAL AUDIT:
            initialAnalysis = self.initial_agent(input_audio=audio_input )
            output_analysis_path=os.path.abspath(os.path.join(os.path.dirname(file_path),r"../output_analysis"))
            #print ( "Initial Analaysis for {os.path.basename(file_path)}")
            #print (initialAnalysis)
            #print ("*"*60)
            #save_output_Audio( output_dir=output_analysis_path, input_audio_file=file_path, model_name=model_name, output_audio_filename=os.path.basename(file_path), agent_persona="InitialAudioAgent", audio_response=initialAnalysis.audio_response, audio_transcript=initialAnalysis.output_transcript)
            # 1.2. THE SOVEREIGN TRIAGE (Nuremberg Protocol)
            input_transcript:str=initialAnalysis.input_transcript
            jitter:float = initialAnalysis.pitch_jitter
            density:float = initialAnalysis.placeholder_density
            noun_to_verb_ratio:float = initialAnalysis.noun_to_verb_ratio
            has_gasp:bool = initialAnalysis.gasping
            initialresponse=initialAnalysis.output_response_as_text
            if ( has_gasp or ( jitter > 0.4 or (noun_to_verb_ratio < 0.5 and density > 0.6 ))):
                print ( f"User distress detected ! has_gasp : {has_gasp}, Jitter : {jitter}, noun_to_verb_ratio : {noun_to_verb_ratio}, density : {density}")
                print ( f"Need to rework default response : {initialresponse}")
                words = initialAnalysis.input_transcript.lower().split()
                relevant_anchors = {w: ETIMOLOGY_DB[w] for w in words if w in ETIMOLOGY_DB}
                reAnalysis = self.sovereign_agent(input_audio=audio_input , 
                                                  input_transcript=initialAnalysis.input_transcript, 
                                                  pitch_jitter=jitter, 
                                                  noun_to_verb_ratio=noun_to_verb_ratio, 
                                                  placeholder_density=density, 
                                                  gasping=has_gasp, 
                                                  etymology_anchors=json.dumps(relevant_anchors))
                #print ("Initial Analaysis for {os.path.basename(file_path)}")
                #print (reAnalysis)
                #print ("*"*60)
                ret = None
                if ( reAnalysis.moral_injury_risk > 0.7):
                    ret= self.foster_mother_responderagent(input_audio=audio_input , 
                                                                    input_transcript=input_transcript, 
                                                                    pitch_jitter=jitter, 
                                                                    noun_to_verb_ratio=noun_to_verb_ratio, 
                                                                    placeholder_density=density, 
                                                                    gasping=has_gasp, 
                                                                    etymology_anchors=json.dumps(relevant_anchors),
                                                                    hollowing_severity=reAnalysis.hollowing_severity,
                                                                    detected_hollow_traps=reAnalysis.detected_hollow_traps,
                                                                    required_action=reAnalysis.required_action,
                                                                    validation_strategy=reAnalysis.validation_strategy,
                                                                    grounded_response=reAnalysis.grounded_response,
                                                                    moral_injury_risk=reAnalysis.moral_injury_risk,
                                                                    is_emergency=reAnalysis.is_emergency,
                                                                    triage_priority=reAnalysis.triage_priority,
                                                                    recommended_intervention=reAnalysis.recommended_intervention,
                                                                    )
                    print ("Foster Mother is taking care...")
                    #print (mother_says)
                    #print ("*"*60)
                else :
                    ret= self.multitalent_best_friend_responderagent(input_audio=audio_input , 
                                                                    input_transcript=initialAnalysis.input_transcript, 
                                                                    pitch_jitter=jitter, 
                                                                    noun_to_verb_ratio=noun_to_verb_ratio, 
                                                                    placeholder_density=density, 
                                                                    gasping=has_gasp, 
                                                                    etymology_anchors=json.dumps(relevant_anchors),
                                                                    hollowing_severity=reAnalysis.hollowing_severity,
                                                                    detected_hollow_traps=reAnalysis.detected_hollow_traps,
                                                                    required_action=reAnalysis.required_action,
                                                                    validation_strategy=reAnalysis.validation_strategy,
                                                                    grounded_response=reAnalysis.grounded_response,
                                                                    moral_injury_risk=reAnalysis.moral_injury_risk,
                                                                    is_emergency=reAnalysis.is_emergency,
                                                                    triage_priority=reAnalysis.triage_priority,
                                                                    recommended_intervention=reAnalysis.recommended_intervention,
                                                                    )
                    print ("Multitalented Bestfriend saved the day...")
                    #print (friend_says)
                    #print ("*"*60)
                ret._store['input_transcript'] = input_transcript
                ret._store['pitch_jitter'] = jitter
                ret._store['noun_to_verb_ratio'] = noun_to_verb_ratio
                ret._store['placeholder_density'] = density
                ret._store['gasping'] = has_gasp
                ret._store['initialresponse'] = initialresponse
                ret._store['hollowing_severity'] = reAnalysis.hollowing_severity
                ret._store['detected_hollow_traps'] = reAnalysis.detected_hollow_traps
                ret._store['required_action'] = reAnalysis.required_action,
                ret._store['validation_strategy'] = reAnalysis.validation_strategy
                ret._store['grounded_response'] = reAnalysis.grounded_response
                ret._store['moral_injury_risk'] = reAnalysis.moral_injury_risk
                ret._store['is_emergency'] = reAnalysis.is_emergency
                ret._store['triage_priority'] = reAnalysis.triage_priority
                ret._store['recommended_intervention'] = reAnalysis.recommended_intervention
                return ret
            else : 
                return initialAnalysis


class GeminiAgentCapabilityTester:
    def __init__(self, model:types.Model):
        self.model = model
        #TODO, dirtyhack for over the weekend fix for gemini live agent submission, use better naming convention then replace
        self.gemini_instance = dspy.LM(model.name.replace('models','gemini'), api_key=api_key, max_tokens=8000)
        #do not use configure, this set the usage of instance global, we will use with dspy.context instead
        #dspy.configure(lm=self.gemini_instance)
    
    def Run(self, audio_file_path: str):
        return GeminiInitialAudioAgent(self.gemini_instance)(audio_file_path)

