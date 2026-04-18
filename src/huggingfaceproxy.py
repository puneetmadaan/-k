import dspy
from gradio_client import Client, handle_file

class QwenOmniGradio(dspy.LM):
    def __init__(self, model="Qwen/Qwen3.5-Omni-Online-Demo", **kwargs):
        self.client = Client(model)
        super().__init__(model)

    def __call__(self, prompt=None, messages=None, audio_link=None, **kwargs):
        # 1. Extract the query
        query = prompt if prompt else ""
        if messages:
            query = messages[-1].get("content", "")

        # 2. Match your successful Gradio call
        target_audio = audio_link if audio_link else 'https://raw.githubusercontent.com/puneetmadaan/-k/refs/heads/main/audio_samples/case_0.wav'
        print ( f"target_audio : {target_audio}")

        try:
            result = self.client.predict(
                audio=handle_file(target_audio),
                video=None,
                # We pass the instruction directly into the history as seen in your trial
                history=[{"role": "user", "content": query}],
                voice_choice="Tina / 中文-甜甜",
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                api_name="/media_predict"
            )
            
            # 3. Extracting the text response
            # Your output showed a list: [None, None, [user_msg, audio_input_msg, assistant_text_msg, assistant_audio_msg]]
            chat_history = result[2]
            
            # Find the assistant message that contains text content
            # Based on your trace, it's the one with 'content' as a string
            response_text = ""
            for msg in reversed(chat_history):
                if msg['role'] == 'assistant' and isinstance(msg['content'], str):
                    response_text = msg['content']
                    break
            
            # 4. Mandatory return for ChatAdapter
            return [{"text": response_text}]
            
        except Exception as e:
            return [{"text": f"Gradio Error: {str(e)}"}]
        
    
def main():
    qwen_omni = QwenOmniGradio()
    from kca import InitialAudioAgentSignature, SovereignTriage, FosterMotherAgent, MultiTalentedCloseFriendAgent
    #dspy.settings.configure(lm=qwen_omni, adapter=dspy.ChatAdapter())
    auditor = dspy.ChainOfThought(InitialAudioAgentSignature)
    # Configure DSPy
    dspy.settings.configure(lm=qwen_omni, adapter=dspy.ChatAdapter())
    with dspy.context(lm=qwen_omni):
        # Using your Case 1 (Frustration Audit)
        audio_case_1 = 'https://raw.githubusercontent.com/puneetmadaan/-k/refs/heads/main/audio_samples/case_4.wav'
        response = auditor(input_audio=audio_case_1)
        
        # Check the "Warranty Gap"
        print(f"Audit Response: {response}")
        
        # TURN ON TRACING HERE
        print("\n--- DSPy Internal Trace ---")
        qwen_omni.inspect_history(n=1)



def mainly():
    # --- Run a side-by-side audit loop ---
    qwen_omni = QwenOmniGradio()

    # Define your Signature
    class VoiceAudit(dspy.Signature):
        """Analyze the audio file and provide a transcript or insight."""
        context = dspy.InputField()
        analysis = dspy.OutputField()

    # Run this before calling your auditor
    dspy.settings.configure(
        lm=qwen_omni, 
        adapter=dspy.ChatAdapter()
    )

    auditor = dspy.Predict(VoiceAudit)

    # Configure DSPy
    dspy.settings.configure(lm=qwen_omni, adapter=dspy.ChatAdapter())

    # Run Audit
    with dspy.context(lm=qwen_omni):
        # Using your Case 1 (Frustration Audit)
        audio_case_1 = 'https://raw.githubusercontent.com/puneetmadaan/-k/refs/heads/main/audio_samples/case_3.wav'
        response = auditor(context="Transcribe this and detect the emotion.", audio_link=audio_case_1)
        
        # Check the "Warranty Gap"
        print(f"Audit Response: {response.analysis}")
        
        # TURN ON TRACING HERE
        print("\n--- DSPy Internal Trace ---")
        qwen_omni.inspect_history(n=1)
    

if __name__ == "__main__":
    main()
