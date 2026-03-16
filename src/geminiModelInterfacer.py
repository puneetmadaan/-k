import google.genai as genai
import asyncio
import os



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
    raise ValueError("API_KEY not found in .env, or in google colab")

client = genai.Client(api_key=api_key)

async def getGoogleModels()->list[str]:
    ret = list()
    for item in client.models.list():
        ret.append(item)
    #print (ret)
    return ret

class GoogleModelsInterfacer:
    def __init__(self):
        self.googleModels = asyncio.run(getGoogleModels())
    
    def getModelNames(self)->list[str]:
        return [ m.name for m in self.googleModels] 
    
    def getGeminiSeriesModels(self)->list[str]:
        return [ m.name for m in self.googleModels if "gemini" in m.name.lower()]
    
    def getGeminiLiveSeriesModels(self)->list[str]:
        return [ m.name for m in self.googleModels if "gemini" in m.name.lower() and "live" in m.name.lower()]
    
    def getGeminiAudioSeriesModels(self)->list[str]:
        return [ m.name for m in self.googleModels if "gemini" in m.name.lower() and "audio" in m.name.lower()]
    
    def getGeminiLatestModels(self)->list[str]:
        return [ m.name for m in self.googleModels if "gemini" in m.name.lower() and "latest" in m.name.lower()]
    
    def getGeminiModalProperties(self, modelName:str) :
        if modelName not in self.getModelNames():
            raise ModuleNotFoundError("Model not found")
        else :
            return [ m for m in self.googleModels if m.name == modelName][0]
        
    def getGeminiLiveModals(self)->list[str]:
        return [m.name for m in self.googleModels if "gemini" in m.name.lower() and "bidiGenerateContent" in m.supported_actions]




if __name__ == "__main__":
    g = GoogleModelsInterfacer()
    print (g.getModelNames())
    print ("******")
    print (g.getGeminiSeriesModels())
    print ("******")
    print (g.getGeminiLiveSeriesModels())
    print ("******")
    print (g.getGeminiAudioSeriesModels())
    print ("******")
    print (g.getGeminiLatestModels())
    print ("******")
    print (g.getGeminiLiveModals())
    print ("******")
    print (g.getGeminiModalProperties(r'models/gemini-2.5-flash-native-audio-latest'))
    print ("******")
    print (g.getGeminiModalProperties(r'models/gemini-2.5-flash-lite'))
    print ("******")
    print (g.getGeminiModalProperties(r'models/gemini-3.1-flash-lite-preview'))
    print ("******")
