from image_agent import ImageAgent
from model import Gemma3ImageDescriber

prompt = "Describe the image in detailed within 200 words. Include features of the landscape, activities, possible region, possible country, quantitative features if applicable. "
max_tokens = 256
describer = Gemma3ImageDescriber(model_id = "google/gemma-3-4b-it", quantization_config=None)

agent = ImageAgent(describer, image_folder="/home/ntu/Downloads/gemma3-test/selected", prompt=prompt, max_tokens=max_tokens)
agent.start()

agent.display_loop()