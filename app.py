import openai # Using OpenAI's Vision API

def analyze_chart(image_path):
    # Set up the analysis rules
    system_prompt = """
    Analyze this 1-minute candlestick chart. 
    1. Identify Engulfing, Hammer, or Shooting Star patterns.
    2. Check for Support/Resistance levels.
    3. Output: DIRECTION (UP/DOWN) and CONFIDENCE (%)
    """
    
    # Send image to AI for analysis
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_path}"}}
            ]}
        ]
    )
    return response.choices[0].message.content
