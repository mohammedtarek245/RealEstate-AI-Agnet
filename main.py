"""
Colab-compatible main.py for the Arabic Real Estate AI Agent.
Integrates all knowledge CSVs and rules.json with Gradio UI.
"""

import gradio as gr
import pandas as pd
import json
import os
from google.colab import files  # Only used if running in Colab

from agent import RealEstateAgent
from phase_manager import PhaseManager
from history import ConversationHistory

# === File names expected ===
csv_files = {
    "properties.csv",
    "area_insights.csv",
    "market_trends.csv",
    "rental_signals.csv",
    "client_signals.csv",
    "objection_patterns.csv",
    "personality types.csv"
}
json_file = "rules.json"

# === Upload & load knowledge files ===
print("📂 Upload CSV and JSON files...")
uploaded = files.upload()

# === Parse all CSVs ===
csv_data = {}
for fname in uploaded:
    if fname.endswith(".csv") and fname in csv_files:
        df = pd.read_csv(fname)
        key = fname.replace(".csv", "")
        csv_data[key] = df
        print(f"✅ Loaded: {fname} ({df.shape[0]} rows)")

# === Load rules.json ===
rules_data = {}
if json_file in uploaded:
    rules_data = json.loads(uploaded[json_file].decode("utf-8"))
    print(f"✅ Loaded: {json_file}")
else:
    print(f"⚠️ Warning: {json_file} not found!")

# === Init agent ===
phase_manager = PhaseManager()
conversation_history = ConversationHistory()
agent = RealEstateAgent(
    phase_manager=phase_manager,
    conversation_history=conversation_history,
    csv_data=csv_data,
    rules_data=rules_data
)


# === Gradio interface ===
def chat(user_input, chat_history=[]):
    try:
        response, new_state = agent.process_message(user_input, chat_history)
    except Exception as e:
        response = f"❌ خطأ: {str(e)}"
    chat_history.append((user_input, response))
    return chat_history, chat_history

# === Gradio UI ===
with gr.Blocks(css=".gradio-container {direction: rtl;}") as demo:
    gr.Markdown("## 🏠 مساعد العقارات الذكي")
    gr.Markdown("أهلاً بيك! اسألني عن عقار، ميزانية، موقع، أو شقة نفسك فيها ✨")

    chatbot = gr.Chatbot()
    user_input = gr.Textbox(placeholder="اكتب سؤالك هنا...", label="رسالة")
    state = gr.State([])

    user_input.submit(chat, [user_input, state], [chatbot, state])
    gr.ClearButton().click(lambda: ([], []), None, [chatbot, state])

# ✅ Launch
demo.launch(share=True)
