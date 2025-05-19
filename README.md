# 🏡 Egyptian Real Estate AI Agent — Arabic Conversational Assistant

This is a full-stack AI agent designed to help users find properties in Egypt via smart, multi-phase conversations — all powered by retrieval-augmented generation (RAG), Egyptian Arabic understanding, and customizable reasoning logic.

🎯 Built for MVPs, demos, and bootstrapping AI real estate products.

---

## 💡 Highlights

- 💬 Speaks Egyptian Arabic (custom prompts)
- 🧠 Tracks 7 smart conversation phases
- 🔍 Uses RAG to answer based on CSV knowledge (properties, trends, insights)
- 🤖 Works with **Gemini API**, **Together AI**, or local models (Yehia, DeepSeek)
- 🧩 Modular architecture: agent, reasoning, history, phase, config
- 🖥️ Streamlit/Gradio-ready with live chat interface
- 🆓 100% free-tier compatible (Colab + API or light models)

---

## 🧠 Conversation Phases

1. **DISCOVERY** – asks about location, budget, and type
2. **SUMMARY** – confirms extracted details from the user
3. **SUGGESTION** – shows matching property options
4. **PERSUASION** – handles objections ("السعر عالي", etc.)
5. **ALTERNATIVE** – suggests fallback listings
6. **URGENCY** – adds pressure to act now
7. **CLOSING** – collects name + phone to close deal

---

## 🧱 File Structure

├── agent.py # 💡 Main agent class, handles logic, user input, response
├── reasoning.py # 🔄 Phase transitions, intent reasoning, entity handling
├── config.py # 🔧 Phase enums, constants
├── phase_manager.py # 📌 Tracks current user phase in the conversation
├── history.py # 🧠 Dialog memory tracking (user & assistant turns)
├── knowledge/ # 🧠 CSV knowledge files (loaded into RAG)
│ ├── properties.csv
│ ├── market_trends.csv
│ ├── area_insights.csv
│ ├── cultural_preferences.csv
│ ├── rental_signals.csv
│ ├── client_signals.csv
├── knowledge_base.py # 🔎 Loads CSVs, supports filtering + RAG
├── retrieval.py # 📤 Queries knowledge base based on user info/context
├── main.py # 🧪 Script to test the agent in demo mode
├── app.py # 🖥️ Streamlit or Gradio interface (UI for the agent)
├── run.py # ▶️ Minimal runner (for Colab or CLI)
├── requirements.txt # 📦 Minimal dependencies (transformers, pandas, etc.)


# Optionally inject in agent.py or config.py
os.environ["API_KEY"] = "your_free_key"

python run.py         # Test CLI
python app.py         # Launch Gradio / Streamlit UI
👤: بدور على شقة في مدينة نصر
🤖: ممكن تقولي الميزانية؟ علشان أقدر أساعدك أكتر.

🧠 Under the Hood

    ✅ Custom prompt injection: ensures model behaves like a smart real estate advisor

    ✅ Modular chain-of-thought engine (reasoning.py)

    ✅ CSV-backed RAG search (filters by location, type, budget, features)

    ✅ Fallback & persuasion logic: if user says "غالي"، agent responds accordingly

    ✅ Dynamic prompts via Gemini or DeepSeek model

🚀 Business Value

    🔥 MVP-ready for investor demos or product trials

    💸 Cost-effective lead generation chatbot

    🇪🇬 Arabic-first: supports dialects and culturally relevant property advice

    🏗️ Scalable — plug in LLM or real-time CRM leads

    ✅ NO ChatGPT dependency. Can run 100% free with small models

..




🤖 Future Options

    Switch to Gemini Pro or Together AI for better replies

    Swap CSVs for a real NoSQL DB

    Add lead capture automation (Zapier/N8N)

    Deploy to Hugging Face Spaces or Replit

📃 License

MIT — Free to use, remix, and deploy...


