import os
import re
import json
import logging
from typing import List, Dict, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from config import ConversationPhase
from phase_manager import PhaseManager
from history import ConversationHistory
from reasoning import Reasoning
from knowledge_base import KnowledgeBase
from retrieval import KnowledgeRetrieval


tokenizer = AutoTokenizer.from_pretrained("lightblue/DeepSeek-R1-Distill-Qwen-1.5B-Multilingual", use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    "lightblue/DeepSeek-R1-Distill-Qwen-1.5B-Multilingual",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer.pad_token = tokenizer.eos_token

# === Define Prompt Function ===
def generate_arabic_response(prompt):
    character_prompt = (
        "أنت وكيل عقارات ذكي باللهجة المصرية. "
        "مهمتك تساعد العميل تلاقي شقة مناسبة حسب المكان، الميزانية، والاحتياجات. "
        "خليك مختصر، واضح، وبلُغة مقنعة وسهلة.\n"
    )
    full_prompt = character_prompt + f"\nالمستخدم قال: {prompt}\nرد الوكيل:"

    inputs = tokenizer(
    full_prompt,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=512  
)
    inputs["attention_mask"] = inputs["input_ids"].ne(tokenizer.pad_token_id)

    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        repetition_penalty=1.3,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.pad_token_id
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("رد الوكيل:")[-1].strip()


# === Logger ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealEstateAgent:
    def __init__(self,
                 phase_manager: PhaseManager,
                 conversation_history: ConversationHistory,
                 csv_data: Dict[str, object],   
                 rules_data: Dict = None,      
                 dialect: str = "Egyptian"):
        
        self.phase_manager = phase_manager
        self.conversation_history = conversation_history
        self.dialect = dialect
        self.reasoning_engine = Reasoning()
        self.user_info = {}
        self.context = {}
        self.selected_properties = []
        self.last_mentioned_property = None
        self.current_phase = self.phase_manager.get_current_phase() or ConversationPhase.DISCOVERY
        self.asked_questions = set()

      
        self.knowledge_base = KnowledgeBase(csv_data=csv_data, rules_data=rules_data)
        self.knowledge_retriever = KnowledgeRetrieval(self.knowledge_base)
        self.rules = rules_data or {}

    def process_message(self, user_message: str, state: list = []) -> Tuple[str, list]:
        self.conversation_history.add_user_message(user_message)
        logger.info(f"📩 Message: {user_message}")
        logger.info(f"🔄 Phase: {self.current_phase}")
        logger.info(f"📌 User info: {self.user_info}")

        self._basic_info_extraction(user_message)

        relevant_knowledge = self.knowledge_retriever.retrieve(
            query=user_message,
            phase=self.current_phase,
            context={"user_info": self.user_info}
        )

        reasoning_result = self.reasoning_engine.analyze(
            message=user_message,
            current_phase=self.current_phase,
            conversation_history=self.conversation_history.get_all(),
            relevant_knowledge=relevant_knowledge,
            context={"user_info": self.user_info}
        )

        extracted_info = reasoning_result.get("extracted_info", {})
        if extracted_info:
            self.user_info.update(extracted_info)

        if self._is_reference_to_previous_property(user_message):
            self.user_info["refers_to"] = self.last_mentioned_property or "غير واضح"

        next_phase = reasoning_result.get("next_phase", self.current_phase)
        if next_phase and next_phase != self.current_phase:
            self.current_phase = next_phase
            self.phase_manager.set_current_phase(next_phase)

        self.context["user_info"] = self.user_info
        response = self._generate_response(user_message, reasoning_result, relevant_knowledge)
        self.conversation_history.add_assistant_message(response)
        return response, state

    def _basic_info_extraction(self, message: str):
        message_lower = message.lower()
        locations = [
            "القاهرة", "الاسكندرية", "الجيزة", "المعادي", "مدينة نصر", "6 أكتوبر", "التجمع",
            "الشروق", "العبور", "الرحاب", "مدينتي", "الشيخ زايد", "المهندسين", "الدقي",
            "الزمالك", "وسط البلد", "مصر الجديدة", "حلوان"
        ]
        for location in locations:
            if location in message and "location" not in self.user_info:
                self.user_info["location"] = location
                break

        budget_pattern = r'(\d[\d,]*)\s*(جنيه|الف|مليون|k|m)'
        match = re.search(budget_pattern, message)
        if match and "budget" not in self.user_info:
            amount = match.group(1).replace(',', '')
            unit = match.group(2)
            if unit in ['k', 'الف']:
                budget = f"{amount} ألف جنيه"
            elif unit in ['m', 'مليون']:
                budget = f"{amount} مليون جنيه"
            else:
                budget = f"{amount} جنيه"
            self.user_info["budget"] = budget

        types = {
            "شقة": ["شقة", "شقه", "apartment"],
            "فيلا": ["فيلا", "فيلات", "villa"],
            "دوبلكس": ["دوبلكس", "duplex"],
            "ستوديو": ["ستوديو", "studio"],
            "محل": ["محل", "محلات", "shop"],
            "مكتب": ["مكتب", "مكاتب", "office"]
        }
        for prop, keys in types.items():
            for k in keys:
                if k in message_lower and "property_type" not in self.user_info:
                    self.user_info["property_type"] = prop
                    break

    def _is_reference_to_previous_property(self, msg: str) -> bool:
        vague = ['هي', 'ده', 'دي', 'العقار ده', 'العرض ده']
        patterns = [rf'{word}.*(مش|ما عجبني|ما عجباني|ما عجبها)' for word in vague]
        return any(re.search(p, msg.lower()) for p in patterns)

    def _apply_rule_logic(self, user_info: dict, property_data: dict = None) -> List[str]:
        advice = []
        budget_val = user_info.get("budget", "")
        if "مليون" in budget_val:
            advice += [r["response"] for r in self.rules.get("budget_advice", []) if r["condition"] == "budget_high"]
        elif "ألف" in budget_val:
            try:
                num = int(re.findall(r'\d+', budget_val)[0])
                cond = "budget_low" if num < 500 else "budget_mid"
                advice += [r["response"] for r in self.rules.get("budget_advice", []) if r["condition"] == cond]
            except:
                pass

        features = []
        if property_data and "features" in property_data:
            features = [f.strip() for f in property_data["features"].split('،')]
        elif "features" in user_info:
            features = user_info["features"]

        for f in features:
            for rule in self.rules.get("property_priority", []):
                if rule["feature"] in f:
                    advice.append(rule["response"])

        if not advice:
            advice.append(generate_arabic_response("قدم نصيحة ذكية عن شراء عقار"))

        return advice

    def _generate_response(self, user_message, reasoning_result, relevant_knowledge={}):
        phase = self.current_phase
        user_info = self.user_info

        if phase == ConversationPhase.DISCOVERY:
            return self._discovery_response(user_info, relevant_knowledge)
        elif phase == ConversationPhase.SUMMARY:
            return self._summary_response(user_info)
        elif phase == ConversationPhase.SUGGESTION:
            return self._suggest_properties(user_info, relevant_knowledge)
        elif phase == ConversationPhase.PERSUASION:
            referred = user_info.get("refers_to", None)
            if referred and isinstance(referred, dict):
                return f"ليه مش عاجبك؟ ده فيه {referred.get('features', 'مميزات')} وموقعه في {referred.get('location', 'مكان ممتاز')}."
            return generate_arabic_response(user_message)
        elif phase == ConversationPhase.ALTERNATIVE:
            return "ممكن نعرض عليك اختيارات تانية قريبة من اللي بتحبّه."
        elif phase == ConversationPhase.URGENCY:
            return "الفرص دي مش بتستنى! تحب نكمل إجراءات المعاينة؟"
        elif phase == ConversationPhase.CLOSING:
            return "تمام، ابعتلي اسمك ورقم تليفونك وهنكلمك في أقرب وقت."
        return "أنا هنا أساعدك. تحب تبدأ بإيه؟"

    def _discovery_response(self, user_info: dict, knowledge: dict = {}) -> str:
        missing = []
        if not user_info.get("location"): missing.append("المكان")
        if not user_info.get("budget"): missing.append("الميزانية")
        if not user_info.get("property_type"): missing.append("نوع العقار")
        if missing:
            suggestions = knowledge.get("phase_knowledge", {}).get("suggested_questions", [])
            example = f"\nمثلاً: {suggestions[0]}" if suggestions else ""
            return f"ممكن تقولي {', '.join(missing)}؟ علشان أقدر أساعدك أكتر.{example}"
        return "تمام! كده أنا عرفت اللي محتاجه، نراجع المعلومات؟"

    def _summary_response(self, user_info: dict) -> str:
        parts = []
        if "location" in user_info: parts.append(f"📍 الموقع: {user_info['location']}")
        if "budget" in user_info: parts.append(f"💰 الميزانية: {user_info['budget']}")
        if "property_type" in user_info: parts.append(f"🏠 النوع: {user_info['property_type']}")
        advice = self._apply_rule_logic(user_info)
        return "دي المعلومات اللي جمعتها:\n" + "\n".join(parts) + (
            "\n\n ملاحظات:\n" + "\n".join(advice) if advice else ""
        ) + "\nهل الكلام ده مظبوط؟"

    def _suggest_properties(self, user_info: dict, knowledge: dict = {}) -> str:
        props = knowledge.get("relevant_properties", [])
        if not props:
            return generate_arabic_response("اقترح عقارات مناسبة حسب مواصفات العميل")

        self.selected_properties = props
        self.last_mentioned_property = props[0]
        reply = "🏡 العقارات دي ممكن تعجبك:\n"
        for prop in props:
            reply += f"- {prop['type']} في {prop['location']} بـ {prop['price']} جنيه\n"

        extras = self._apply_rule_logic(user_info, self.last_mentioned_property)
        if extras:
            reply += "\n ملاحظات:\n" + "\n".join(extras)
        reply += "\nهل في واحد منهم شد انتباهك؟"
        return reply
