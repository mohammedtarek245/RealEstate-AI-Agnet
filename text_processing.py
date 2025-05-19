"""
Text processing utilities for the Arabic Real Estate AI Agent.
Provides functions for extracting entities, analyzing intent and sentiment,
and other text processing capabilities.
"""
import logging
import re
from typing import Dict, List, Any

from config import DEBUG
from arabic_nlp_helpers import (
    detect_arabic_entities, 
    analyze_arabic_sentiment,
    analyze_arabic_intent,
    normalize_arabic_text
)

# Configure logging
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO)
logger = logging.getLogger(__name__)

def extract_entities(text: str) -> Dict[str, Any]:
    logger.debug(f"Extracting entities from text: {text[:50]}...")
    normalized_text = normalize_arabic_text(text)
    entities = detect_arabic_entities(normalized_text)
    if not entities:
        entities = rule_based_entity_extraction(normalized_text)
    logger.debug(f"Extracted entities: {entities}")
    return entities

def rule_based_entity_extraction(text: str) -> Dict[str, Any]:
    entities = {}
    location_patterns = [
        r'في ([\u0600-\u06FF\s]+?)(?:\.|،|$|\s\w)',
        r'منطقة ([\u0600-\u06FF\s]+?)(?:\.|،|$|\s\w)',
        r'حي ([\u0600-\u06FF\s]+?)(?:\.|،|$|\s\w)',
        r'مدينة ([\u0600-\u06FF\s]+?)(?:\.|،|$|\s\w)'
    ]
    locations = []
    for pattern in location_patterns:
        matches = re.findall(pattern, text)
        if matches:
            locations.extend([match.strip() for match in matches if len(match.strip()) > 2])
    if locations:
        entities['LOC'] = locations

    money_pattern = r'(\d[\d,]*(?:\.\d+)?)\s*(?:ريال|الف|مليون|ألف|ر\.س|جنيه)'
    money_matches = re.findall(money_pattern, text)
    if money_matches:
        entities['MONEY'] = [match for match in money_matches]

    number_pattern = r'\b(\d+)\b'
    number_matches = re.findall(number_pattern, text)
    if number_matches:
        entities['NUMBER'] = [match for match in number_matches]

    property_types = ['شقة', 'فيلا', 'منزل', 'دوبلكس', 'استوديو', 'بنتهاوس', 'مكتب', 'محل']
    found_types = [pt for pt in property_types if pt in text]
    if found_types:
        entities['PROPERTY_TYPE'] = found_types

    features = ['حديقة', 'مسبح', 'تكييف', 'مفروش', 'مطبخ', 'شرفة', 'موقف', 'جراج', 'مصعد', 'أمن']
    found_features = [f for f in features if f in text]
    if found_features:
        entities['FEATURE'] = found_features

    person_patterns = [
        r'اسمي ([\u0600-\u06FF\s]+?)(?:\.|،|$|\s\w)',
        r'أنا ([\u0600-\u06FF\s]+?)(?:\.|،|$|\s\w)'
    ]
    persons = []
    for pattern in person_patterns:
        matches = re.findall(pattern, text)
        if matches:
            persons.extend([match.strip() for match in matches if len(match.strip()) > 2])
    if persons:
        entities['PERSON'] = persons

    return entities

def analyze_intent(text: str) -> str:
    logger.debug(f"Analyzing intent of text: {text[:50]}...")
    normalized = normalize_arabic_text(text)

    intent = analyze_arabic_intent(normalized)
    if intent:
        return intent

    # Enhanced fallback rules
    intent_patterns = {
        'inquiry': [r'هل', r'كم', r'متى', r'أين', r'ما هو', r'كيف'],
        'interest': [r'مهتم', r'أريد', r'أبحث', r'أفضل', r'يعجبني'],
        'objection': [r'لكن', r'غالي', r'بعيد', r'صغير', r'لا أحب', r'مشكلة', r'مش مناسب'],
        'ready': [
            r'مستعد', r'موافق', r'جاهز', r'أوافق', r'نعم', r'تمام', r'أكيد',
            r'أيوه', r'اوكي', r'تمام كده', r'طبعاً'
        ],
        'rejection': [
            r'لا', r'مش حابب', r'مش عاجبني', r'مش مناسب', r'ما عجبنيش'
        ],
        'greeting': [r'مرحبا', r'السلام', r'أهلا', r'صباح', r'مساء']
    }

    for intent_type, patterns in intent_patterns.items():
        for pattern in patterns:
            if re.search(pattern, normalized, re.IGNORECASE):
                return intent_type

    return 'general'

def analyze_sentiment(text: str) -> str:
    logger.debug(f"Analyzing sentiment of text: {text[:50]}...")
    normalized = normalize_arabic_text(text)
    sentiment = analyze_arabic_sentiment(normalized)
    if sentiment:
        return sentiment

    positive_words = ['جيد', 'رائع', 'ممتاز', 'أحب', 'أعجبني', 'جميل', 'مناسب', 'موافق', 'تمام', 'أيوه']
    negative_words = ['سيء', 'غالي', 'بعيد', 'مشكلة', 'لا أحب', 'غير مناسب', 'مش عاجبني', 'مش حلو', 'رفض']

    positive_count = sum(1 for word in positive_words if word in normalized)
    negative_count = sum(1 for word in negative_words if word in normalized)

    if positive_count > negative_count:
        return 'positive'
    elif negative_count > positive_count:
        return 'negative'
    else:
        return 'neutral'

def extract_questions(text: str) -> List[str]:
    questions = []
    parts = text.split('؟')
    for i, part in enumerate(parts[:-1]):
        question = part.strip()
        if 3 <= len(question) <= 150:
            question_words = ['هل', 'ما', 'متى', 'أين', 'كيف', 'لماذا', 'من', 'كم']
            if any(question.startswith(word) or f' {word} ' in question for word in question_words):
                questions.append(question + '؟')
    return questions

def extract_preferences(text: str) -> Dict[str, Any]:
    preferences = {}
    entities = extract_entities(text)

    if 'LOC' in entities:
        preferences['location'] = entities['LOC'][0]
    if 'MONEY' in entities:
        preferences['budget'] = entities['MONEY'][0]
    if 'PROPERTY_TYPE' in entities:
        preferences['property_type'] = entities['PROPERTY_TYPE'][0]
    if 'NUMBER' in entities and any(x in text for x in ['غرف', 'غرفة']):
        preferences['bedrooms'] = entities['NUMBER'][0]
    if 'FEATURE' in entities:
        preferences['features'] = entities['FEATURE']

    return preferences
