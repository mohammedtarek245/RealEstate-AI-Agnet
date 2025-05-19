"""
Arabic NLP helper functions for the Real Estate AI Agent.
Implements Arabic-specific text processing, entity recognition, 
sentiment analysis, and other NLP functionality.
"""
import logging
import re
from typing import Dict, List, Any, Optional
import string

from config import DEBUG

# Configure logging
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO)
logger = logging.getLogger(__name__)

# Arabic stop words (common words that don't carry much meaning)
ARABIC_STOP_WORDS = [
    'من', 'إلى', 'عن', 'على', 'في', 'هذا', 'هذه', 'ذلك', 'تلك', 'هو', 'هي',
    'أنا', 'نحن', 'أنت', 'أنتم', 'هم', 'كان', 'كانت', 'يكون', 'تكون', 'و',
    'أو', 'ثم', 'بل', 'لا', 'إن', 'إذا', 'حتى', 'ف', 'قد', 'ما', 'لم',
    'لن', 'أن', 'كل', 'بعض', 'غير', 'بين', 'أمام', 'خلف', 'فوق', 'تحت'
]

def normalize_arabic_text(text: str) -> str:
    """
    Normalize Arabic text by removing diacritics, normalizing letters, etc.
    
    Args:
        text: Input Arabic text
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Replace Arabic diacritics with nothing
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    
    # Normalize alef variations to simple alef
    text = re.sub(r'[إأآا]', 'ا', text)
    
    # Normalize hamza variations
    text = re.sub(r'[ؤئ]', 'ء', text)
    
    # Normalize teh marbuta to heh
    text = text.replace('ة', 'ه')
    
    # Normalize yeh variations
    text = re.sub(r'[يى]', 'ي', text)
    
    # Remove non-Arabic characters except spaces and numbers
    arabic_pattern = r'[^\u0600-\u06FF\s0-9]'
    text = re.sub(arabic_pattern, ' ', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def detect_arabic_entities(text: str) -> Dict[str, Any]:
    """
    Detect named entities in Arabic text.
    Simple implementation without external dependencies.
    
    Args:
        text: Input Arabic text
        
    Returns:
        Dictionary of entity types and their values
    """
    entities = {}
    
    # Locations - cities and neighborhoods in Saudi Arabia
    saudi_cities = [
        'الرياض', 'جدة', 'مكة', 'المدينة', 'الدمام', 'الخبر', 'الظهران',
        'تبوك', 'أبها', 'القصيم', 'بريدة', 'نجران', 'جازان', 'حائل',
        'عسير', 'الباحة', 'الجوف', 'عرعر', 'ينبع', 'الطائف'
    ]
    
    riyadh_neighborhoods = [
        'النخيل', 'العليا', 'الملقا', 'الياسمين', 'الورود', 'الرحمانية',
        'السليمانية', 'المروج', 'الربوة', 'العزيزية', 'الملز', 'النزهة',
        'الفلاح', 'المصيف', 'الروضة', 'الشفا', 'الدرعية', 'المربع',
        'البطحاء', 'الديرة', 'الخالدية', 'النسيم'
    ]
    
    locations = []
    for city in saudi_cities:
        if city in text:
            locations.append(city)
    
    for neighborhood in riyadh_neighborhoods:
        if neighborhood in text:
            locations.append(neighborhood)
    
    if locations:
        entities['LOC'] = locations
    
    # Property types
    property_types = {
        'شقة': 'apartment',
        'فيلا': 'villa',
        'منزل': 'house',
        'دوبلكس': 'duplex',
        'استوديو': 'studio', 
        'بنتهاوس': 'penthouse',
        'محل': 'shop',
        'مكتب': 'office'
    }
    
    found_types = []
    for prop_type in property_types:
        if prop_type in text:
            found_types.append(prop_type)
    
    if found_types:
        entities['PROPERTY_TYPE'] = found_types
    
    # Money amounts
    money_pattern = r'(\d[\d,]*(?:\.\d+)?)\s*(?:ريال|الف|مليون|ألف|ر\.س)'
    money_matches = re.findall(money_pattern, text)
    
    if money_matches:
        entities['MONEY'] = money_matches
    
    # Person detection - simple implementation based on common patterns
    person_patterns = [
        r'اسمي ([\u0600-\u06FF\s]+?)(?:\.|،|$|\s)',
        r'انا ([\u0600-\u06FF\s]+?)(?:\.|،|$|\s)'
    ]
    
    for pattern in person_patterns:
        matches = re.findall(pattern, text)
        if matches:
            # Take first word only as name
            name = matches[0].split()[0]
            if len(name) > 2:  # Avoid too short names
                entities['PERSON'] = name
                break
    
    # Number detection (bedrooms, bathrooms, area, etc.)
    number_patterns = {
        'bedrooms': [r'(\d+)\s*غرف[ة]?', r'(\d+)\s*bed'],
        'bathrooms': [r'(\d+)\s*حمام', r'(\d+)\s*bath'],
        'area': [r'مساحة\s*(\d+)', r'(\d+)\s*متر', r'(\d+)\s*م']
    }
    
    for entity_type, patterns in number_patterns.items():
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                entities[entity_type.upper()] = matches[0]
                break
    
    return entities

def analyze_arabic_sentiment(text: str) -> str:
    """
    Analyze sentiment in Arabic text.
    Simple implementation without external dependencies.
    
    Args:
        text: Input Arabic text
        
    Returns:
        Sentiment label (positive, negative, neutral)
    """
    # Normalize text
    text = normalize_arabic_text(text)
    
    # Define sentiment lexicons
    positive_words = [
        'جيد', 'رائع', 'ممتاز', 'جميل', 'مناسب', 'موافق', 'أعجبني', 'أحب',
        'سعيد', 'فرح', 'حلو', 'لطيف', 'مفيد', 'ناجح', 'مريح', 'ملائم',
        'إيجابي', 'فعال', 'مثالي', 'متميز', 'نعم', 'أوافق', 'تمام', 'مهتم',
        'أرغب', 'ممكن', 'معقول', 'مربح', 'اشتري', 'أختار'
    ]
    
    negative_words = [
        'سيء', 'ردئ', 'غالي', 'بعيد', 'صغير', 'مشكلة', 'لا أحب', 'لا أريد',
        'غير مناسب', 'صعب', 'معقد', 'قبيح', 'مزعج', 'غير مريح', 'سلبي',
        'فاشل', 'ضعيف', 'باهظ', 'لا', 'غير', 'رفض', 'محبط', 'خائب', 'غاضب',
        'لا أوافق', 'لا يناسب', 'مرتفع', 'غير معقول', 'بطيء', 'متعب', 'مرهق'
    ]
    
    # Count occurrences of sentiment words
    positive_count = sum(1 for word in positive_words if word in text)
    negative_count = sum(1 for word in negative_words if word in text)
    
    # Apply negation handling - check for phrases like "not good"
    negation_words = ['لا', 'ليس', 'غير', 'ما', 'لم', 'لن']
    
    # Check for negated positives (e.g., "not good")
    for neg in negation_words:
        for pos in positive_words:
            if f"{neg} {pos}" in text or f"{neg}{pos}" in text:
                positive_count -= 1
                negative_count += 1
    
    # Check for negated negatives (e.g., "not bad")
    for neg in negation_words:
        for neg_word in negative_words:
            if f"{neg} {neg_word}" in text or f"{neg}{neg_word}" in text:
                negative_count -= 1
                positive_count += 1
    
    # Determine sentiment based on counts
    if positive_count > negative_count:
        return 'positive'
    elif negative_count > positive_count:
        return 'negative'
    else:
        return 'neutral'

def analyze_arabic_intent(text: str) -> str:
    """
    Analyze intent in Arabic text.
    Simple implementation without external dependencies.
    
    Args:
        text: Input Arabic text
        
    Returns:
        Intent label
    """
    # Normalize text
    text = normalize_arabic_text(text)
    
    # Define intent patterns
    intent_patterns = {
        'inquiry': [
            r'هل', r'كم', r'متى', r'أين', r'ما هو', r'كيف',
            r'اريد ان اعرف', r'يمكنك اخباري', r'اخبرني عن'
        ],
        'interest': [
            r'مهتم', r'أريد', r'أبحث', r'أفضل', r'يعجبني', r'ابغى', 
            r'عندي رغبة', r'عاجبني', r'حلو', r'يناسبني'
        ],
        'objection': [
            r'لكن', r'غالي', r'بعيد', r'مشكلة', r'صغير', r'كبير',
            r'لا أحب', r'لا يناسب', r'غير مناسب', r'صعب'
        ],
        'ready': [
            r'مستعد', r'موافق', r'جاهز', r'أوافق', r'نعم',
            r'تمام', r'اشتري', r'أقبل', r'أتفق', r'حاضر'
        ],
        'greeting': [
            r'مرحبا', r'السلام', r'أهلا', r'صباح', r'مساء',
            r'كيف الحال', r'اهلين', r'مرحبتين'
        ],
        'closing': [
            r'موعد', r'زيارة', r'معاينة', r'اتصال', r'تواصل',
            r'رقم', r'هاتف', r'جوال', r'ايميل', r'بريد'
        ]
    }
    
    # Check each pattern category
    for intent_type, patterns in intent_patterns.items():
        for pattern in patterns:
            if re.search(r'\b' + pattern + r'\b', text, re.IGNORECASE):
                return intent_type
    
    # Default intent if no matches
    return 'general'

def tokenize_arabic(text: str) -> List[str]:
    """
    Tokenize Arabic text into words.
    
    Args:
        text: Input Arabic text
        
    Returns:
        List of tokens
    """
    # Normalize text
    text = normalize_arabic_text(text)
    
    # Tokenize by whitespace
    tokens = text.split()
    
    # Remove punctuation and empty tokens
    cleaned_tokens = []
    arabic_punctuation = '،؛؟!.'
    all_punctuation = string.punctuation + arabic_punctuation
    
    for token in tokens:
        token = token.strip(''.join(all_punctuation))
        if token:
            cleaned_tokens.append(token)
    
    return cleaned_tokens

def remove_arabic_stop_words(tokens: List[str]) -> List[str]:
    """
    Remove Arabic stop words from a list of tokens.
    
    Args:
        tokens: List of tokens
        
    Returns:
        List of tokens with stop words removed
    """
    return [token for token in tokens if token not in ARABIC_STOP_WORDS]

def stem_arabic_word(word: str) -> str:
    """
    Apply very basic Arabic stemming.
    This is a simplified version and not a complete stemmer.
    
    Args:
        word: Arabic word
        
    Returns:
        Stemmed word
    """
    # Remove common prefixes
    prefixes = ['ال', 'و', 'ف', 'ب', 'ل', 'لل', 'وال', 'فال', 'بال', 'كال']
    for prefix in prefixes:
        if word.startswith(prefix):
            word = word[len(prefix):]
            break
    
    # Remove common suffixes
    suffixes = ['ون', 'ات', 'ين', 'ان', 'تي', 'تن', 'كن', 'هن', 'نا', 'ها', 'ية']
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:  # Ensure word won't be too short after stemming
            word = word[:-len(suffix)]
            break
    
    return word

def preprocess_arabic_text(text: str) -> List[str]:
    """
    Preprocess Arabic text: normalize, tokenize, remove stop words.
    
    Args:
        text: Input Arabic text
        
    Returns:
        List of preprocessed tokens
    """
    normalized_text = normalize_arabic_text(text)
    tokens = tokenize_arabic(normalized_text)
    tokens_without_stopwords = remove_arabic_stop_words(tokens)
    return tokens_without_stopwords
